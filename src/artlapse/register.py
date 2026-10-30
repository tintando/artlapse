"""Registration: for every sampled frame, find where the canvas sits on screen.

Each frame is matched against a feature map kept in canvas coordinates. The map starts
from the reference frame and grows with every keyframe. Matching against the map, not
against the previous frame, avoids drift. It also lets tracking recover after a scene
change or another document, since those frames simply fail to match.

Flips: descriptors in the map always describe the canvas the right way round. A frame
whose view is mirrored is mirrored before feature extraction. The fitted transform is
then an ordinary similarity composed with that mirror.
"""
from __future__ import annotations

import enum
import logging
from collections.abc import Callable, Iterable
from itertools import chain
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import features
from . import geometry as geo
from .config import Config, RegisterCfg

log = logging.getLogger(__name__)

READ_AHEAD = 240  # analysis frames decoded ahead of matching on the GPU (about 125 MB at 960 px)
PATCH = 7  # analysis pixels: the neighbourhood a pixel only one view predicts is judged by
PATCH_CONTRAST = 16.0  # grey standard deviation above which a patch has texture
PATCH_ERROR = 8.0  # mean grey error under which a prediction matches a patch


class Status(enum.IntEnum):
    UNSEEN = 0  # not processed
    MATCHED = 1  # transform from feature matching
    STATIC = 2  # frame unchanged since the last match; transform carried over
    LOST = 3  # no convincing match: scene change, other document, heavy occlusion


@dataclass
class Track:
    """Per-sample registration results. `m[n]` maps full-resolution screen pixels to canvas pixels."""

    fps: float
    status: np.ndarray  # (N,) uint8 Status
    m: np.ndarray  # (N, 3, 3) float64
    inliers: np.ndarray  # (N,) int32

    @staticmethod
    def empty(count: int, fps: float) -> "Track":
        return Track(fps, np.zeros(count, np.uint8), np.tile(np.eye(3), (count, 1, 1)), np.zeros(count, np.int32))

    def ok(self, n: int) -> bool:
        return self.status[n] in (Status.MATCHED, Status.STATIC)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, fps=self.fps, status=self.status, m=self.m, inliers=self.inliers)

    @staticmethod
    def load(path: Path) -> "Track":
        d = np.load(path)
        return Track(float(d["fps"]), d["status"], d["m"], d["inliers"])


def screen_mask(cfg: Config, width: int, height: int, scale: float) -> np.ndarray:
    """255 where the art program's canvas view may be seen, at `scale` x full resolution."""
    mask = np.zeros((height, width), np.uint8)

    def fill(rect, value):
        x, y, w, h = (round(v * scale) for v in rect)
        mask[max(y, 0):max(y + h, 0), max(x, 0):max(x + w, 0)] = value

    fill(cfg.workspace, 255)
    for r in cfg.exclude:
        fill(r, 0)
    return mask


def reference_transform(cfg: Config) -> tuple[np.ndarray, tuple[int, int]]:
    """Screen->canvas transform of the reference frame, and the canvas size (w, h), both even."""
    x, y, w, h = cfg.reference.target
    if w <= 0 or h <= 0:
        raise ValueError("reference.target must be a non-empty rectangle")
    k = cfg.canvas.long_side / max(w, h)
    size = (2 * round(w * k / 2), 2 * round(h * k / 2))
    return geo.scale_translate(k, -x * k, -y * k), size


class FeatureMap:
    """Canvas-space SIFT features holding the latest appearance of every canvas area.

    Features are kept at most one per grid cell per zoom level. A keyframe first removes the
    features it should have seen again (inside its view, at its own or coarser levels), then
    adds its own. Content the artist moved, resized or painted over therefore disappears from
    the map instead of lingering as a second, conflicting copy.
    """

    def __init__(self):
        self.slots: dict[tuple[int, int, int], int] = {}
        self.pts = np.zeros((0, 2), np.float32)
        self.desc = np.zeros((0, 128), np.float32)
        self.level = np.zeros(0, np.int16)
        self.alive = np.zeros(0, bool)
        self._matcher: cv2.FlannBasedMatcher | None = None
        self._index_rows = np.zeros(0, int)  # map row of each indexed descriptor

    def __len__(self) -> int:
        return int(self.alive.sum())

    def add(self, canvas_pts: np.ndarray, desc: np.ndarray, cell: float,
            seen: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        """Insert features observed at grid size `cell` (canvas units). `seen(points)` tells which
        existing canvas points the observing frame covered; those are replaced."""
        level = int(round(np.log2(cell)))
        if seen is not None and self.alive.any():
            rows = np.nonzero(self.alive & (self.level >= level - 1))[0]
            stale = rows[seen(self.pts[rows])]
            for r in stale:
                p = self.pts[r]
                size = 2.0 ** self.level[r]
                self.slots.pop((int(self.level[r]), int(p[0] // size), int(p[1] // size)), None)
            self.alive[stale] = False
        size = 2.0 ** level
        new_pts, new_desc = [], []
        n0 = len(self.pts)
        for p, d in zip(canvas_pts, desc):
            key = (level, int(p[0] // size), int(p[1] // size))
            slot = self.slots.get(key)
            if slot is None:
                self.slots[key] = n0 + len(new_pts)
                new_pts.append(p)
                new_desc.append(d)
            elif slot >= n0:  # cell first filled earlier in this same batch
                new_pts[slot - n0] = p
                new_desc[slot - n0] = d
            else:
                self.pts[slot] = p
                self.desc[slot] = d
        if new_pts:
            self.pts = np.vstack([self.pts, np.float32(new_pts)])
            self.desc = np.vstack([self.desc, np.float32(new_desc)])
            self.level = np.concatenate([self.level, np.full(len(new_pts), level, np.int16)])
            self.alive = np.concatenate([self.alive, np.ones(len(new_pts), bool)])
        if len(self.alive) > 2 * max(len(self), 1000):
            self._compact()
        self._rebuild()

    def _compact(self) -> None:
        keep = np.nonzero(self.alive)[0]
        remap = {int(old): new for new, old in enumerate(keep)}
        self.slots = {k: remap[v] for k, v in self.slots.items() if v in remap}
        self.pts, self.desc, self.level = self.pts[keep], self.desc[keep], self.level[keep]
        self.alive = np.ones(len(keep), bool)

    def _rebuild(self) -> None:
        self._index_rows = np.nonzero(self.alive)[0]
        if len(self._index_rows) < 2:
            self._matcher = None
            return
        self._matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
        self._matcher.add([self.desc[self._index_rows]])
        self._matcher.train()

    def match(self, desc: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
        """Ratio-test matches: (query indices, canvas points)."""
        if self._matcher is None or desc is None or len(desc) < 2:
            return np.zeros(0, int), np.zeros((0, 2), np.float32)
        qi, pts = [], []
        for m in self._matcher.knnMatch(desc, k=2):
            if len(m) < 2:
                continue
            best, second = m
            pb = self.pts[self._index_rows[best.trainIdx]]
            ps = self.pts[self._index_rows[second.trainIdx]]
            # the same canvas location seen at two zoom levels is not ambiguous
            if best.distance < ratio * second.distance or np.linalg.norm(pb - ps) < 4.0:
                qi.append(best.queryIdx)
                pts.append(pb)
        return np.array(qi, int), np.float32(pts).reshape(-1, 2)


@dataclass
class _Frame:
    m: np.ndarray  # screen(full) -> canvas
    gray: np.ndarray  # analysis frame


class Registrar:
    def __init__(self, cfg: Config, full_width: int, full_height: int, device: str = "cpu"):
        self.cfg = cfg
        rc: RegisterCfg = cfg.register
        self.rc = rc
        self.r = rc.analysis_width / full_width  # analysis px per full px
        self.aw = rc.analysis_width
        self.ah = round(full_height * self.r / 2) * 2
        self.mask = screen_mask(cfg, self.aw, self.ah, self.r)
        self.mask_flipped = cv2.flip(self.mask, 1)
        self.mask_bool = self.mask > 0
        self.mirror = geo.mirror_x(self.aw)
        self.sift = features.backend(device, rc.max_features)
        self._feat_gray: np.ndarray | None = None  # the frame whose features are in _feat
        self._feat: dict[bool, tuple[np.ndarray, np.ndarray | None]] = {}
        self._tex_gray: np.ndarray | None = None  # the frame whose patch contrast is in _tex
        self._tex: np.ndarray | None = None
        self.map = FeatureMap()
        self.flipped = False
        self.last: _Frame | None = None  # last matched frame, cleared when tracking is lost
        self.anchor: _Frame | None = None  # last matched frame, kept across lost stretches
        self.last_key: _Frame | None = None  # last frame added to the map
        self.decision: dict | None = None  # how step() chose, for a frame it did not find unmoved

    # analysis-pixel <-> full-pixel helpers
    def _a2c(self, m_full: np.ndarray) -> np.ndarray:
        return m_full @ geo.scale_translate(1 / self.r)

    def _c2full(self, m_a: np.ndarray) -> np.ndarray:
        return m_a @ geo.scale_translate(self.r)

    def _features(self, gray: np.ndarray, flipped: bool) -> tuple[np.ndarray, np.ndarray | None]:
        """Keypoints in (possibly mirrored) analysis coordinates, and their descriptors. Kept for the
        latest frame, which a keyframe adds to the map after matching it."""
        if gray is not self._feat_gray or flipped not in self._feat:
            self._detect(gray, [flipped])
        return self._feat[flipped]

    def _detect(self, gray: np.ndarray, states: list[bool]) -> None:
        """Extract features of `gray` in the given flip states, in one call to the backend."""
        if gray is not self._feat_gray:
            self._feat_gray, self._feat = gray, {}
        images = [(cv2.flip(gray, 1), self.mask_flipped) if f else (gray, self.mask) for f in states]
        self._feat.update(zip(states, self.sift.detect(images)))

    def seed(self, gray: np.ndarray, m_full: np.ndarray) -> None:
        """Start (or restart) from a frame with a known transform, e.g. the reference frame."""
        self.map = FeatureMap()
        self.flipped = geo.Similarity.of(m_full).flipped
        frame = _Frame(m_full, gray)
        self._add_keyframe(frame)
        self.last = self.anchor = frame

    def _add_keyframe(self, frame: _Frame) -> None:
        flipped = geo.Similarity.of(frame.m).flipped
        pts, desc = self._features(frame.gray, flipped)
        if desc is None or not len(pts):
            return
        m_a = self._a2c(frame.m)  # analysis(unmirrored) -> canvas
        if flipped:
            m_a = m_a @ self.mirror  # mirrored analysis -> canvas
        canvas_pts = geo.apply(m_a, pts)
        cell = self.rc.map_cell_px * geo.Similarity.of(m_a).scale
        to_screen = np.linalg.inv(self._a2c(frame.m))  # canvas -> analysis (unmirrored)

        def seen(cpts: np.ndarray) -> np.ndarray:
            xy = np.round(geo.apply(to_screen, cpts)).astype(int)
            inside = (xy[:, 0] >= 0) & (xy[:, 0] < self.aw) & (xy[:, 1] >= 0) & (xy[:, 1] < self.ah)
            out = np.zeros(len(cpts), bool)
            out[inside] = self.mask_bool[xy[inside, 1], xy[inside, 0]]
            return out

        self.map.add(canvas_pts, desc, cell, seen)
        self.last_key = frame

    def _fit(self, src_canvas: np.ndarray, dst_screen: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        """Similarity canvas -> screen(analysis, maybe mirrored) by RANSAC; returns (3x3, inlier mask)."""
        if len(src_canvas) < 3:
            return None, np.zeros(len(src_canvas), bool)
        a, inl = cv2.estimateAffinePartial2D(
            src_canvas, dst_screen, method=cv2.RANSAC, ransacReprojThreshold=self.rc.ransac_px,
            maxIters=4000, confidence=0.995, refineIters=20)
        if a is None:
            return None, np.zeros(len(src_canvas), bool)
        return geo.to3(a), inl.ravel().astype(bool)

    def _inliers_of(self, c2s: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        err = np.linalg.norm(geo.apply(c2s, src) - dst, axis=1)
        return err < self.rc.ransac_px

    def _predict(self, m: np.ndarray, ref: _Frame) -> tuple[np.ndarray, np.ndarray]:
        """The earlier matched frame `ref` moved by the view change that `m` (screen(full) -> canvas)
        implies, as this frame would show it, and where that prediction is defined."""
        rel = np.linalg.inv(self._a2c(ref.m)) @ self._a2c(m)  # this frame's analysis px -> ref's
        size = (self.aw, self.ah)
        flags = cv2.WARP_INVERSE_MAP
        moved = cv2.warpAffine(ref.gray, rel[:2], size, flags=flags | cv2.INTER_LINEAR)
        return moved, self.mask_bool & (cv2.warpAffine(self.mask, rel[:2], size, flags=flags | cv2.INTER_NEAREST) > 0)

    def _compare(self, gray: np.ndarray, a: np.ndarray, b: np.ndarray, ref: _Frame) -> tuple[int, int] | None:
        """How many pixels of this frame views `a` and `b` each explain better. Only pixels where their
        predictions disagree and at least one is right count, so what both explain equally, or neither
        does (an avatar passing, a new stroke, the symmetric parts of a mirrored drawing), stays out of
        it. A pixel only one view predicts counts for it where it is right and the frame has texture
        there, which a view that leaves it unexplained (as new canvas) would match only by chance: a
        large overlay moving over a still canvas otherwise outvotes the canvas, whose unchanged part
        the overlay's view does not predict at all. None when there is too little to tell; a view
        sharing next to nothing with `ref` to predict anything gets no pixels. A zoom out from a
        close-up shares only a small part, which still counts."""
        pa, va = self._predict(a, ref)
        pb, vb = self._predict(b, ref)
        enough = 0.02 * self.mask_bool.sum()
        if va.sum() < enough or vb.sum() < enough:
            return int(va.sum() >= enough), int(vb.sum() >= enough)
        ea, eb = cv2.absdiff(gray, pa).astype(np.int16), cv2.absdiff(gray, pb).astype(np.int16)
        decisive = va & vb & (cv2.absdiff(pa, pb) > 16) & (np.minimum(ea, eb) < 16)
        wa = int((decisive & (ea + 8 < eb)).sum()) + int((va & ~vb & self._matched_patches(gray, ea, va)).sum())
        wb = int((decisive & (eb + 8 < ea)).sum()) + int((vb & ~va & self._matched_patches(gray, eb, vb)).sum())
        return None if wa + wb < 50 else (wa, wb)

    def _matched_patches(self, gray: np.ndarray, err: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Pixels whose PATCH x PATCH neighbourhood has contrast and is matched by a prediction with
        errors `err`, all of it inside where the prediction is defined (`valid`). Single pixels match by
        chance (the edge of a line has the value of the paper beside it); a patch with texture does not."""
        k = (PATCH, PATCH)
        if gray is not self._tex_gray:
            g = gray.astype(np.float32)
            mean = cv2.boxFilter(g, -1, k)
            self._tex_gray, self._tex = gray, cv2.boxFilter(g * g, -1, k) - mean * mean > PATCH_CONTRAST ** 2
        inside = cv2.erode(valid.astype(np.uint8), np.ones(k, np.uint8), borderValue=0) > 0
        return inside & self._tex & (cv2.boxFilter(err.astype(np.float32), -1, k) < PATCH_ERROR)

    def _try(self, gray: np.ndarray, flipped: bool) -> tuple[list[tuple[np.ndarray, int]], int]:
        """Match against the map in one flip state. Returns the convincing views as (screen(full)->canvas,
        inliers), the one to prefer when nothing else decides first, and the number of matches."""
        pts, desc = self._features(gray, flipped)
        if desc is None:
            return [], 0
        qi, canvas = self.map.match(desc, self.rc.ratio)
        if len(qi) < self.rc.min_inliers:
            return [], len(qi)
        screen = pts[qi]
        c2s, inl = self._fit(canvas, screen)
        if c2s is None:
            return [], len(qi)
        # The matches may support two views at once: the canvas moved, while overlays fixed on screen
        # still match where they were; or the view stayed, while the artist transformed a layer (or
        # a second view shows the same art). Feature counts cannot tell these apart: after a big zoom
        # few canvas features survive, and a chat box or a toolbar can outnumber them. All of them
        # are returned, and the pixels decide in step().
        candidates = [c2s]
        rest = ~inl
        if rest.sum() >= self.rc.min_inliers:
            second, inl2 = self._fit(canvas[rest], screen[rest])
            if second is not None and inl2.sum() >= self.rc.min_inliers:
                candidates.append(second)
        if self.anchor is not None and geo.Similarity.of(self.anchor.m).flipped == flipped:
            prev = self._a2c(self.anchor.m)
            if flipped:
                prev = prev @ self.mirror
            prev_inl = self._inliers_of(np.linalg.inv(prev), canvas, screen)
            # while the previous view still explains a solid share of the matches, spread across the
            # view (an overlay that stays put supports it only from one compact spot), prefer it
            if (prev_inl.sum() >= max(self.rc.min_inliers, self.rc.keep_view_ratio * inl.sum())
                    and _hull_area(screen[prev_inl]) >= self.rc.keep_view_spread * _hull_area(screen)):
                keep, _ = self._fit(canvas[prev_inl], screen[prev_inl])
                if keep is not None:
                    candidates.insert(0, keep)
        views = []
        for c in candidates:
            n_inl = int(self._inliers_of(c, canvas, screen).sum())
            if n_inl < self.rc.min_inliers or n_inl < self.rc.min_inlier_ratio * len(qi):
                continue
            if not geo.is_similarity(c, tol=1e-2):
                continue
            m_a = np.linalg.inv(c)  # screen(analysis, maybe mirrored) -> canvas
            if flipped:
                m_a = m_a @ self.mirror
            views.append((self._c2full(m_a), n_inl))
        return views, len(qi)

    def step(self, gray: np.ndarray) -> tuple[Status, np.ndarray | None, int]:
        self.decision = None
        if self.last is not None:
            diff = cv2.absdiff(gray, self.last.gray)
            if float(diff[self.mask_bool].mean()) < self.rc.static_diff:
                return Status.STATIC, self.last.m, 0
        views, matches = [], []
        self._detect(gray, [self.flipped, not self.flipped])
        for flipped in (self.flipped, not self.flipped):
            found, n_matches = self._try(gray, flipped)
            views += [(m, k, flipped) for m, k in found]
            matches.append(n_matches)
        d = self.decision = {"matches": matches, "map": len(self.map), "views": [self._describe(v[0], v[1]) for v in views],
                             "ref": "last" if self.last is not None else "anchor" if self.anchor is not None else None,
                             "votes": []}
        if not views:
            self.last = None  # forget the old view: after a scene change anything goes
            return Status.LOST, None, 0
        # The pixels decide: the right view turns the previous frame into this one, a wrong one only
        # overlays, a transformed layer or the mirror image of a symmetric drawing. While tracking,
        # "the view held" competes too, so a frame mostly covered by an avatar or a popup cannot
        # switch the view on the strength of a few features. After a lost stretch the last matched
        # frame stands in, and without one the most supported view wins.
        ref = self.last or self.anchor
        if ref is None:
            views.sort(key=lambda v: -v[1])
        held = self.last is not None
        best = (self.last.m, 0, self.flipped) if held else views[0]
        won = "held" if held else 0  # the view chosen, as an index into d["views"]
        for i, v in enumerate(views):
            if ref is None or v is best:
                continue
            cmp = self._compare(gray, v[0], best[0], ref)
            holding = best[0] is self.last.m if held else False
            if cmp is None:
                replace = holding  # the same view: take the fresh estimate, it keeps the map current
            else:
                replace = cmp[0] > cmp[1]
            d["votes"].append([i, won, *(cmp or [None, None]), bool(replace)])
            if replace:
                best, won = v, i
        d["won"] = won
        m, n_inl, self.flipped = best
        if self.last is not None and m is self.last.m:
            return Status.STATIC, m, 0
        frame = _Frame(m, gray)
        d["keyframe"] = self._is_keyframe(frame)
        if d["keyframe"]:
            self._add_keyframe(frame)
        self.last = self.anchor = frame
        return Status.MATCHED, m, n_inl

    def _describe(self, m: np.ndarray, n_inl: int) -> dict:
        """A view for the decision log: inliers, flip, zoom (canvas px per screen px), rotation, and where
        it puts the centre of the screen on the canvas."""
        s = geo.Similarity.of(m)
        cx, cy = geo.apply(m, [[self.aw / self.r / 2, self.ah / self.r / 2]])[0]
        return {"inliers": n_inl, "flip": s.flipped, "scale": round(s.scale, 4), "angle": round(s.angle_deg, 2),
                "centre": [round(float(cx), 1), round(float(cy), 1)]}

    def _is_keyframe(self, frame: _Frame) -> bool:
        if self.last_key is None:
            return True
        ds, shift = geo.view_change(self.last_key.m, frame.m)
        canvas_per_px = geo.Similarity.of(frame.m).scale
        if ds > 0.05 or shift > 20 * canvas_per_px:
            return True
        if geo.Similarity.of(frame.m).flipped != geo.Similarity.of(self.last_key.m).flipped:
            return True
        # same view: add when enough has been painted since
        diff = cv2.absdiff(frame.gray, self.last_key.gray)
        return float((diff[self.mask_bool] > 24).mean()) > 0.02


def _hull_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(np.float32(pts))))


def register(cfg: Config, progress: Callable[[int, int, Status, int], None] | None = None,
             device: str = "cpu", decisions: Callable[[int, dict], None] | None = None) -> Track:
    """Register every sample of the video: backward from the reference frame, then forward.
    `decisions(n, record)` receives how each sample not found unmoved was decided (Registrar.step)."""
    from .video import Sampler, prefetch

    sampler = Sampler(cfg.video, cfg.sample_fps, width=cfg.register.analysis_width, gray=True, cache=True)
    info = sampler.info
    track = Track.empty(sampler.count, cfg.sample_fps)
    m_ref, _ = reference_transform(cfg)
    spans = sample_spans(cfg, sampler.count)
    ref = round(cfg.reference.time * cfg.sample_fps)
    if not any(a <= ref < b for a, b in spans):
        raise ValueError("reference.time must lie between start and end, outside the cuts")
    total = sum(b - a for a, b in spans)
    ref_gray = sampler.frame(ref)

    reg = Registrar(cfg, info.width, info.height, device)
    track.status[ref], track.m[ref] = Status.MATCHED, m_ref

    # FLANN's randomised trees and RANSAC draw from OpenCV's RNG. Seeded by the sample, a sample's result
    # does not depend on how many draws came before it: not on what ran earlier in the process, nor on
    # the other pass, so a run over part of the range reproduces that part
    def run(frames: Iterable[tuple[int, np.ndarray]]) -> None:
        cv2.setRNGSeed(ref)
        reg.seed(ref_gray, m_ref)
        for n, gray in frames:
            cv2.setRNGSeed(n)
            status, m, n_inl = reg.step(gray)
            if decisions and reg.decision is not None:
                decisions(n, {"status": status.name.lower(), **reg.decision})
            track.status[n] = status
            track.inliers[n] = n_inl
            if m is not None:
                track.m[n] = m
            if progress:
                progress(n, total, status, n_inl)

    # a cut is passed over like a stretch where the view was lost
    backward = chain.from_iterable(sampler.frames_backward(min(b, ref), a, chunk=READ_AHEAD) for a, b in reversed(spans) if a < ref)
    forward = chain.from_iterable(sampler.frames(max(a, ref + 1), b) for a, b in spans if b > ref + 1)
    if device == "cuda":
        # decode the next stretch while this one is matched; the depth covers a whole backward chunk.
        # OpenCV's SIFT spreads each call over every core, and a decoder running alongside stalls it
        backward, forward = prefetch(backward, depth=READ_AHEAD), prefetch(forward, depth=READ_AHEAD)
    run(backward)
    run(forward)
    return track


def sample_range(cfg: Config, count: int) -> tuple[int, int]:
    lo = max(0, round(cfg.start * cfg.sample_fps))
    hi = count if cfg.end is None else min(count, round(cfg.end * cfg.sample_fps))
    return lo, hi


def sample_spans(cfg: Config, count: int) -> list[tuple[int, int]]:
    """The stretches [a, b) of sample indices that are processed: start to end, without the cuts."""
    lo, hi = sample_range(cfg, count)
    spans = []
    for c0, c1 in sorted((round(a * cfg.sample_fps), round(b * cfg.sample_fps)) for a, b in cfg.cuts):
        if c0 > lo:
            spans.append((lo, min(c0, hi)))
        lo = max(lo, c1)
        if lo >= hi:
            break
    if lo < hi:
        spans.append((lo, hi))
    return [(a, b) for a, b in spans if a < b]


def kept_samples(cfg: Config, index: np.ndarray) -> np.ndarray:
    """How many processed samples (sample_spans) come before each sample index: its stream time with the cuts left out."""
    spans = sample_spans(cfg, int(np.max(index, initial=0)) + 1)
    return sum(np.clip(index - a, 0, b - a) for a, b in spans) if spans else np.zeros_like(index)
