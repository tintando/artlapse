"""Checks of a track that do not rely on how register decided.

Jumps: wherever the track moves the view between two samples half a second apart, the frames say
whether it moved. The earlier frame, moved as the track says, should explain the later one better
than the earlier frame left where it was. When it does not, the track probably slipped: a large
overlay walking over a still canvas, say, carried the view with it.

Agreement: frames matched directly against a canvas image in the track's coordinates (the work
directory's final canvas, or another one framed the same way), in both flip states, and compared
with where the track puts the same features. A slip shows as a stretch of samples that disagree,
starting or ending where it happened.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from . import geometry as geo
from .config import Config, clock
from .register import PATCH, PATCH_CONTRAST, PATCH_ERROR, Status, Track, sample_range, screen_mask

log = logging.getLogger(__name__)

JUMP_GAP = 1.0  # seconds: samples further apart than this are not compared
JUMP_SHIFT = 0.05  # a move of the screen's centre by this share of the screen's width is a jump
JUMP_SCALE = 0.05  # ... and so is a zoom by this much (log scale)
SUSPECT_SHARE = 0.1  # a jump is suspect where the unmoved frame explains at least this share of the textured patches, and more than the moved one

AGREE_FEATURES = 8000
AGREE_RATIO = 0.75
AGREE_MIN_INLIERS = 30
AGREE_RANSAC_PX = 4.0  # canvas px
AGREE_PX = 5.0  # screen px: farther from the direct match than this, a sample disagrees


@dataclass
class Jump:
    n: int  # the sample the view jumped at
    prev: int  # the sample before it
    shift: float  # move of the screen's centre, as a share of the screen's width
    zoom: float  # log scale change
    flip: bool  # the flip state changed
    moved: float  # share of the textured patches the earlier frame explains, moved as the track says
    held: float  # ... left where it was
    suspect: bool

    def line(self, fps: float) -> str:
        what = f"shift {self.shift:.0%} of the screen, zoom {np.exp(self.zoom):.2f}x" + (", flip" if self.flip else "")
        return (f"{clock(self.n / fps)} (sample {self.n}): {what}; the frame before explains {self.moved:.0%} "
                f"moved, {self.held:.0%} unmoved{'  <-- suspect' if self.suspect else ''}")


def explained(gray: np.ndarray, pred: np.ndarray, valid: np.ndarray, mask: np.ndarray) -> float:
    """Share of the frame's textured PATCH x PATCH neighbourhoods inside `mask` that the prediction
    (defined where `valid`) matches; register's vote judges one-sided pixels the same way."""
    k = (PATCH, PATCH)
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, k)
    textured = (cv2.boxFilter(g * g, -1, k) - mean * mean > PATCH_CONTRAST ** 2) & mask
    if textured.sum() < 100:
        return 0.0
    inside = cv2.erode(valid.astype(np.uint8), np.ones(k, np.uint8), borderValue=0) > 0
    err = cv2.boxFilter(cv2.absdiff(gray, pred).astype(np.float32), -1, k)
    return float((textured & inside & (err < PATCH_ERROR)).sum() / textured.sum())


def candidate_jumps(track: Track, size: tuple[int, int], lo: int = 0, hi: int | None = None) -> list[Jump]:
    """Pairs of placed samples at most JUMP_GAP apart, the later one matched, between which the track
    moves the view (on a screen of `size` px) by more than JUMP_SHIFT or JUMP_SCALE, or flips it."""
    hi = len(track.status) if hi is None else hi
    gap = max(1, round(JUMP_GAP * track.fps))
    width = size[0]
    centre = np.array([[size[0] / 2, size[1] / 2]])
    out, prev = [], None
    for n in range(lo, hi):
        if not track.ok(n):
            continue
        if prev is not None and n - prev <= gap and track.status[n] == Status.MATCHED:
            motion = np.linalg.inv(track.m[n]) @ track.m[prev]  # the earlier screen -> this one
            s = geo.Similarity.of(motion)
            shift = float(np.linalg.norm(geo.apply(motion, centre) - centre)) / width
            zoom = float(np.log(s.scale))
            if shift > JUMP_SHIFT or abs(zoom) > JUMP_SCALE or s.flipped:
                out.append(Jump(n, prev, shift, zoom, s.flipped, 0.0, 0.0, False))
        prev = n
    return out


def jumps(cfg: Config, track: Track, lo: int | None = None, hi: int | None = None) -> list[Jump]:
    """The track's jumps (candidate_jumps), each judged by the frames on both sides of it."""
    from .video import Sampler

    sampler = Sampler(cfg.video, cfg.sample_fps, width=cfg.register.analysis_width, gray=True, cache=True)
    info = sampler.info
    r0, r1 = sample_range(cfg, sampler.count)
    found = candidate_jumps(track, (info.width, info.height), r0 if lo is None else lo, min(r1, len(track.status)) if hi is None else hi)
    if not found:
        return []
    r = cfg.register.analysis_width / info.width
    to_a = geo.scale_translate(r)
    frames = dict(sampler.select(sorted({j.n for j in found} | {j.prev for j in found})))
    some = next(iter(frames.values()))
    h, w = some.shape
    mask = screen_mask(cfg, w, h, r)
    mask_bool = mask > 0
    for j in found:
        before, after = frames[j.prev], frames[j.n]
        motion = to_a @ np.linalg.inv(track.m[j.n]) @ track.m[j.prev] @ np.linalg.inv(to_a)  # analysis px
        moved = cv2.warpAffine(before, motion[:2], (w, h), flags=cv2.INTER_LINEAR)
        valid = mask_bool & (cv2.warpAffine(mask, motion[:2], (w, h), flags=cv2.INTER_NEAREST) > 0)
        j.moved = explained(after, moved, valid, mask_bool)
        j.held = explained(after, before, mask_bool, mask_bool)
        j.suspect = j.held >= SUSPECT_SHARE and j.held > j.moved
    return found


@dataclass
class Agreement:
    n: int
    flip: bool  # the flip state the frame matched the canvas in
    inliers: int
    track: str  # the track's status
    error: float | None  # median distance between where the track and the canvas put the matched features, screen px
    error_canvas: float | None  # ... in canvas px

    def disagrees(self) -> bool:
        return self.error is not None and self.error > AGREE_PX


def agreement(cfg: Config, track: Track, canvas: np.ndarray, samples: Iterable[int]) -> list[Agreement]:
    """Each sample's frame matched directly against `canvas` (BGR, in the track's canvas coordinates)."""
    from .video import Sampler, probe

    info = probe(cfg.video)
    mask = screen_mask(cfg, info.width, info.height, 1.0)
    mirror = geo.mirror_x(info.width)
    sift = cv2.SIFT_create(nfeatures=AGREE_FEATURES)
    kc, dc = sift.detectAndCompute(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), None)
    if dc is None:
        raise ValueError("the canvas has no features")
    pc = np.float32([k.pt for k in kc])
    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 64})
    out = []
    for n, frame in Sampler(cfg.video, cfg.sample_fps, cache=True).select(sorted(samples)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best = None
        for flip in (False, True):
            kf, df = sift.detectAndCompute(cv2.flip(gray, 1) if flip else gray, cv2.flip(mask, 1) if flip else mask)
            if df is None or len(kf) < 2:
                continue
            good = [a for a, b in (p for p in matcher.knnMatch(df, dc, k=2) if len(p) == 2) if a.distance < AGREE_RATIO * b.distance]
            if len(good) < AGREE_MIN_INLIERS:
                continue
            seen = np.float32([kf[g.queryIdx].pt for g in good])  # in the (maybe mirrored) frame
            at = pc[[g.trainIdx for g in good]]
            a, inl = cv2.estimateAffinePartial2D(seen, at, method=cv2.RANSAC, ransacReprojThreshold=AGREE_RANSAC_PX,
                                                 maxIters=5000, confidence=0.999)
            if a is not None and (best is None or int(inl.sum()) > best[1]):
                screen = geo.apply(mirror, seen) if flip else seen.astype(float)
                inl = inl.ravel().astype(bool)
                best = (flip, int(inl.sum()), screen[inl], at[inl])
        status = Status(track.status[n]).name.lower() if n < len(track.status) else "unseen"
        if best is None or best[1] < AGREE_MIN_INLIERS:
            continue
        flip, n_inl, screen, at = best
        err = err_c = None
        if track.ok(n):
            err_c = float(np.median(np.linalg.norm(geo.apply(track.m[n], screen) - at, axis=1)))
            err = err_c / geo.Similarity.of(track.m[n]).scale
        out.append(Agreement(n, flip, n_inl, status, err, err_c))
    return out


def stretches(found: list[Agreement]) -> list[tuple[int, int, int]]:
    """Runs of consecutive checked samples that disagree with the canvas: (first, last, how many)."""
    runs, cur = [], None
    for a in found:
        if a.error is None:
            continue
        if a.disagrees():
            cur = [a.n, a.n, 1] if cur is None else [cur[0], a.n, cur[2] + 1]
        elif cur is not None:
            runs.append(tuple(cur))
            cur = None
    if cur is not None:
        runs.append(tuple(cur))
    return runs


def report(cfg: Config, found_jumps: list[Jump] | None, found: list[Agreement] | None) -> dict:
    """The checks as a dict for check.json, logging a summary."""
    fps = cfg.sample_fps
    out: dict = {}
    if found_jumps is not None:
        suspect = [j for j in found_jumps if j.suspect]
        log.info("jumps: %d, suspect %d", len(found_jumps), len(suspect))
        for j in suspect:
            log.info("  %s", j.line(fps))
        out["jumps"] = [asdict(j) | {"time": clock(j.n / fps)} for j in found_jumps]
    if found is not None:
        errs = [a.error for a in found if a.error is not None]
        runs = stretches(found)
        log.info("agreement with the canvas: %d samples matched it, %d of them placed by the track; median %.1f screen px, "
                 "%d farther than %.0f px", len(found), len(errs), float(np.median(errs)) if errs else float("nan"),
                 sum(e > AGREE_PX for e in errs), AGREE_PX)
        if missed := [a for a in found if a.error is None]:
            log.info("  the track lost %d of them: %s", len(missed), ", ".join(clock(a.n / fps) for a in missed[:20]))
        for first, last, count in runs:
            log.info("  disagrees %s to %s (%d samples checked)", clock(first / fps), clock(last / fps), count)
        out["agreement"] = [asdict(a) | {"time": clock(a.n / fps)} for a in found]
        out["stretches"] = [{"from": clock(f / fps), "to": clock(t / fps), "first": f, "last": t, "count": c} for f, t, c in runs]
    return out
