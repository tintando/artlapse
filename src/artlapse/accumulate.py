"""Compositing: rebuild the canvas from registered frames.

Each canvas pixel keeps its committed colour C and the sharpness Q of the view it came
from (screen pixels per canvas pixel). A frame may only change a pixel after it has shown the
same new value for `stable_seconds`. Strokes stay for minutes and pass, while cursors,
alerts and raid animations leave within seconds. The delay is invisible in a timelapse,
where a second covers minutes of stream. A frame that agrees with the canvas
but shows it more zoomed in replaces the pixel with the sharper observation.

A view further out shows unchanged detail softer, differing from it by about as much as a faint
change does, so faint changes come only from views at least as sharp as the pixel's, with one
exception: where the strokes' ink changed. Blur spreads a stroke's ink but keeps it, so compared at
the view's own scale an unchanged stroke agrees, while a faded sketch layer or a removed stroke loses
ink along its whole length.

Most samples during painting change only a stroke's worth of pixels. While the view stays
put, only the canvas tiles behind screen tiles that changed since they were last processed
are updated, along with tiles that still hold uncommitted changes. Slow changes (a fading
layer, an opacity slider) therefore still update a whole area at once. When the view
moves, everything visible is updated.

A canvas can start from an earlier part's final canvas (`seed`). Those pixels count as seen, but less
sharply than from any view, so the first view that agrees with one replaces it at once, and one that
differs changes it after `stable_seconds`, as on a canvas seen before.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from . import geometry as geo
from .config import AccumulateCfg

TILE = 128  # canvas pixels
SCREEN_TILE = 32  # screen pixels, for locating changed areas
MARGIN = 8  # canvas pixels of context computed around each updated box, so boxes join seamlessly
INK_SIGMA = 2.0  # screen pixels: the scale at which a view's strokes are compared with sharper canvas pixels
INK_GROW = 1.0  # screen pixels: how far a change in ink is grown back over the stroke
SEEDED = 1e-3  # the quality of pixels from an earlier part's canvas: known, blurrier than any view


@dataclass
class FrameResult:
    changed: int  # canvas pixels committed by this frame
    processed: int  # canvas pixels examined


class Canvas:
    incremental = True  # the Compositor may update only the tiles that changed

    def __init__(self, size: tuple[int, int], cfg: AccumulateCfg, background: tuple[int, int, int], sample_fps: float):
        w, h = size
        self.size = size
        self.cfg = cfg
        self.stable = int(np.clip(round(cfg.stable_seconds * sample_fps), 1, 255))  # samples
        self.background = np.array(background, np.uint8)
        self.color = np.empty((h, w, 3), np.uint8)
        self.color[:] = self.background  # unseen pixels (quality 0) show the background
        self.quality = np.zeros((h, w), np.float32)  # 0 = never seen
        self.pending = np.zeros((h, w, 3), np.uint8)
        self.count = np.zeros((h, w), np.uint8)
        # boxes are disjoint, and OpenCV/numpy release the GIL, so boxes update in parallel
        self.pool = ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2))

    def image(self) -> np.ndarray:
        return self.color.copy()

    def seed(self, image: np.ndarray, known: np.ndarray) -> None:
        """Start from `image` where `known` (an earlier part's final canvas, see continuation.py)."""
        self.color[known] = image[known]
        self.quality[known] = SEEDED

    def visible_box(self, valid: np.ndarray, m: np.ndarray) -> tuple[int, int, int, int] | None:
        """Canvas bounding box (x0, y0, x1, y1) of the valid screen region under `m`."""
        ys, xs = np.nonzero(valid[::8, ::8])
        if not len(xs):
            return None
        corners = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.min(), ys.max()], [xs.max(), ys.max()]], float) * 8
        return self._clip(geo.apply(m, corners), 8 * geo.Similarity.of(m).scale + 8)

    def _clip(self, pts: np.ndarray, pad: float) -> tuple[int, int, int, int] | None:
        w, h = self.size
        x0, y0 = np.floor(pts.min(axis=0) - pad).astype(int)
        x1, y1 = np.ceil(pts.max(axis=0) + pad).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    def add(self, frame: np.ndarray, valid: np.ndarray, m: np.ndarray,
            boxes: list[tuple[int, int, int, int]] | None = None) -> FrameResult:
        """Merge a full-resolution BGR frame. `valid` is the screen mask (uint8), `m` maps screen -> canvas.
        Only the canvas `boxes` are updated; None means everything the frame shows."""
        if boxes is None:
            box = self.visible_box(valid, m)
            boxes = _merge_tiles([box], self.size) if box else []
        scale = geo.Similarity.of(m).scale  # canvas px per screen px
        src = frame
        if scale < 0.6:  # zoomed in further than the canvas resolution: low-pass before shrinking
            src = cv2.GaussianBlur(frame, (0, 0), 0.4 / scale)
        res = FrameResult(0, 0)
        for r in self.pool.map(lambda b: self._update(src, valid, m, scale, b), boxes):
            res.changed += r.changed
            res.processed += r.processed
        return res

    def _update(self, src, valid, m, scale, box) -> FrameResult:
        cfg = self.cfg
        w, h = self.size
        x0, y0, x1, y1 = box
        roi = (slice(y0, y1), slice(x0, x1))
        q = min(1.0 / scale, 4.0)
        ink = cfg.ink_threshold > 0 and bool((self.quality[roi] > q * cfg.refine_gain).any())
        # work on the box plus a margin (for warping and blurring), write only the box
        margin = max(MARGIN, ink_reach(scale)) if ink else MARGIN
        ex0, ey0, ex1, ey1 = max(x0 - margin, 0), max(y0 - margin, 0), min(x1 + margin, w), min(y1 + margin, h)
        inner = (slice(y0 - ey0, y1 - ey0), slice(x0 - ex0, x1 - ex0))
        mm = (geo.scale_translate(1, -ex0, -ey0) @ m)[:2]
        size = (ex1 - ex0, ey1 - ey0)
        v_ex = cv2.warpAffine(valid, mm, size, flags=cv2.INTER_NEAREST, borderValue=0) > 0
        v = v_ex[inner]
        if not v.any():
            return FrameResult(0, 0)
        f_ex = cv2.warpAffine(src, mm, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # compare at the coarser of the two resolutions, with a little smoothing against compression noise
        sigma = max(0.8, 0.5 * scale)
        fb = cv2.GaussianBlur(f_ex, (0, 0), sigma)[inner]
        cb = cv2.GaussianBlur(self.color[ey0:ey1, ex0:ex1], (0, 0), sigma)[inner]
        f = f_ex[inner]
        c, qual, p, cnt = self.color[roi], self.quality[roi], self.pending[roi], self.count[roi]

        # a faint change is trusted only from a view at least as sharp as the one the pixel came from
        # (a blurrier view differs that much from sharp detail anyway), or where the ink changed
        d = cv2.absdiff(fb, cb).max(axis=2)
        known = qual > 0
        trusted = q * cfg.refine_gain >= qual
        if ink:
            trusted |= ink_changed(f_ex, self.color[ey0:ey1, ex0:ex1], v_ex, scale, cfg.ink_threshold)[inner]
        faint = (d >= cfg.faint_threshold) & trusted
        agree = known & (d < cfg.diff_threshold) & ~faint
        differ = v & ~agree
        agree &= v

        sharper = agree & (q > qual * cfg.refine_gain)
        c[sharper] = f[sharper]
        qual[sharper] = q

        same_pending = cv2.absdiff(f, p).max(axis=2) < cfg.diff_threshold
        cont = differ & (cnt > 0) & same_pending
        fresh = differ & ~cont
        cnt[cont] += 1
        p[fresh] = f[fresh]
        cnt[fresh] = 1
        cnt[agree] = 0

        commit = differ & (cnt >= self.stable)
        c[commit] = f[commit]
        qual[commit] = q
        cnt[commit] = 0
        return FrameResult(int(commit.sum()), int(v.sum()))


class Compositor:
    """Feeds frames into a Canvas, updating only what changed while the view holds still. A canvas
    that is not `incremental` (the GPU one) gets the whole view every frame."""

    def __init__(self, canvas: Canvas, valid: np.ndarray, screen_threshold: int = 10):
        self.canvas = canvas
        self.valid = valid
        self.valid_bool = valid > 0
        self.screen_threshold = screen_threshold
        self.seen: np.ndarray | None = None  # each screen tile as it looked when last processed
        self.prev_m: np.ndarray | None = None
        self.full_updates = 0
        self.partial_updates = 0
        self.skipped = 0

    def add(self, frame: np.ndarray, m: np.ndarray) -> FrameResult:
        same_view = self.prev_m is not None and self._same_view(self.prev_m, m)
        if not same_view:
            self.full_updates += 1
            res = self.canvas.add(frame, self.valid, m)
            self.prev_m = m
            if self.canvas.incremental:
                self.seen = frame.copy()
            return res
        m = self.prev_m  # ignore sub-pixel re-estimation jitter
        if not self.canvas.incremental:
            self.full_updates += 1
            return self.canvas.add(frame, self.valid, m)
        boxes = self._changed_boxes(frame, m) + self._pending_boxes()
        if not boxes:
            self.skipped += 1
            return FrameResult(0, 0)
        self.partial_updates += 1
        return self.canvas.add(frame, self.valid, m, _merge_tiles(boxes, self.canvas.size))

    @staticmethod
    def _same_view(a: np.ndarray, b: np.ndarray) -> bool:
        ds, shift = geo.view_change(a, b)
        return ds < 0.003 and shift < geo.Similarity.of(b).scale  # under one screen pixel over 1000 px

    def _changed_boxes(self, frame: np.ndarray, m: np.ndarray) -> list[tuple[int, int, int, int]]:
        diff = cv2.absdiff(frame, self.seen).max(axis=2) > self.screen_threshold
        diff &= self.valid_bool
        h, w = diff.shape
        th, tw = -(-h // SCREEN_TILE), -(-w // SCREEN_TILE)
        padded = np.zeros((th * SCREEN_TILE, tw * SCREEN_TILE), bool)
        padded[:h, :w] = diff
        tiles = padded.reshape(th, SCREEN_TILE, tw, SCREEN_TILE).any(axis=(1, 3))
        boxes = []
        pad = 2 * geo.Similarity.of(m).scale + 8
        for ty, tx in zip(*np.nonzero(tiles)):
            x, y = tx * SCREEN_TILE, ty * SCREEN_TILE
            self.seen[y:y + SCREEN_TILE, x:x + SCREEN_TILE] = frame[y:y + SCREEN_TILE, x:x + SCREEN_TILE]
            quad = np.array([[x, y], [x + SCREEN_TILE, y], [x, y + SCREEN_TILE], [x + SCREEN_TILE, y + SCREEN_TILE]], float)
            box = self.canvas._clip(geo.apply(m, quad), pad)
            if box:
                boxes.append(box)
        return boxes

    def _pending_boxes(self) -> list[tuple[int, int, int, int]]:
        cnt = self.canvas.count
        h, w = cnt.shape
        th, tw = -(-h // TILE), -(-w // TILE)
        padded = np.zeros((th * TILE, tw * TILE), bool)
        padded[:h, :w] = cnt > 0
        tiles = padded.reshape(th, TILE, tw, TILE).any(axis=(1, 3))
        return [(tx * TILE, ty * TILE, min((tx + 1) * TILE, w), min((ty + 1) * TILE, h)) for ty, tx in zip(*np.nonzero(tiles))]


def _merge_tiles(boxes: list[tuple[int, int, int, int]], size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """Snap boxes to the tile grid and merge them into horizontal runs per tile row."""
    w, h = size
    th, tw = -(-h // TILE), -(-w // TILE)
    grid = np.zeros((th, tw), bool)
    for x0, y0, x1, y1 in boxes:
        grid[y0 // TILE:(y1 - 1) // TILE + 1, x0 // TILE:(x1 - 1) // TILE + 1] = True
    out = []
    for ty in range(th):
        row = grid[ty]
        tx = 0
        while tx < tw:
            if row[tx]:
                start = tx
                while tx < tw and row[tx]:
                    tx += 1
                out.append((start * TILE, ty * TILE, min(tx * TILE, w), min((ty + 1) * TILE, h)))
            else:
                tx += 1
    return out


def ink_reach(scale: float) -> int:
    """Canvas pixels around a box that ink_changed() looks at: the blur, the opening and the growing back."""
    s = INK_SIGMA * scale
    return 3 * math.ceil(s) + 2 * max(1, round(scale)) + math.ceil(INK_GROW * scale)


def ink_changed(f: np.ndarray, c: np.ndarray, v: np.ndarray, scale: float, threshold: int) -> np.ndarray:
    """Where frame `f` and canvas `c` (BGR, same box) differ by `threshold` once both are blurred to
    INK_SIGMA screen pixels, over whole strokes: specks narrower than a screen pixel are dropped, and
    what remains is grown back over the stroke. Only visible pixels (`v`) count."""
    s = INK_SIGMA * scale
    k = int(round(s * 6 + 1)) | 1
    wv = v.astype(np.float32)
    norm = cv2.GaussianBlur(wv, (k, k), s)
    fb = cv2.GaussianBlur(f.astype(np.float32) * wv[..., None], (k, k), s)
    cb = cv2.GaussianBlur(c.astype(np.float32) * wv[..., None], (k, k), s)
    d = np.abs(fb - cb).max(axis=2) / np.maximum(norm, 1e-6)
    changed = ((d >= threshold) & v & (norm > 0.25)).astype(np.uint8)
    r = max(1, round(scale))
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    g = math.ceil(INK_GROW * scale)
    return cv2.dilate(changed, np.ones((2 * g + 1, 2 * g + 1), np.uint8)) > 0 if g else changed > 0


def valid_mask(screen_mask: np.ndarray, erode_px: int) -> np.ndarray:
    if erode_px <= 0:
        return screen_mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erode_px + 1, 2 * erode_px + 1))
    return cv2.erode(screen_mask, k)
