"""Starting a canvas from the final canvas of another work directory.

A drawing continued over several streams (or a stream run in parts) would otherwise start each part
from an empty canvas and build it up again from that part's first views, although the stream shows
the picture as the earlier part left it. Instead the later part starts from the earlier part's final
canvas, placed where the picture sits in this part's canvas: each part has its own reference frame
and target, so the two canvases frame the picture differently. The placement comes from matching
the earlier canvas against frames from the start of this part, whose place in this canvas the track
gives.
"""
from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from . import geometry as geo

FIRST_SECONDS = 600.0  # the frames matched come from this long after the part's start
CANDIDATES = 24  # ... this many of them, spread over that time
MIN_INLIERS = 50
RATIO = 0.75
RANSAC_PX = 3.0  # screen pixels


def candidates(placed: list[int], sample_fps: float) -> list[int]:
    """The samples to match the earlier canvas against: spread over the first FIRST_SECONDS of `placed`."""
    first = [n for n in placed if n < placed[0] + FIRST_SECONDS * sample_fps] if placed else []
    return sorted({first[i] for i in np.linspace(0, len(first) - 1, CANDIDATES).astype(int)}) if first else []


def placement(prev: np.ndarray, frames: Iterable[tuple[int, np.ndarray]], track_m: np.ndarray,
              mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """The transform from the earlier canvas `prev` (BGR) to this canvas, from the frame (n, BGR frame)
    that matches it best: the earlier canvas -> screen similarity found by RANSAC, followed by the
    frame's screen -> canvas transform `track_m[n]`. `mask` is the screen area the canvas may show."""
    sift = cv2.SIFT_create(nfeatures=8000)
    kp, desc = sift.detectAndCompute(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), None)
    if desc is None:
        raise ValueError("the earlier part's final canvas has no features")
    pts = np.float32([k.pt for k in kp])
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
    best: tuple[int, int, np.ndarray] | None = None
    for n, frame in frames:
        kf, df = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask)
        if df is None or len(kf) < 2:
            continue
        good = [a for a, b in (p for p in matcher.knnMatch(desc, df, k=2) if len(p) == 2) if a.distance < RATIO * b.distance]
        if len(good) < MIN_INLIERS:
            continue
        src = pts[[g.queryIdx for g in good]]
        dst = np.float32([kf[g.trainIdx].pt for g in good])
        a, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX,
                                             maxIters=5000, confidence=0.999)
        if a is not None and (best is None or int(inl.sum()) > best[1]):
            best = (n, int(inl.sum()), track_m[n] @ geo.to3(a))
    if best is None or best[1] < MIN_INLIERS:
        raise ValueError("the earlier part's final canvas matches none of this part's first frames")
    n, inliers, m = best
    s = geo.Similarity.of(m)
    return m, {"sample": n, "inliers": inliers, "scale": round(s.scale, 4), "angle": round(s.angle_deg, 2),
               "shift": [round(s.tx, 1), round(s.ty, 1)]}


def seeded(prev: np.ndarray, m: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The earlier canvas warped into this one by `m`, and the mask of where it lands."""
    scale = geo.Similarity.of(m).scale
    src = cv2.GaussianBlur(prev, (0, 0), 0.4 / scale) if scale < 0.6 else prev  # low-pass before shrinking
    image = cv2.warpAffine(src, m[:2], size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    known = cv2.warpAffine(np.full(prev.shape[:2], 255, np.uint8), m[:2], size, flags=cv2.INTER_NEAREST, borderValue=0) > 0
    return image, known
