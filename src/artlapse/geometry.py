"""2D similarity transforms with optional mirroring.

Art programs move the view by zoom, rotation, translation and horizontal flip,
so the map from screen to canvas is always a similarity, possibly reflected.
Transforms are stored as 3x3 homogeneous matrices.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def to3(a: np.ndarray) -> np.ndarray:
    """2x3 affine -> 3x3 homogeneous."""
    m = np.eye(3)
    m[:2] = a
    return m


def scale_translate(s: float, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    return np.array([[s, 0, tx], [0, s, ty], [0, 0, 1]], dtype=float)


def mirror_x(width: float) -> np.ndarray:
    """Horizontal mirror of an image `width` pixels wide (pixel centres stay on the grid)."""
    return np.array([[-1, 0, width - 1], [0, 1, 0], [0, 0, 1]], dtype=float)


def apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 transform to an (N, 2) array of points."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    return pts @ m[:2, :2].T + m[:2, 2]


@dataclass(frozen=True)
class Similarity:
    """Decomposition of a similarity: p' = R(angle) * diag(scale) * F * p + t."""

    scale: float
    angle_deg: float
    flipped: bool
    tx: float
    ty: float

    @staticmethod
    def of(m: np.ndarray) -> "Similarity":
        a = m[:2, :2]
        det = float(np.linalg.det(a))
        flipped = det < 0
        scale = math.sqrt(abs(det))
        # with a mirror, the first column is R*s*(-1, 0): undo the sign to read the angle
        cx, cy = (-a[0, 0], -a[1, 0]) if flipped else (a[0, 0], a[1, 0])
        return Similarity(scale, math.degrees(math.atan2(cy, cx)), flipped, float(m[0, 2]), float(m[1, 2]))


def is_similarity(m: np.ndarray, tol: float = 1e-3) -> bool:
    a = m[:2, :2]
    s2 = abs(np.linalg.det(a))
    return bool(np.allclose(a @ a.T, s2 * np.eye(2), atol=tol * max(s2, 1e-9)))


def view_change(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    """How much a screen->canvas transform moved: (relative scale change, max corner shift in canvas units)
    measured over a nominal 1000x1000 screen patch."""
    corners = np.array([[0, 0], [1000, 0], [0, 1000], [1000, 1000]], float)
    shift = float(np.max(np.linalg.norm(apply(prev, corners) - apply(cur, corners), axis=1)))
    ds = abs(math.log(Similarity.of(cur).scale / Similarity.of(prev).scale))
    return ds, shift
