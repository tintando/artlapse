"""Synthetic streams with known ground truth.

A procedurally drawn artwork is revealed stroke by stroke and shown through a moving
view (zoom, rotation, pan, flip) inside a fake art-program window. The window has a
static toolbar, a bobbing "avatar" overlay, a wandering cursor and a scene cut. Optionally,
textured panels (chat, a tool strip) sit fixed on screen inside the workspace, not excluded, and
the view jumps straight between close-ups and the whole artwork. Optionally, a large textured figure
the user did not exclude (a second avatar) walks across the canvas in jumps while the view holds still.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from artlapse import geometry as geo

SCREEN = (1280, 720)
WORKSPACE = [0, 0, 1120, 720]
TOOLBAR = [1120, 0, 160, 720]
AVATAR = [900, 420, 380, 300]  # overlaps the workspace; excluded in the config
ART = (1600, 1200)
GRAY = (70, 70, 70)


def artwork(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w, h = ART
    img = np.full((h, w, 3), 255, np.uint8)
    for _ in range(140):
        color = tuple(int(c) for c in rng.integers(0, 230, 3))
        kind = rng.integers(0, 4)
        if kind == 0:
            pts = rng.integers(0, [w, h], (int(rng.integers(3, 7)), 2)).astype(np.int32)
            cv2.polylines(img, [pts], False, color, int(rng.integers(2, 7)), cv2.LINE_AA)
        elif kind == 1:
            c = tuple(int(v) for v in rng.integers(0, [w, h]))
            axes = tuple(int(v) for v in rng.integers(10, 90, 2))
            cv2.ellipse(img, c, axes, float(rng.uniform(0, 180)), 0, 360, color, -1, cv2.LINE_AA)
        elif kind == 2:
            p = tuple(int(v) for v in rng.integers(0, [w - 60, h - 60]))
            cv2.rectangle(img, p, (p[0] + int(rng.integers(15, 80)), p[1] + int(rng.integers(15, 80))), color, -1)
        else:
            p = tuple(int(v) for v in rng.integers(0, [w - 200, h - 20]))
            cv2.putText(img, "".join(rng.choice(list("ABCDEFGHJKMNPRSTUVWXYZ"), 5)), p,
                        cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.8, 2.0)), color, int(rng.integers(2, 4)))
    return img


def reveal_masks(n: int, seed: int = 0) -> list[np.ndarray]:
    """n monotonically growing masks; the last one is full. Early stages already show some structure."""
    rng = np.random.default_rng(seed + 1)
    w, h = ART
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (300, 200), (1300, 1000), 255, -1)  # "sketch" stage: centre already painted
    out = []
    for i in range(n):
        for _ in range(6):
            c = tuple(int(v) for v in rng.integers(0, [w, h]))
            cv2.circle(mask, c, int(rng.integers(40, 160)), 255, -1)
        out.append(np.full((h, w), 255, np.uint8) if i == n - 1 else mask.copy())
    return out


@dataclass
class Synth:
    path: Path
    fps: float
    count: int
    views: list[np.ndarray]  # canvas(ART pixels) -> screen per sample; None during the scene cut
    final: np.ndarray
    reference: int
    cut: range = field(default_factory=lambda: range(0))


def view(zoom: float, angle: float, flip: bool, cx: float, cy: float) -> np.ndarray:
    """canvas -> screen: canvas point (cx, cy) lands at the workspace centre."""
    m = geo.to3(cv2.getRotationMatrix2D((0, 0), angle, zoom))
    if flip:
        m = m @ np.diag([-1.0, 1.0, 1.0])
    wx, wy = WORKSPACE[0] + WORKSPACE[2] / 2, WORKSPACE[1] + WORKSPACE[3] / 2
    return geo.scale_translate(1, wx, wy) @ m @ geo.scale_translate(1, -cx, -cy)


def transformed(art: np.ndarray) -> np.ndarray:
    """The artwork after the artist scaled a large region up by 1.4 about its centre (a layer transform)."""
    out = art.copy()
    x0, y0, x1, y1 = 450, 300, 1150, 900
    patch = art[y0:y1, x0:x1]
    big = cv2.resize(patch, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_LINEAR)
    bh, bw = big.shape[:2]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    out[cy - bh // 2:cy - bh // 2 + bh, cx - bw // 2:cx - bw // 2 + bw] = big
    return out


def panels(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Textured stream panels fixed on screen inside the workspace (a chat box, a tool strip), as an
    image and a mask. They are not excluded: the user missed them."""
    rng = np.random.default_rng(seed + 3)
    img = np.zeros((SCREEN[1], SCREEN[0], 3), np.uint8)
    mask = np.zeros(SCREEN[::-1], np.uint8)
    for x, y, w, h in [(10, 450, 330, 260), (0, 0, 1120, 46)]:
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 30, 50), -1)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        for ty in range(y + 30, y + h, 26):
            words = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789 "), 28))
            cv2.putText(img, words, (x + 8, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return img, mask


def visitor(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A large textured figure, as an image and a mask: an ellipse cut from other artwork."""
    img = cv2.resize(artwork(seed + 7)[200:950, 300:900], (440, 560))
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(mask, (220, 280), (215, 275), 0, 0, 360, 255, -1)
    return img, mask


VISITOR_SAMPLES = range(80, 100)  # the view holds still and the visitor moves every other sample
VISITOR_X = [20, 660, 40, 640, 20, 660, 40, 640, 20, 660]


def make(path: Path, seed: int = 0, count: int = 120, transform_at: int | None = None, with_panels: bool = False,
         big_zooms: bool = False, with_visitor: bool = False) -> Synth:
    rng = np.random.default_rng(seed + 2)
    panel_img, panel_mask = panels(seed)
    art = artwork(seed)
    art_after = transformed(art) if transform_at is not None else art
    hold = 24  # the finished artwork stays on screen for the last 12 s, as at the end of a stream
    masks = reveal_masks(count - hold, seed)
    masks += [masks[-1]] * hold
    # piecewise-constant views with jumps every few samples, like keyboard zoom/rotate/flip
    script = [(0.55, 0, False, 800, 600)]  # reference: the whole artwork visible
    while len(script) < count // 4:
        z, a, f, cx, cy = script[-1]
        r = rng.random()
        if r < 0.3:
            z = float(np.clip(z * rng.choice([0.67, 1.5, 2.0, 0.5]), 0.4, 2.5))
        elif r < 0.5:
            a = float(rng.choice([0, 15, -30, 45, 90]))
        elif r < 0.6:
            f = not f
        else:
            cx, cy = float(rng.uniform(300, 1300)), float(rng.uniform(250, 950))
        script.append((z, a, f, cx, cy))
    if big_zooms:  # close-ups and the whole artwork in single jumps, both ways
        script[5:10] = [(2.4, 0, False, 600, 450), (0.5, 0, False, 800, 600), (2.4, 0, True, 1050, 750),
                        (0.5, 0, True, 800, 600), (2.2, 45, True, 500, 800)]
    if with_visitor:
        script[VISITOR_SAMPLES.start // 4:VISITOR_SAMPLES.stop // 4] = [(2.4, 0, False, 800, 600)] * (len(VISITOR_SAMPLES) // 4)
    visitor_img, visitor_mask = visitor(seed)
    cut =range(count // 2, count // 2 + 6)
    views, frames = [], []
    avatar_img = np.zeros((AVATAR[3], AVATAR[2], 3), np.uint8)
    cv2.circle(avatar_img, (190, 170), 120, (60, 180, 240), -1)
    cv2.circle(avatar_img, (150, 140), 15, (20, 20, 20), -1)
    cv2.circle(avatar_img, (230, 140), 15, (20, 20, 20), -1)
    for n in range(count):
        if n in cut:
            scene = np.full((SCREEN[1], SCREEN[0], 3), (180, 120, 40), np.uint8)
            cv2.putText(scene, "BE RIGHT BACK", (300, 380), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
            frames.append(scene)
            views.append(None)
            continue
        z, a, f, cx, cy = script[min(n // 4, len(script) - 1)]
        m = view(z, a, f, cx, cy)
        shown = art_after if transform_at is not None and n >= transform_at else art
        state = np.where(masks[n][..., None] > 0, shown, 255).astype(np.uint8)
        screen = np.full((SCREEN[1], SCREEN[0], 3), GRAY, np.uint8)
        ws = cv2.warpAffine(state, m[:2], SCREEN, flags=cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=GRAY)
        x, y, w, h = WORKSPACE
        screen[y:y + h, x:x + w] = ws[y:y + h, x:x + w]
        if with_panels:
            screen[panel_mask > 0] = panel_img[panel_mask > 0]
        if with_visitor and n in VISITOR_SAMPLES:
            vx = VISITOR_X[(n - VISITOR_SAMPLES.start) // 2]
            vh, vw = visitor_mask.shape
            sub = screen[80:80 + vh, vx:vx + vw]
            sub[visitor_mask > 0] = visitor_img[visitor_mask > 0]
        x, y, w, h = TOOLBAR
        screen[y:y + h, x:x + w] = (45, 45, 45)
        for i in range(12):
            cv2.rectangle(screen, (x + 20 + 60 * (i % 2), 20 + 55 * (i // 2)), (x + 70 + 60 * (i % 2), 60 + 55 * (i // 2)), (90 + 12 * i, 120, 200 - 10 * i), -1)
        ax, ay, aw, ah = AVATAR
        bob = int(10 * np.sin(n))
        sub = screen[ay + 20 + bob:ay + ah - 20 + bob, ax:ax + aw]
        a_img = avatar_img[20:ah - 20]
        sel = a_img.any(axis=2)
        sub[sel] = a_img[sel]
        cur = tuple(int(v) for v in rng.integers(50, [1000, 650]))
        cv2.circle(screen, cur, 14, (0, 0, 0), 1, cv2.LINE_AA)
        frames.append(screen)
        views.append(m)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{SCREEN[0]}x{SCREEN[1]}",
                             "-r", "2", "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-g", "4", str(path)],
                            stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return Synth(path, 2.0, count, views, art_after, reference=0, cut=cut)
