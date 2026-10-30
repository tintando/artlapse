"""Which of several tracks of one work directory places the view right, after they part.

An early frame (from before the tracks part, where they agree) and later frames, every --every
seconds, are warped into the canvas by each track in turn; phase correlation on their strokes gives
the shift left between them. Under a right track, strokes that did not change in between coincide,
so the smaller residual wins. Rows without a correlation peak (r near 0: too little in common) say
nothing. With --lag, each later frame is compared with the frame that long before it instead, so
tracks that agree nowhere can be compared too.

    uv run python scripts/residual.py work/vod_exp --early 11:35 work/vod_exp/track.npz other/track.npz
    uv run python scripts/residual.py work/vod_exp --lag 60 work/vod_exp/track.npz other/track.npz
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from artlapse import geometry as geo
from artlapse.config import Config, clock, seconds
from artlapse.register import Track, reference_transform, screen_mask
from artlapse.video import Sampler, probe

PEAK = 0.3  # correlation response above which a residual counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("work")
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--early", type=seconds, help="the early frame's time (seconds or h:mm:ss)")
    which.add_argument("--lag", type=float, help="seconds between each later frame and the frame it is compared with")
    p.add_argument("tracks", nargs="+")
    p.add_argument("--every", type=float, default=30.0, help="seconds between the later frames")
    p.add_argument("--from", dest="start", type=seconds, help="the first later frame (default: 50 s after the early one, "
                                                             "or the lag after the start)")
    a = p.parse_args()

    cfg = Config.load(f"{a.work}/config.yaml")
    tracks = [Track.load(t) for t in a.tracks]
    _, size = reference_transform(cfg)
    info = probe(cfg.video)
    mask = screen_mask(cfg, info.width, info.height, 1.0)
    placed = lambda n: all(t.ok(n) for t in tracks)
    if a.lag is None:
        n0 = round(a.early * cfg.sample_fps)
        if not placed(n0):
            raise SystemExit(f"not every track places the early frame ({clock(a.early)})")
        start = a.start if a.start is not None else a.early + 50
    else:
        start = a.start if a.start is not None else cfg.start + a.lag
    lag = None if a.lag is None else round(a.lag * cfg.sample_fps)

    def before(n: int) -> int | None:
        """The frame `n` is compared with: the early one, or the nearest placed one about a lag earlier."""
        if lag is None:
            return n0
        return next((k for d in range(0, lag // 2) for k in (n - lag - d, n - lag + d) if k >= 0 and placed(k)), None)

    later = [(before(n), n) for n in range(round(start * cfg.sample_fps), len(tracks[0].status),
                                           max(1, round(a.every * cfg.sample_fps))) if placed(n)]
    later = [(k, n) for k, n in later if k is not None]
    frames = dict(Sampler(cfg.video, cfg.sample_fps, cache=True).select(sorted({x for pair in later for x in pair})))

    def strokes(t: Track, n: int) -> tuple[np.ndarray, np.ndarray]:
        g = cv2.warpAffine(cv2.cvtColor(frames[n], cv2.COLOR_BGR2GRAY), t.m[n][:2], size).astype(np.float32)
        return g, cv2.warpAffine(mask, t.m[n][:2], size, flags=cv2.INTER_NEAREST) > 0

    print("time     sample  zoom  " + "  ".join(f"track {i + 1:<12d}" for i in range(len(tracks))))
    medians = [[] for _ in tracks]
    for k, n in later:
        out = []
        for i, t in enumerate(tracks):
            (g0, v0), (g1, v1) = strokes(t, k), strokes(t, n)
            both = cv2.erode((v0 & v1).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            if both.mean() < 0.05:
                out.append(f"{'too little overlap':18s}")
                continue
            ys, xs = np.nonzero(both)
            box = (slice(ys.min(), ys.max()), slice(xs.min(), xs.max()))
            grads = []
            for g in (g0, g1):
                g = np.where(both, g, 0)[box]
                grads.append(np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2))
            (dx, dy), r = cv2.phaseCorrelate(*grads, cv2.createHanningWindow(grads[0].shape[::-1], cv2.CV_32F))
            out.append(f"{np.hypot(dx, dy):6.1f} px (r {r:.2f})  ")
            if r >= PEAK:
                medians[i].append(np.hypot(dx, dy))
        s = geo.Similarity.of(tracks[0].m[n])
        print(f"{clock(n / cfg.sample_fps)} {n:6d}  {s.scale:4.2f}{'F' if s.flipped else ' '} " + "  ".join(out), flush=True)
    print(f"median over rows with a peak (r >= {PEAK}): " + "  ".join(
        f"track {i + 1} {np.median(m):.1f} px ({len(m)} rows)" if m else f"track {i + 1} -" for i, m in enumerate(medians)))


if __name__ == "__main__":
    main()
