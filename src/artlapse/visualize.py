"""Visual checks: setup previews and registration overlays."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import geometry as geo
from . import pipeline
from .config import Config
from .register import Status, Track, reference_transform
from .video import Encoder, Sampler, grab, prefetch, probe

STATUS_COLOR = {Status.MATCHED: (0, 200, 0), Status.STATIC: (200, 160, 0), Status.LOST: (0, 0, 255), Status.UNSEEN: (128, 128, 128)}


def _rect(img, r, color, scale=1.0, thickness=2, label=None):
    x, y, w, h = (round(v * scale) for v in r)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    if label:
        cv2.putText(img, label, (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def preview_setup(cfg: Config, time: float | None = None) -> np.ndarray:
    """A frame with the workspace (green), exclusions (red) and target (magenta) drawn on it."""
    t = cfg.reference.time if time is None else time
    img = grab(cfg.video, t).copy()
    _rect(img, cfg.workspace, (0, 220, 0), label="workspace")
    for r in cfg.exclude:
        _rect(img, r, (0, 0, 255), label="exclude")
    if t == cfg.reference.time:
        _rect(img, cfg.reference.target, (255, 0, 255), label="target")
    return img


def canvas_outline(m_full: np.ndarray, size: tuple[int, int], scale: float) -> np.ndarray:
    """The target canvas rectangle projected into a frame, as an int polygon at `scale` x full res."""
    w, h = size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    pts = geo.apply(np.linalg.inv(m_full), corners) * scale
    return pts.round().astype(np.int32)


def overlay_video(cfg: Config, track: Track, out: Path, start: float, end: float, width: int = 960) -> None:
    """Render the sampled frames between start and end (seconds) with the tracked canvas outline."""
    sampler = Sampler(cfg.video, cfg.sample_fps, width=width, cache=True)
    r = width / sampler.info.width
    _, size = reference_transform(cfg)
    lo, hi = round(start * cfg.sample_fps), round(end * cfg.sample_fps)
    with Encoder(out, sampler.width, sampler.height, fps=10, crf=23) as enc:
        for n, frame in sampler.frames(lo, hi):
            img = frame.copy()
            st = Status(track.status[n])
            if track.ok(n):
                poly = canvas_outline(track.m[n], size, r)
                cv2.polylines(img, [poly], True, (255, 0, 255), 2)
                cv2.circle(img, tuple(poly[0]), 6, (255, 0, 255), -1)  # canvas origin, shows rotation/flip
            s = geo.Similarity.of(track.m[n])
            text = f"{n / cfg.sample_fps:8.1f}s {st.name} inl={track.inliers[n]}"
            if track.ok(n):
                text += f" zoom={1 / s.scale:.2f} rot={s.angle_deg:.0f}{' FLIP' if s.flipped else ''}"
            cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, STATUS_COLOR[st], 2)
            enc.write(img)


def side_by_side(works: list[Path], out: Path, height: int = 1080, crf: int = 20) -> None:
    """The stream next to its timelapse, frame by frame: each timelapse frame beside the stream at the
    moment its canvas shows. Several work directories (consecutive parts of one stream) play one after
    another."""
    parts = []
    for work in works:
        cfg = Config.load(work / pipeline.CONFIG)
        snaps = np.load(work / pipeline.SNAPSHOTS)
        index = snaps["index"][pipeline.render_picks(cfg, snaps)]
        cap = cv2.VideoCapture(str(work / pipeline.TIMELAPSE))
        frames, w, h = (int(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if frames != len(index):
            raise ValueError(f"{work / pipeline.TIMELAPSE} has {frames} frames, the render settings give {len(index)}: render again")
        parts.append((work, cfg, index, round(w * height / h / 2) * 2))
    info = probe(parts[0][1].video)
    stream_w = round(info.width * height / info.height / 2) * 2
    lapse_w = max(p[3] for p in parts)
    gap = round(height / 120) * 2  # dark divider, so a white canvas does not run into a white stream
    fps = parts[0][1].render.fps
    with Encoder(out, stream_w + gap + lapse_w, height, fps, crf) as enc:
        for work, cfg, index, w in parts:
            sampler = Sampler(cfg.video, cfg.sample_fps, width=stream_w, cache=True)
            stream = prefetch(sampler.select(index.tolist()), depth=16)
            cap = cv2.VideoCapture(str(work / pipeline.TIMELAPSE))
            n, shown = -1, None
            for want in index:
                while n < want:
                    n, shown = next(stream)
                ok, lapse = cap.read()
                if not ok:
                    raise RuntimeError(f"{work / pipeline.TIMELAPSE} ended early")
                img = np.zeros((height, stream_w + gap + lapse_w, 3), np.uint8)
                img[:shown.shape[0], :stream_w] = shown
                x = stream_w + gap + (lapse_w - w) // 2
                img[:, x:x + w] = cv2.resize(lapse, (w, height), interpolation=cv2.INTER_AREA)
                t = round(n / cfg.sample_fps)
                label = f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}"
                scale = height / 1080
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, 2)
                pad = round(12 * scale)
                cv2.rectangle(img, (0, 0), (tw + 2 * pad, th + 2 * pad), (0, 0, 0), -1)
                cv2.putText(img, label, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, (255, 255, 255), 2, cv2.LINE_AA)
                enc.write(img)
            cap.release()
            stream.close()
