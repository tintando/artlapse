"""Project configuration, stored as YAML in the work directory.

All screen coordinates are full-resolution video pixels. Rectangles are [x, y, w, h]. Times are
seconds into the video, given as a number or as "h:mm:ss" / "m:ss" (decimals allowed).
The canvas coordinate system is the reference frame's screen, shifted so the target
region starts at the origin and scaled so its long side is `canvas.long_side` pixels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

Rect = list[float]  # [x, y, w, h]


@dataclass
class Reference:
    time: float = 0.0  # the frame that defines canvas coordinates
    target: Rect = field(default_factory=lambda: [0, 0, 0, 0])  # area to rebuild, in that frame


@dataclass
class CanvasCfg:
    long_side: int = 2048  # rebuilt canvas resolution (long side of the target region)
    background: list[int] = field(default_factory=lambda: [255, 255, 255])  # BGR for never-seen pixels
    continues: str = ""  # a work directory (relative to this one) whose final canvas this one starts from


@dataclass
class RegisterCfg:
    analysis_width: int = 960  # frames are downscaled to this width for feature matching
    max_features: int = 1500
    ratio: float = 0.75  # Lowe ratio test
    ransac_px: float = 2.0  # reprojection threshold, analysis pixels
    min_inliers: int = 25
    min_inlier_ratio: float = 0.15
    keep_view_ratio: float = 0.3  # the previous view stays in the running while it explains this share of the best inlier count
    keep_view_spread: float = 0.2  # ... and its supporting points cover this share of the matched area (a still overlay does not)
    static_diff: float = 1.5  # mean abs grey difference under which a frame counts as unmoved
    map_cell_px: float = 3.0  # feature map grid cell, in analysis pixels at the observing zoom (merges near-duplicates only)
    gpu_features: bool = False  # with device cuda: SiftGPU instead of OpenCV's SIFT; faster, but loses the view more often


@dataclass
class AccumulateCfg:
    diff_threshold: int = 28  # per-channel difference that counts as "changed"
    faint_threshold: int = 10  # smaller changes (a faint layer hidden) count only from views at least as sharp
    ink_threshold: int = 8  # ... or where the strokes' ink changed this much at the view's scale (0: never)
    stable_seconds: float = 8.0  # a change is committed once it has held this long (overlays rarely sit still that long)
    erode_px: int = 3  # shrink the valid region (screen px) to avoid view-border artefacts
    refine_gain: float = 1.15  # replace an agreeing pixel when the new view is this much sharper
    min_changed_px: int = 40  # snapshots are written when at least this many canvas pixels changed


@dataclass
class RenderCfg:
    duration: float = 60.0  # seconds of timelapse, excluding the final hold
    fps: int = 30
    hold_end: float = 3.0
    activity_weight: float = 0.8  # 1 = pace by painting activity, 0 = by stream time
    crf: int = 18


@dataclass
class Config:
    video: str = ""
    device: str = "auto"  # cpu, cuda, or auto: cuda when the gpu extra is installed and sees a device
    sample_fps: float = 2.0
    start: float = 0.0  # only this part of the video is processed
    end: float | None = None
    cuts: list[list[float]] = field(default_factory=list)  # [from, to] stretches inside it that are left out
    workspace: Rect = field(default_factory=lambda: [0, 0, 0, 0])  # where the art program shows the canvas
    exclude: list[Rect] = field(default_factory=list)  # overlays on top of the workspace
    reference: Reference = field(default_factory=Reference)
    canvas: CanvasCfg = field(default_factory=CanvasCfg)
    register: RegisterCfg = field(default_factory=RegisterCfg)
    accumulate: AccumulateCfg = field(default_factory=AccumulateCfg)
    render: RenderCfg = field(default_factory=RenderCfg)

    @staticmethod
    def load(path: Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text()) or {}
        cfg = _build(Config, data)
        cfg.start, cfg.end, cfg.reference.time = seconds(cfg.start), seconds(cfg.end), seconds(cfg.reference.time)
        cfg.cuts = [[seconds(a), seconds(b)] for a, b in cfg.cuts]
        return cfg

    def save(self, path: Path) -> None:
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))


def seconds(value: float | str | None) -> float | None:
    """A time as seconds: a number, or "h:mm:ss" / "m:ss" with optional decimals."""
    if value is None or isinstance(value, (int, float)):
        return None if value is None else float(value)
    total = 0.0
    for part in str(value).strip().split(":"):
        total = total * 60 + float(part)
    return total


def clock(t: float) -> str:
    """Seconds as h:mm:ss.s, the way the stream's player shows them."""
    t = round(t, 1)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:04.1f}"


def _build(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else known[name].default
        kwargs[name] = _build(type(default), value or {}) if is_dataclass(default) else value
    return cls(**kwargs)
