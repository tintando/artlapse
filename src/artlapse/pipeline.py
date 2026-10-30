"""Pipeline stages and the files they leave in a work directory.

    config.yaml      setup: video, screen regions, reference frame, tunables
    track.npz        register: screen->canvas transform and status per sample
    decisions.jsonl.gz  register: how each sample not found unmoved was decided (Registrar.step)
    canvas.mp4       composite: the canvas each time it changed (near-lossless)
    snapshots.npz    composite: sample index and changed-pixel count per canvas.mp4 frame
    final.png        composite: the rebuilt canvas at the end of the stream
    timelapse.mp4    render
    runs.jsonl       every stage: when it ran, on which engine, config and device, what it found
    history/         every stage: its small outputs and the config, per engine, config and device,
                     and the source's uncommitted changes (source.diff) when it ran from a dirty checkout
    check.json       check: the track's jumps and its agreement with the final canvas
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import shutil
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

from . import check, continuation
from .accumulate import Canvas, Compositor, valid_mask
from .config import Config, clock
from .register import Status, Track, kept_samples, reference_transform, register, sample_range, screen_mask
from .video import Encoder, Sampler, prefetch, probe

log = logging.getLogger(__name__)

CONFIG = "config.yaml"
TRACK = "track.npz"
DECISIONS = "decisions.jsonl.gz"
CANVAS = "canvas.mp4"
SNAPSHOTS = "snapshots.npz"
FINAL = "final.png"
TIMELAPSE = "timelapse.mp4"
RUNS = "runs.jsonl"
HISTORY = "history"
CHECK = "check.json"
SOURCE_DIFF = "source.diff"


SRC = Path(__file__).parent


def source_hash(names: list[str] | None = None) -> str:
    """A hash of the engine's source files (all of them, or these)."""
    h = hashlib.sha256()
    for path in sorted(SRC.glob("*.py")):
        if names is None or path.name in names:
            h.update(path.name.encode() + path.read_bytes())
    return h.hexdigest()[:10]


def git(*args: str) -> str:
    """A git command's output, run in the source directory; raises with git's message when it fails."""
    done = subprocess.run(["git", *args], cwd=SRC, capture_output=True, text=True, timeout=30)
    if done.returncode:
        raise RuntimeError(done.stderr.strip() or f"git {args[0]} exited with {done.returncode}")
    return done.stdout


def engine_id() -> dict:
    """Which engine produces an output: the package version, a hash of its source (tells uncommitted
    states apart) and the git commit when it runs from a checkout, or why there is none."""
    entry = {"version": version("artlapse"), "source": source_hash(), "commit": None}
    try:
        entry["commit"] = git("describe", "--always", "--dirty").strip() or None
    except (OSError, subprocess.SubprocessError, RuntimeError) as e:
        entry["commit_error"] = str(e)[:300]
    return entry


def source_diff() -> str:
    """The engine source's changes against the commit: what a dirty commit id leaves out."""
    diff = git("diff", "HEAD", "--", ".")
    for name in git("ls-files", "--others", "--exclude-standard", "--", ".").split():
        done = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", name], cwd=SRC, capture_output=True,
                              text=True, timeout=30)
        diff += done.stdout
    return diff


def resolve_device(cfg: Config) -> str:
    """The device a run uses: "cpu" or "cuda"."""
    if cfg.device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device must be auto, cpu or cuda, not {cfg.device!r}")
    if cfg.device == "cpu":
        return "cpu"
    from . import accumulate_gpu
    if accumulate_gpu.available():
        return "cuda"
    if cfg.device == "cuda":
        raise RuntimeError("device: cuda needs CuPy and a CUDA device (uv sync --extra gpu)")
    return "cpu"


def register_device(cfg: Config) -> str:
    """Register runs on the GPU only with register.gpu_features: SiftGPU is faster, but its tracks are
    less reliable than OpenCV's (on a test stream it lost the view for minutes that OpenCV follows), so by
    default register is the same on every device."""
    if not cfg.register.gpu_features or resolve_device(cfg) != "cuda":
        return "cpu"
    from . import features
    if not features.gpu_available():
        raise RuntimeError("register.gpu_features needs pycolmap with CUDA (uv sync --extra gpu)")
    return "cuda"


# the settings each stage reads, its own and those of the stages before it
STAGE_SETTINGS = {
    "register": ("video", "sample_fps", "start", "end", "cuts", "workspace", "exclude", "reference", "register"),
    "composite": ("canvas", "accumulate"),
    "render": ("render",),
}
STAGES = list(STAGE_SETTINGS)
# the source files each stage's code is in, counted like the settings; pipeline.py from composite on,
# since its register step only calls register()
STAGE_SOURCES = {
    "register": ("register.py", "features.py", "geometry.py", "video.py", "config.py"),
    "composite": ("accumulate.py", "accumulate_gpu.py", "continuation.py", "pipeline.py"),
    "render": (),
}


def continued(cfg: Config, work: Path) -> Path | None:
    """The final canvas of the part this work directory continues (`canvas.continues`), if any."""
    if not cfg.canvas.continues:
        return None
    path = (work / cfg.canvas.continues / FINAL).resolve()
    if not path.exists():
        raise FileNotFoundError(f"canvas.continues: {path} does not exist (composite that part first)")
    return path


def stage_inputs(cfg: Config, stage: str, work: Path | None = None) -> str:
    """A hash of everything a stage's outputs depend on: the video file, and the source, settings and
    devices of the stage and the stages before it; from composite on, the final canvas of the part this
    one continues."""
    data = asdict(cfg)
    upto = STAGES[:STAGES.index(stage) + 1]
    names = [k for s in upto for k in STAGE_SETTINGS[s]]
    video = Path(cfg.video)
    st = video.stat() if video.exists() else None
    key = {"engine": source_hash([f for s in upto for f in STAGE_SOURCES[s]]), "settings": {k: data[k] for k in names},
           "video": [str(video.resolve()), st.st_size, st.st_mtime_ns] if st else None,
           "register_device": register_device(cfg)}
    if stage != "register":
        key["device"] = resolve_device(cfg)
        if work is not None and (prev := continued(cfg, work)):
            key["continues"] = hashlib.sha256(prev.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


def runs(work: Path) -> list[dict]:
    path = work / RUNS
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


@contextmanager
def logged(work: Path, stage: str, outputs: list[Path], device: str = "cpu", writes: list[Path] | None = None) -> Iterator[dict]:
    """Record a stage in runs.jsonl and keep its outputs under history/<engine source>-<config hash>/
    (with -<device> appended for a stage that ran on another device than the CPU), so a later run does
    not erase what an earlier engine, setting or device produced. `writes` lists everything the stage
    writes, when that is more than it keeps. The stage puts what it found in the yielded dict's "result"."""
    config = (work / CONFIG).read_bytes()
    entry = {"stage": stage, "started": datetime.now().isoformat(timespec="seconds"), "engine": engine_id(),
             "device": device, "config": hashlib.sha256(config).hexdigest()[:10],
             "inputs": stage_inputs(Config.load(work / CONFIG), stage, work), "result": {}}
    t0 = time.monotonic()
    yield entry
    entry["seconds"] = round(time.monotonic() - t0)
    suffix = "" if device == "cpu" else f"-{device}"
    keep = work / HISTORY / f"{entry['engine']['source']}-{entry['config']}{suffix}"
    keep.mkdir(parents=True, exist_ok=True)
    (keep / CONFIG).write_bytes(config)
    if (entry["engine"]["commit"] or "").endswith("-dirty"):
        try:
            if diff := source_diff():  # empty when only files outside the source changed
                (keep / SOURCE_DIFF).write_text(diff)
        except (OSError, subprocess.SubprocessError, RuntimeError) as e:
            log.warning("could not keep the source's changes: %s", e)
    for path in outputs:
        shutil.copy2(path, keep / path.name)
    entry["kept"] = str(keep.relative_to(work))
    entry["outputs"] = file_stamps(writes or outputs)
    with open(work / RUNS, "a") as f:
        f.write(json.dumps(entry) + "\n")


class Progress:
    """Throttled progress logging."""

    def __init__(self, label: str, total: int, every: float = 10.0):
        self.label, self.total, self.every = label, total, every
        self.start = self.last = time.monotonic()
        self.done = 0

    def tick(self, extra: str = "") -> None:
        self.done += 1
        now = time.monotonic()
        if now - self.last >= self.every:
            self.last = now
            rate = self.done / (now - self.start)
            eta = (self.total - self.done) / rate if rate else 0
            log.info("%s %d/%d (%.0f/s, eta %dm%02ds) %s", self.label, self.done, self.total, rate,
                     eta // 60, eta % 60, extra)


STAGE_OUTPUTS = {"register": [TRACK], "composite": [CANVAS, SNAPSHOTS, FINAL], "render": [TIMELAPSE]}


def file_stamps(paths: list[Path]) -> dict[str, list[int]]:
    """Size and modification time of each file, to tell later whether it is still what a run wrote."""
    return {p.name: [(st := p.stat()).st_size, st.st_mtime_ns] for p in paths}


def up_to_date(work: Path, stage: str, inputs: str) -> dict | None:
    """The run that left the stage's outputs in the work directory, if it had these inputs and its
    outputs are unchanged since (a run that failed part way, say, changes them)."""
    last = next((r for r in reversed(runs(work)) if r["stage"] == stage), None)
    if not last or last.get("inputs") != inputs:
        return None
    paths = [work / name for name in STAGE_OUTPUTS[stage]]
    if not all(p.exists() for p in paths) or file_stamps(paths) != {k: last.get("outputs", {}).get(k) for k in STAGE_OUTPUTS[stage]}:
        return None
    return last


def restore_register(work: Path, inputs: str) -> dict | None:
    """Put back the track of an earlier register run with these inputs, from history/."""
    for r in reversed(runs(work)):
        kept = work / r.get("kept", "") / TRACK
        if r["stage"] == "register" and r.get("inputs") == inputs and "kept" in r and kept.exists():
            shutil.copy2(kept, work / TRACK)
            if (kept.parent / DECISIONS).exists():
                shutil.copy2(kept.parent / DECISIONS, work / DECISIONS)
            entry = {"stage": "register", "started": datetime.now().isoformat(timespec="seconds"),
                     "engine": r["engine"], "device": r["device"], "config": r["config"], "inputs": inputs,
                     "outputs": file_stamps([work / TRACK]), "restored": True, "kept": r["kept"], "result": r["result"], "seconds": 0}
            with open(work / RUNS, "a") as f:
                f.write(json.dumps(entry) + "\n")
            return r
    return None


def run_all(work: Path, force: bool = False) -> Path:
    """Register, composite and render. A stage whose inputs (stage_inputs) have not changed since it left
    its outputs in the work directory is skipped, and so is register when an earlier run with the same
    inputs kept its track in history/. Once a stage runs, the ones after it run too; `force` runs every
    stage."""
    cfg = Config.load(work / CONFIG)
    for stage, fn in (("register", run_register), ("composite", run_composite), ("render", run_render)):
        inputs = stage_inputs(cfg, stage, work)
        if not force and (last := up_to_date(work, stage, inputs)):
            log.info("%s: unchanged since the run of %s, skipped", stage, last["started"])
        elif not force and stage == "register" and (old := restore_register(work, inputs)):
            log.info("register: the same as the run of %s, track restored from %s", old["started"], old["kept"])
        else:
            fn(work)
            force = True
    return work / TIMELAPSE


def run_register(work: Path) -> Track:
    cfg = Config.load(work / CONFIG)
    counts = {s: 0 for s in Status}
    prog: Progress | None = None

    def progress(n, total, status, n_inl):
        nonlocal prog
        prog = prog or Progress("register", total)
        counts[status] += 1
        prog.tick(f"at {clock(n / cfg.sample_fps)} " + " ".join(f"{s.name.lower()}={c}" for s, c in counts.items() if c))

    device = register_device(cfg)
    with logged(work, "register", [work / TRACK, work / DECISIONS], device) as run, \
            gzip.open(work / DECISIONS, "wt") as decisions:
        def decided(n: int, record: dict) -> None:
            decisions.write(json.dumps({"n": n, "time": clock(n / cfg.sample_fps), **record}) + "\n")

        track = register(cfg, progress, device, decided)
        track.save(work / TRACK)
        run["result"] = {s.name.lower(): int((track.status == s).sum()) for s in Status}
        found = check.jumps(cfg, track)
        suspect = [j for j in found if j.suspect]
        run["result"]["jumps"] = len(found)
        run["result"]["suspect_jumps"] = [clock(j.n / cfg.sample_fps) for j in suspect]
        for j in suspect:
            log.warning("register: suspect jump at %s", j.line(cfg.sample_fps))
    log.info("register done: %s", run["result"])
    return track


def run_check(work: Path, track_path: Path | None = None, canvas_path: Path | None = None, step: float = 30.0,
              start: float | None = None, end: float | None = None, out: Path | None = None) -> dict:
    """Check a track (the work directory's, or another in the same canvas coordinates) for suspect jumps
    and against a canvas (the final one by default), every `step` seconds; writes check.json."""
    cfg = Config.load(work / CONFIG)
    track = Track.load(track_path or work / TRACK)
    canvas_path = canvas_path or work / FINAL
    lo, hi = sample_range(cfg, len(track.status))
    lo = lo if start is None else max(lo, round(start * cfg.sample_fps))
    hi = hi if end is None else min(hi, round(end * cfg.sample_fps))
    every = max(1, round(step * cfg.sample_fps))
    canvas = cv2.imread(str(canvas_path))
    if canvas is None:
        raise FileNotFoundError(f"{canvas_path}: no canvas to check against (composite first, or --canvas)")
    result = {"track": str(track_path or work / TRACK), "canvas": str(canvas_path), "step": step,
              "from": clock(lo / cfg.sample_fps), "to": clock(hi / cfg.sample_fps), "engine": engine_id(),
              **check.report(cfg, check.jumps(cfg, track, lo, hi), check.agreement(cfg, track, canvas, range(lo, hi, every)))}
    out = out or work / CHECK
    out.write_text(json.dumps(result, indent=1) + "\n")
    log.info("wrote %s", out)
    return result


def run_composite(work: Path) -> None:
    cfg = Config.load(work / CONFIG)
    track = Track.load(work / TRACK)
    info = probe(cfg.video)
    _, size = reference_transform(cfg)
    device = resolve_device(cfg)
    if device == "cuda":
        from .accumulate_gpu import GpuCanvas as canvas_class
    else:
        canvas_class = Canvas
    canvas = canvas_class(size, cfg.accumulate, tuple(cfg.canvas.background), cfg.sample_fps)
    valid = valid_mask(screen_mask(cfg, info.width, info.height, 1.0), cfg.accumulate.erode_px)
    comp = Compositor(canvas, valid)
    sampler = Sampler(cfg.video, cfg.sample_fps, cache=True)
    prev = continued(cfg, work)

    with logged(work, "composite", [work / FINAL], device, [work / CANVAS, work / SNAPSHOTS, work / FINAL]) as run:
        snap_idx, snap_changed = [], []
        pending_changes = 0
        lo, hi = sample_range(cfg, sampler.count)
        placed = [n for n in range(lo, min(hi, len(track.status))) if track.ok(n)]  # only these are decoded
        prog = Progress("composite", len(placed))
        result = {}
        with Encoder(work / CANVAS, *size, fps=30, crf=8, preset="veryfast") as enc:
            if prev:
                prev_img = cv2.imread(str(prev))
                frames = sampler.select(continuation.candidates(placed, cfg.sample_fps))
                m, result["continues"] = continuation.placement(prev_img, frames, track.m, screen_mask(cfg, info.width, info.height, 1.0))
                canvas.seed(*continuation.seeded(prev_img, m, size))
                log.info("composite: starts from %s, placed %s", prev, result["continues"])
                enc.write(canvas.image())  # the first snapshot, shown with no activity of its own
                snap_idx.append(lo)
                snap_changed.append(0)
            for n, frame in prefetch(sampler.select(placed)):
                prog.tick(f"full={comp.full_updates} partial={comp.partial_updates} skipped={comp.skipped} snapshots={len(snap_idx)}")
                res = comp.add(frame, track.m[n])
                pending_changes += res.changed
                if pending_changes >= cfg.accumulate.min_changed_px:
                    enc.write(canvas.image())
                    snap_idx.append(n)
                    snap_changed.append(pending_changes)
                    pending_changes = 0
            final = canvas.image()
            enc.write(final)
            snap_idx.append(hi - 1)
            snap_changed.append(pending_changes)
        np.savez(work / SNAPSHOTS, index=np.array(snap_idx), changed=np.array(snap_changed))
        cv2.imwrite(str(work / FINAL), final)
        run["result"] = {"snapshots": len(snap_idx), "full": comp.full_updates, "partial": comp.partial_updates, **result}
    log.info("composite done: %d snapshots", len(snap_idx))


def pick_frames(index: np.ndarray, changed: np.ndarray, n_out: int, activity_weight: float) -> np.ndarray:
    """Choose which snapshots to show at each of n_out output frames (monotonic, repeats allowed).

    Progress through the timelapse mixes painting activity (changed pixels) with stream time,
    so long idle stretches shrink and busy ones get room.
    """
    act = np.cumsum(changed, dtype=float)
    act = act / act[-1] if act[-1] > 0 else np.linspace(0, 1, len(act))
    t = index.astype(float)
    t = (t - t[0]) / (t[-1] - t[0]) if t[-1] > t[0] else np.linspace(0, 1, len(t))
    u = activity_weight * act + (1 - activity_weight) * t
    targets = np.linspace(0, 1, n_out)
    return np.clip(np.searchsorted(u, targets, side="left"), 0, len(u) - 1)


def render_picks(cfg: Config, snaps) -> np.ndarray:
    """The snapshot (canvas.mp4 frame) shown at each frame of the timelapse, the hold at the end included."""
    rc = cfg.render
    # the stream time the pace mixes in leaves the cuts out
    picks = pick_frames(kept_samples(cfg, snaps["index"]), snaps["changed"], max(2, round(rc.duration * rc.fps)), rc.activity_weight)
    return np.concatenate([picks, np.full(round(rc.hold_end * rc.fps), len(snaps["index"]) - 1)])


def run_render(work: Path, out: Path | None = None) -> Path:
    cfg = Config.load(work / CONFIG)
    rc = cfg.render
    picks = render_picks(cfg, np.load(work / SNAPSHOTS))
    out = out or work / TIMELAPSE
    # render itself runs on the CPU; the device keeps its record with the composite it renders
    with logged(work, "render", [out], resolve_device(cfg)) as run:
        cap = cv2.VideoCapture(str(work / CANVAS))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        with Encoder(out, w, h, rc.fps, rc.crf) as enc:
            i, frame = -1, None
            for p in picks:
                while i < p:
                    ok, nxt = cap.read()
                    if not ok:
                        break
                    frame, i = nxt, i + 1
                enc.write(frame)
        cap.release()
        run["result"] = {"frames": len(picks)}
    log.info("wrote %s (%d frames)", out, len(picks))
    return out
