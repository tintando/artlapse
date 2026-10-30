"""artlapse: rebuild a painting timelapse from a stream recording."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

from . import pipeline
from .config import Config, seconds
from .video import probe

log = logging.getLogger("artlapse")

DEFAULT_RECOVER = Path.home() / "projects" / "twitch-recover"


def cmd_init(a) -> None:
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    path = work / pipeline.CONFIG
    if path.exists() and not a.force:
        sys.exit(f"{path} exists (use --force to overwrite)")
    info = probe(a.video)
    cfg = Config(video=str(Path(a.video).resolve()), workspace=[0, 0, info.width, info.height])
    cfg.save(path)
    print(f"wrote {path}; next: artlapse setup {work}")


def cmd_setup(a) -> None:
    from .setup_ui import serve

    serve(Path(a.work), port=a.port, open_browser=not a.no_browser)


def cmd_preview(a) -> None:
    from .visualize import preview_setup

    work = Path(a.work)
    cfg = Config.load(work / pipeline.CONFIG)
    out = Path(a.out or work / "preview.png")
    cv2.imwrite(str(out), preview_setup(cfg, a.time))
    print(out)


def cmd_register(a) -> None:
    pipeline.run_register(Path(a.work))


def cmd_composite(a) -> None:
    pipeline.run_composite(Path(a.work))


def cmd_render(a) -> None:
    print(pipeline.run_render(Path(a.work), Path(a.out) if a.out else None))


def cmd_run(a) -> None:
    print(pipeline.run_all(Path(a.work), force=a.force))


def cmd_check(a) -> None:
    pipeline.run_check(Path(a.work), Path(a.track) if a.track else None, Path(a.canvas) if a.canvas else None,
                       a.step, a.start, a.end, Path(a.out) if a.out else None)


def cmd_overlay(a) -> None:
    from .visualize import overlay_video
    from .register import Track

    work = Path(a.work)
    cfg = Config.load(work / pipeline.CONFIG)
    out = Path(a.out or work / f"overlay_{int(a.start)}_{int(a.end)}.mp4")
    overlay_video(cfg, Track.load(work / pipeline.TRACK), out, a.start, a.end)
    print(out)


def cmd_sidebyside(a) -> None:
    from .visualize import side_by_side

    works = [Path(w) for w in a.work]
    out = Path(a.out or works[0] / "side_by_side.mp4")
    side_by_side(works, out, a.height)
    print(out)


def cmd_cache(a) -> None:
    from .video import cache_root

    root = cache_root()
    if root is None:
        sys.exit("sample caching is off (ARTLAPSE_CACHE=off)")
    videos = {str(Path(v).resolve()) for v in a.video}
    total = 0
    for d in sorted(root.glob("*/")) if root.exists() else []:
        about = json.loads((d / "video.json").read_text()) if (d / "video.json").exists() else {}
        if videos and about.get("video") not in videos:
            continue
        size = sum(f.stat().st_size for f in d.glob("*.mkv"))
        total += size
        print(f"{size / 1e9:6.1f} GB  {len(list(d.glob('*.mkv'))):4d} segments  {about.get('sample_fps')} fps  "
              f"{about.get('video', '?')}")
        if a.clear:
            shutil.rmtree(d)
    print(f"{total / 1e9:6.1f} GB {'removed from' if a.clear else 'in'} {root}")


def cmd_fetch(a) -> None:
    """Recover and download a Twitch VOD with the twitch-recover tool, then init a work dir for it."""
    recover = Path(a.recover_dir)
    script = recover / "recover.py"
    if not script.exists():
        sys.exit(f"twitch-recover not found at {recover} (use --recover-dir)")
    python = recover / ".venv" / "bin" / "python"
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    video = work / "stream.mp4"
    cmd = [str(python if python.exists() else shutil.which("python3")), str(script), a.url,
           "--download", "--output", str(video.resolve())]
    if a.timestamp:
        cmd += ["--timestamp", a.timestamp]
    subprocess.run(cmd, check=True)
    a.video, a.force = str(video), False
    cmd_init(a)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="artlapse", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create a work directory for a video")
    s.add_argument("video")
    s.add_argument("work")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("fetch", help="recover a Twitch VOD from a TwitchTracker URL and init a work directory")
    s.add_argument("url")
    s.add_argument("work")
    s.add_argument("--timestamp", help="UTC start 'YYYY-MM-DD HH:MM:SS' if TwitchTracker blocks scraping")
    s.add_argument("--recover-dir", default=str(DEFAULT_RECOVER))
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("setup", help="pick workspace, overlays and target area in the browser")
    s.add_argument("work")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=cmd_setup)

    s = sub.add_parser("preview", help="draw the configured regions on a frame")
    s.add_argument("work")
    s.add_argument("--time", type=seconds, help="seconds or h:mm:ss")
    s.add_argument("-o", "--out")
    s.set_defaults(fn=cmd_preview)

    for name, fn, help_ in [("register", cmd_register, "track the canvas through the video"),
                            ("composite", cmd_composite, "rebuild the canvas over time")]:
        s = sub.add_parser(name, help=help_)
        s.add_argument("work")
        s.set_defaults(fn=fn)

    s = sub.add_parser("run", help="register, composite and render, skipping stages whose inputs did not change")
    s.add_argument("work")
    s.add_argument("--force", action="store_true", help="run every stage")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("check", help="look for slips in the track: jumps the frames do not support, and "
                                     "disagreement with the final canvas")
    s.add_argument("work")
    s.add_argument("--step", type=float, default=30.0, help="seconds between the samples matched against the canvas")
    s.add_argument("--from", dest="start", type=seconds, help="seconds or h:mm:ss")
    s.add_argument("--to", dest="end", type=seconds)
    s.add_argument("--track", help="another track in the same canvas coordinates (a copy from history/, say)")
    s.add_argument("--canvas", help="another canvas image framed like this one (default: final.png)")
    s.add_argument("-o", "--out", help="default: check.json in the work directory")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("render", help="encode the timelapse")
    s.add_argument("work")
    s.add_argument("-o", "--out")
    s.set_defaults(fn=cmd_render)

    s = sub.add_parser("overlay", help="video of a time range with the tracked canvas outline")
    s.add_argument("work")
    s.add_argument("start", type=seconds, help="seconds or h:mm:ss")
    s.add_argument("end", type=seconds)
    s.add_argument("-o", "--out")
    s.set_defaults(fn=cmd_overlay)

    s = sub.add_parser("sidebyside", help="the stream next to its timelapse, in sync")
    s.add_argument("work", nargs="+", help="work directories, played one after another (parts of one stream)")
    s.add_argument("--height", type=int, default=1080)
    s.add_argument("-o", "--out")
    s.set_defaults(fn=cmd_sidebyside)

    s = sub.add_parser("cache", help="list the sample caches (lossless copies of the samples), or remove them")
    s.add_argument("video", nargs="*", help="only the caches of these videos")
    s.add_argument("--clear", action="store_true")
    s.set_defaults(fn=cmd_cache)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    a.fn(a)
