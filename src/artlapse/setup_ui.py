"""Browser-based setup: scrub the video and draw the regions the pipeline needs.

Serves the video with HTTP range support, so the browser can seek anywhere in a
multi-hour recording, together with a single page that edits config.yaml. What only the page
uses (the names given to exclusions) goes in setup.json beside it, so the engine never sees it.
The work directory may start empty: the page picks the video from the files on this computer.
A setup exported from the page is config.yaml with the names added, and imports back.

The page can also make the timelapse: `artlapse run` in a child process, whose log the server
reads for the page's progress display and copies to setup-run.log.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import webbrowser
import yaml
from collections import deque
from dataclasses import asdict
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import Config, _build
from .pipeline import CONFIG, TIMELAPSE

EDITABLE = ("workspace", "exclude", "start", "end", "cuts")
PAGE_STATE = "setup.json"
RUN_LOG = "setup-run.log"

# what a run's log lines say about its progress
PROGRESS = re.compile(r"(register|composite) (\d+)/(\d+) \(.*eta (\d+)m(\d+)s\)")
SKIPPED = re.compile(r"(register|composite|render): (unchanged since|the same as)")
FINISHED = re.compile(r"(register|composite) done|wrote .*" + re.escape(TIMELAPSE))
STAGES = ("register", "composite", "render")
# what the file picker offers besides folders
VIDEO_TYPES = {".mp4", ".mkv", ".mov", ".webm", ".flv", ".ts", ".avi", ".m4v"}
VIDEO_MIME = {".webm": "video/webm", ".mkv": "video/webm", ".mov": "video/quicktime"}  # Chrome plays mkv as webm


class Run:
    """The timelapse being made from the page: `artlapse run` in a child process, its log read as it goes."""

    def __init__(self, work: Path):
        self.work = work
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.state = "idle"  # idle, running, done, failed or stopped
        self.lines: deque[str] = deque(maxlen=40)
        self.stages: dict[str, str] = {}  # stage -> running, skipped or done
        self.progress: dict = {}
        self.started = self.ended = None
        self.config_sha = None

    def _config_sha(self) -> str | None:
        path = self.work / CONFIG
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

    def start(self) -> bool:
        with self.lock:
            if self.state == "running":
                return False
            self.state, self.stages, self.progress = "running", {"register": "running"}, {}
            self.lines.clear()
            self.started, self.ended, self.config_sha = time.time(), None, self._config_sha()
            cmd = [sys.executable, "-c", "import sys; from artlapse.cli import main; main(sys.argv[1:])", "run", str(self.work)]
            # its own process group, so stopping it also stops the encoders it starts
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                         start_new_session=os.name == "posix")
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()
        return True

    def _read(self, proc: subprocess.Popen) -> None:
        with open(self.work / RUN_LOG, "a") as log:
            log.write(f"--- {datetime.now().isoformat(timespec='seconds')} artlapse run, from the setup page\n")
            for line in proc.stdout:
                log.write(line)
                log.flush()
                self._parse(proc, line.rstrip("\n"))
            code = proc.wait()
            log.write(f"--- exit {code}\n")
        with self.lock:
            if self.proc is not proc:  # restarted meanwhile: this run's end is not news
                return
            if self.state == "running":
                self.state = "done" if code == 0 else "failed"
            self.ended = time.time()

    def _parse(self, proc: subprocess.Popen, line: str) -> None:
        with self.lock:
            if self.proc is not proc:
                return
            self.lines.append(line)
            if m := PROGRESS.search(line):
                stage, done, total, mins, secs = m.groups()
                self.stages[stage] = "running"
                self.progress = {"stage": stage, "done": int(done), "total": int(total), "eta": int(mins) * 60 + int(secs)}
            elif m := SKIPPED.search(line):
                self._finished(m.group(1), "skipped")
            elif m := FINISHED.search(line):
                self._finished(m.group(1) or "render", "done")

    def _finished(self, stage: str, how: str) -> None:
        self.stages[stage] = how
        nxt = STAGES.index(stage) + 1
        if nxt < len(STAGES):
            self.stages[STAGES[nxt]] = "running"
            self.progress = {}

    def stop(self) -> None:
        with self.lock:
            if self.state != "running":
                return
            self.state = "stopped"
            proc = self.proc
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()

    def status(self) -> dict:
        out = self.work / TIMELAPSE
        with self.lock:
            return {"state": self.state, "stages": dict(self.stages), "progress": dict(self.progress),
                    "started": self.started, "ended": self.ended, "lines": list(self.lines)[-8:],
                    # the config was saved again since this run started: it may not show the latest setup
                    "changed": self.config_sha is not None and self.config_sha != self._config_sha(),
                    "timelapse": out.stat().st_mtime if out.exists() else None,
                    "folder": str(self.work.resolve()), "log": RUN_LOG}


def browse(folder: str | None) -> dict:
    """A folder's subfolders and videos, for the page's file picker; hidden ones left out."""
    path = Path(folder).expanduser() if folder else Path.home()
    path = path.resolve()
    entries = []
    for e in sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):  # folders first
        if e.name.startswith("."):
            continue
        try:
            if e.is_dir():
                entries.append({"name": e.name, "path": str(e), "dir": True})
            elif e.suffix.lower() in VIDEO_TYPES:
                entries.append({"name": e.name, "path": str(e), "dir": False, "size": e.stat().st_size})
        except OSError:  # a broken link, a folder that can't be read
            continue
    return {"dir": str(path), "parent": str(path.parent) if path.parent != path else None,
            "home": str(Path.home()), "entries": entries}


def load_video(cfg_path: Path, video: Path) -> None:
    """Make `video` the work directory's video. The boxes stay when it has the same size as the one
    before; the start, end and cuts belong to the old recording and are reset."""
    from .video import probe
    info = probe(video)
    if cfg_path.exists():
        cfg = Config.load(cfg_path)
        old = probe(cfg.video) if cfg.video and Path(cfg.video).exists() else None
    else:
        cfg, old = Config(), None
    if old is not None and (old.width, old.height) != (info.width, info.height):
        cfg.workspace, cfg.exclude = [0, 0, info.width, info.height], []
        cfg.reference.target = [0, 0, 0, 0]
        names = cfg_path.parent / PAGE_STATE
        if names.exists():
            names.write_text(json.dumps({"names": []}, indent=1) + "\n")
    elif cfg.workspace[2] <= 0:  # a new work directory
        cfg.workspace = [0, 0, info.width, info.height]
    cfg.video = str(video.resolve())
    cfg.start, cfg.end, cfg.cuts = 0.0, None, []
    cfg.reference.time = min(cfg.reference.time, info.duration)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg_path)


def import_setup(cfg_path: Path, text: str) -> tuple[list[str], bool]:
    """Take the settings of an exported setup (or a config.yaml). This work directory's video stays
    if it has one; otherwise the setup's own video is used if it is on this computer. A setup made
    on another video brings its boxes and settings, but not its start, end and cuts. Returns the
    exclusions' names, and whether the setup was made on another video."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("this is not an artlapse setup")
    names = data.pop("names", None) or []
    current = Config.load(cfg_path).video if cfg_path.exists() else ""
    theirs = str(data.get("video") or "")
    if current or not Path(theirs).is_file():
        data["video"] = current
    cfg = _build(Config, data)  # unknown settings are refused before anything is written
    other = bool(current) and theirs != current
    if other:
        cfg.start, cfg.end, cfg.cuts = 0.0, None, []
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg_path)
    return [str(n) for n in names], other


def open_folder(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _handler(work: Path, run: Run):
    cfg_path = work / CONFIG
    page_path = work / PAGE_STATE

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, body: bytes, ctype: str, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status=HTTPStatus.OK):
            self._send(json.dumps(data).encode(), "application/json", status)

        def _names(self) -> list[str]:
            return json.loads(page_path.read_text()).get("names", []) if page_path.exists() else []

        def do_GET(self):
            url = urlparse(self.path)
            path, query = url.path, parse_qs(url.query)
            if path == "/":
                page = resources.files("artlapse").joinpath("setup.html").read_bytes()
                self._send(page, "text/html; charset=utf-8")
            elif path == "/config" and not (cfg_path.exists() and Path(Config.load(cfg_path).video).is_file()):
                self._json({"video": None, "folder": str(work.resolve())})  # the page asks for one
            elif path == "/config":
                cfg = Config.load(cfg_path)
                from .video import probe
                info = probe(cfg.video)
                data = {k: getattr(cfg, k) for k in EDITABLE}
                data["reference"] = {"time": cfg.reference.time, "target": cfg.reference.target}
                data["duration"] = cfg.render.duration
                data["video"] = {"path": cfg.video, "width": info.width, "height": info.height, "duration": info.duration}
                data["names"] = self._names()
                data["folder"] = str(work.resolve())
                self._json(data)
            elif path == "/browse":
                try:
                    self._json(browse(query.get("dir", [None])[0]))
                except OSError as e:
                    self._json({"error": f"can't open {e.filename}: {e.strerror}"}, HTTPStatus.BAD_REQUEST)
            elif path == "/export":
                if not cfg_path.exists():
                    return self._send(b"nothing to export yet", "text/plain", HTTPStatus.NOT_FOUND)
                text = cfg_path.read_text() + yaml.safe_dump({"names": self._names()}, sort_keys=False)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/yaml")
                self.send_header("Content-Disposition", f'attachment; filename="{work.resolve().name}-setup.yaml"')
                body = text.encode()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/example.svg":  # the example setup in "How it works"
                self._send(resources.files("artlapse").joinpath("setup-example.svg").read_bytes(), "image/svg+xml")
            elif path == "/video":
                self._file(Path(Config.load(cfg_path).video) if cfg_path.exists() else work / "no video")
            elif path == "/timelapse":
                self._file(work / TIMELAPSE)
            elif path == "/run":
                self._json(run.status())
            else:
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

        def _file(self, path: Path):
            if not path.exists():
                return self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
            size = path.stat().st_size
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:  # suffix range: last N bytes
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1, start + 8 * 2**20 - 1)  # serve at most 8 MiB per request
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", VIDEO_MIME.get(path.suffix.lower(), "video/mp4"))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            try:
                with path.open("rb") as f:
                    f.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = f.read(min(left, 2**20))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser cancels requests whenever it seeks

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/run":  # start, or restart with the config as it is now
                run.stop()
                run.start()
                return self._json(run.status())
            if path == "/run/stop":
                run.stop()
                return self._json(run.status())
            if path == "/open":
                open_folder(work.resolve())
                return self._json({"ok": True})
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if path == "/video":
                video = Path(json.loads(body)["path"]).expanduser()
                if not video.is_file():
                    return self._json({"error": f"{video} is not a file"}, HTTPStatus.BAD_REQUEST)
                try:
                    load_video(cfg_path, video)
                except Exception as e:  # probe could not read it
                    return self._json({"error": f"can't read {video.name} as a video ({e})"}, HTTPStatus.BAD_REQUEST)
                return self._json({"ok": True})
            if path == "/import":
                try:
                    names, other = import_setup(cfg_path, body.decode())
                except Exception as e:  # not YAML, or not a setup
                    return self._json({"error": f"not an artlapse setup ({e})"}, HTTPStatus.BAD_REQUEST)
                page_path.write_text(json.dumps({"names": names}, indent=1) + "\n")
                return self._json({"ok": True, "other_video": other})
            if path != "/config":
                return self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
            if not cfg_path.exists():
                return self._json({"error": "no video loaded yet"}, HTTPStatus.CONFLICT)
            data = json.loads(body)
            cfg = Config.load(cfg_path)
            for k in EDITABLE:
                if k in data:
                    setattr(cfg, k, data[k])
            if "reference" in data:
                cfg.reference.time = float(data["reference"]["time"])
                cfg.reference.target = [float(v) for v in data["reference"]["target"]]
            if "duration" in data:
                cfg.render.duration = float(data["duration"])
            # unchanged settings are not written again, so a run in progress is not told they changed
            if asdict(cfg) != asdict(Config.load(cfg_path)):
                cfg.save(cfg_path)
            if "names" in data:
                page_path.write_text(json.dumps({"names": data["names"]}, indent=1) + "\n")
            self._json({"ok": True})

    return Handler


def serve(work: Path, port: int = 8765, open_browser: bool = True) -> None:
    run = Run(work)
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(work, run))
    url = f"http://127.0.0.1:{port}/"
    print(f"setup for {work} at {url} (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        run.stop()  # a timelapse started from the page stops with it
        server.server_close()
