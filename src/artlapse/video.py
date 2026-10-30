"""Frame sampling through ffmpeg.

Samples are addressed by index n at time n / fps, so separate passes over the same
video (forward, chunked backward, full or reduced resolution) agree on which frame is which.

A sample cache keeps a lossless copy of the samples, so later passes over the same video
decode a small 2 fps file instead of the stream recording, and get the same pixels.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import subprocess
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration: float
    pix_fmt: str = "yuv420p"
    color: tuple[tuple[str, str], ...] = ()  # ffmpeg output options that keep the source's colour tags


COLOR_TAGS = {"color_range": "-color_range", "color_space": "-colorspace", "color_transfer": "-color_trc",
              "color_primaries": "-color_primaries"}


def probe(path: str | Path) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,pix_fmt," + ",".join(COLOR_TAGS) + ":format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    s = data["streams"][0]
    color = tuple((opt, s[k]) for k, opt in COLOR_TAGS.items() if s.get(k, "unknown") != "unknown")
    return VideoInfo(int(s["width"]), int(s["height"]), float(data["format"]["duration"]), s.get("pix_fmt", ""), color)


LOSSLESS_FORMATS = {"yuv420p", "yuvj420p", "yuv422p", "yuv444p", "nv12", "yuv420p10le", "yuv422p10le", "yuv444p10le"}


def cache_root() -> Path | None:
    """Where sample caches live: $ARTLAPSE_CACHE, or ~/.cache/artlapse; ARTLAPSE_CACHE=off turns caching off."""
    env = os.environ.get("ARTLAPSE_CACHE")
    if env == "off":
        return None
    if env:
        return Path(env)
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "artlapse"


class SampleCache:
    """Lossless copies of a video's samples, in segments of SEGMENT samples (5 min at 2 fps), each
    written in full by the first pass that decodes it. Holds the source's own pixel format, so a
    frame read from it converts to exactly the frame the source gives (about 3 GB per hour of
    1080p stream). One cache serves every work directory on the same video."""

    SEGMENT = 600

    def __init__(self, directory: Path, video: str | Path, fps: float):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        about = self.dir / "video.json"
        if not about.exists():
            about.write_text(json.dumps({"video": str(video), "sample_fps": fps}))

    @staticmethod
    def for_video(path: str | Path, fps: float, info: VideoInfo) -> "SampleCache | None":
        root = cache_root()
        if root is None or info.pix_fmt not in LOSSLESS_FORMATS:
            return None
        path = Path(path).resolve()
        st = path.stat()
        key = hashlib.sha256(f"{path}|{st.st_size}|{st.st_mtime_ns}|{fps}".encode()).hexdigest()[:16]
        return SampleCache(root / key, path, fps)

    def path(self, segment: int) -> Path:
        return self.dir / f"{segment:06d}.mkv"


class Sampler:
    """Decodes a video at a fixed sample rate, optionally downscaled, as BGR or grey frames. With
    `cache`, samples come from the video's sample cache, and passes over at least a quarter of a
    segment that is not cached yet decode the whole segment and store it."""

    def __init__(self, path: str | Path, fps: float, width: int | None = None, gray: bool = False,
                 cache: bool = False):
        self.path = str(path)
        self.fps = fps
        self.info = probe(path)
        self.width = width or self.info.width
        self.height = round(self.info.height * self.width / self.info.width / 2) * 2
        self.channels = 1 if gray else 3
        self.pix_fmt = "gray" if gray else "bgr24"
        self.cache = SampleCache.for_video(path, fps, self.info) if cache else None

    @property
    def count(self) -> int:
        return math.floor(self.info.duration * self.fps)

    def frames(self, start: int = 0, stop: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (index, frame) for sample indices in [start, stop)."""
        stop = self.count if stop is None else min(stop, self.count)
        if start >= stop:
            return
        if self.cache is None:
            yield from self._decode(start, stop)
            return
        size = self.cache.SEGMENT
        for seg in range(start // size, (stop - 1) // size + 1):
            s0, s1 = seg * size, min((seg + 1) * size, self.count)
            a, b = max(start, s0), min(stop, s1)
            path = self.cache.path(seg)
            if path.exists():
                yield from self._read_cached(path, s0, a, b)
            elif 4 * (b - a) >= min(size, s1 - s0):
                yield from self._decode(a, b, fill=(path, s0, s1))
            else:
                yield from self._decode(a, b)

    def _scale(self) -> str:
        return f"scale={self.width}:{self.height}:flags=area" if self.width != self.info.width else "null"

    def _decode(self, start: int, stop: int, fill: tuple[Path, int, int] | None = None) -> Iterator[tuple[int, np.ndarray]]:
        """Samples [start, stop) from the source; with `fill` = (path, s0, s1), decode the whole segment
        [s0, s1) and also write it to the cache at `path`."""
        lo, hi = (fill[1], fill[2]) if fill else (start, stop)
        # sample n is the frame on screen at n / fps, as a player shows it: the fps filter puts the last
        # frame at or before each sample time in its slot, except slot 0 (the seek time), which it fills
        # with the first frame after the seek; so the pass starts a sample early and skips that slot
        first = max(0, lo - 1)
        sample = f"fps={self.fps}:round=up:start_time=0,trim=start_frame={lo - first}"
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{first / self.fps:.6f}", "-i", self.path, "-an"]
        duration = ["-t", f"{(hi - first) / self.fps:.6f}"]  # an output option, given to each output
        if fill:
            tmp = fill[0].with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            color = [x for opt in self.info.color for x in opt]
            cmd += ["-filter_complex", f"[0:v]{sample},split[keep][out];[out]{self._scale()}[scaled]",
                    "-map", "[keep]", *duration, "-fps_mode", "passthrough", "-c:v", "libx264", "-qp", "0",
                    "-preset", "ultrafast", "-pix_fmt", self.info.pix_fmt, *color, "-f", "matroska", str(tmp),
                    "-map", "[scaled]"]
        else:
            cmd += ["-vf", f"{sample},{self._scale()}"]
        cmd += [*duration, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", self.pix_fmt, "-"]
        try:
            n = yield from self._run(cmd, lo, start, stop, drain=fill is not None)
            if fill and n == hi:
                os.replace(tmp, fill[0])
        finally:
            if fill:
                tmp.unlink(missing_ok=True)

    def _read_cached(self, path: Path, s0: int, start: int, stop: int) -> Iterator[tuple[int, np.ndarray]]:
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-an", "-vf",
               f"trim=start_frame={start - s0}:end_frame={stop - s0},{self._scale()}",
               "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", self.pix_fmt, "-"]
        yield from self._run(cmd, start, start, stop)

    def _run(self, cmd: list[str], first: int, start: int, stop: int, drain: bool = False):
        """Yield the frames of `cmd`'s output, the first of which is sample `first`, for indices in
        [start, stop). With `drain`, read the rest of the output after `stop` too, and wait for ffmpeg to
        finish. Returns the index after the last frame read."""
        size = self.width * self.height * self.channels
        shape = (self.height, self.width) if self.channels == 1 else (self.height, self.width, 3)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=size * 4)
        n = first
        finished = False
        try:
            while n < stop or drain:
                buf = proc.stdout.read(size)
                if len(buf) < size:
                    break
                if start <= n < stop:
                    yield n, np.frombuffer(buf, np.uint8).reshape(shape)
                n += 1
            finished = True
        finally:
            proc.stdout.close()
            if not (drain and finished):
                proc.kill()
            if proc.wait() != 0 and drain and finished:
                n = first  # ffmpeg failed: what it wrote is not a complete segment
        return n

    def select(self, indices: Iterable[int], gap: int = 20) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (index, frame) for sorted sample indices, one pass per stretch of indices less than `gap`
        apart, so long jumps seek instead of decoding everything between."""
        wanted = sorted(set(indices))
        i = 0
        while i < len(wanted):
            j = i
            while j + 1 < len(wanted) and wanted[j + 1] - wanted[j] < gap:
                j += 1
            keep = set(wanted[i:j + 1])
            for n, frame in self.frames(wanted[i], wanted[j] + 1):
                if n in keep:
                    yield n, frame
            i = j + 1

    def frames_backward(self, start: int, stop: int = 0, chunk: int = 240) -> Iterator[tuple[int, np.ndarray]]:
        """Yield frames with indices start-1, start-2, ..., stop, decoding forward in chunks."""
        hi = start
        while hi > stop:
            lo = max(stop, hi - chunk)
            yield from reversed(list(self.frames(lo, hi)))
            hi = lo

    def frame(self, index: int) -> np.ndarray:
        for _, f in self.frames(index, index + 1):
            return f
        raise IndexError(index)


def prefetch(items: Iterable, depth: int = 8) -> Iterator:
    """Iterate `items` on a background thread, keeping up to `depth` ready (overlaps decode with work)."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    done = object()
    stop = threading.Event()

    def worker():
        try:
            for item in items:
                if stop.is_set():
                    break
                q.put(item)
        except BaseException as e:  # surface decode errors in the consumer
            q.put(e)
        q.put(done)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        while (item := q.get()) is not done:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while t.is_alive():  # unblock the worker if it is waiting on a full queue
            try:
                q.get_nowait()
            except queue.Empty:
                t.join(0.05)


def grab(path: str | Path, time: float) -> np.ndarray:
    """One full-resolution BGR frame at `time` seconds."""
    info = probe(path)
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{time:.3f}", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(out, np.uint8).reshape(info.height, info.width, 3)


class Encoder:
    """Pipes BGR frames into an ffmpeg H.264 encode. Frames are handed to ffmpeg on a background
    thread, so the caller only waits when ffmpeg falls behind by more than a few frames; a frame
    must not be modified after it is written."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float, crf: int = 18, preset: str = "medium"):
        self.size = (width, height)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE,
        )
        self.queue: queue.Queue = queue.Queue(maxsize=4)
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._feed, daemon=True)
        self.thread.start()

    def _feed(self) -> None:
        while (frame := self.queue.get()) is not None:
            if self.error is None:
                try:
                    self.proc.stdin.write(np.ascontiguousarray(frame).data)
                except BaseException as e:  # reported to the caller on the next write or close
                    self.error = e

    def write(self, frame: np.ndarray) -> None:
        assert frame.shape[1::-1] == self.size, (frame.shape, self.size)
        if self.error is not None:
            raise RuntimeError("ffmpeg encode failed") from self.error
        self.queue.put(frame)

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        if self.proc.wait() != 0 or self.error is not None:
            raise RuntimeError("ffmpeg encode failed") from self.error

    def __enter__(self) -> "Encoder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
