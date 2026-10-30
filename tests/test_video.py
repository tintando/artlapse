"""Sampling: every pass over a video, wherever it starts, gives the same frame for the same sample
index."""
import subprocess

import numpy as np
import pytest

from artlapse.video import SampleCache, Sampler, prefetch

FPS = 7  # source frame rate, so that every other sample time falls between frames; frame k is a flat grey of level 20 + 3k
FRAMES = 70
SAMPLE_FPS = 2.0


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "64x48",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-qp", "0", "-g", str(2 * FPS),
                             "-pix_fmt", "yuv420p", str(path)],
                            stdin=subprocess.PIPE)
    for k in range(FRAMES):
        proc.stdin.write(np.full((48, 64), 20 + 3 * k, np.uint8).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return path


def source_frame(frame):
    return round((float(frame.mean()) - 20) / 3)  # levels come back within ±1 of what was encoded


def expected(n):
    """The frame on screen at n / fps: the last one shown at or before that time."""
    return int(np.floor(n / SAMPLE_FPS * FPS + 1e-9))


def test_passes_agree_wherever_they_start(clip):
    s = Sampler(clip, SAMPLE_FPS, gray=True)
    for start in range(1, s.count - 2):
        got = [(n, source_frame(f)) for n, f in s.frames(start, start + 2)]
        assert got == [(n, expected(n)) for n in (start, start + 1)]


@pytest.mark.parametrize("read_ahead", [False, True])
def test_backward_chunks_give_the_forward_frames_reversed(clip, read_ahead):
    s = Sampler(clip, SAMPLE_FPS, gray=True)
    forward = [(n, source_frame(f)) for n, f in s.frames(1, s.count)]
    assert forward == [(n, expected(n)) for n in range(1, s.count)]
    backward = s.frames_backward(s.count, 1, chunk=3)
    if read_ahead:
        backward = prefetch(backward, depth=3)
    assert [(n, source_frame(f)) for n, f in backward] == forward[::-1]


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTLAPSE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(SampleCache, "SEGMENT", 8)
    return tmp_path / "cache"


def segments(cache):
    return sorted(p.name for p in cache.glob("*/*.mkv"))


def test_cached_passes_agree_wherever_they_start(clip, cache):
    s = Sampler(clip, SAMPLE_FPS, gray=True, cache=True)
    assert s.count == 20  # segments [0, 8), [8, 16), [16, 20)
    for start in range(1, 16):
        got = [(n, source_frame(f)) for n, f in s.frames(start, start + 1)]
        assert got == [(start, expected(start))]
    assert segments(cache) == []  # passes over less than a quarter of a segment read the source
    forward = [(n, source_frame(f)) for n, f in s.frames(0, s.count)]
    assert forward == [(n, expected(n)) for n in range(s.count)]
    assert segments(cache) == ["000000.mkv", "000001.mkv", "000002.mkv"]
    assert [(n, source_frame(f)) for n, f in s.frames_backward(s.count, 1, chunk=3)] == forward[:0:-1]
    for start in range(1, s.count - 2):
        got = [(n, source_frame(f)) for n, f in s.frames(start, start + 2)]
        assert got == [(n, expected(n)) for n in (start, start + 1)]


@pytest.fixture(scope="module")
def noisy(tmp_path_factory):
    """A lossy colour clip of noise, so that any change in decoding or conversion shows."""
    path = tmp_path_factory.mktemp("video") / "noisy.mp4"
    rng = np.random.default_rng(0)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "96x64",
                             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "30", "-pix_fmt", "yuv420p",
                             "-color_range", "tv", "-colorspace", "bt709", str(path)],
                            stdin=subprocess.PIPE)
    for _ in range(FRAMES):
        proc.stdin.write(rng.integers(0, 256, (64, 96, 3), np.uint8).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return path


@pytest.mark.parametrize("width, gray", [(None, False), (48, True), (48, False)])
def test_cached_frames_are_the_source_frames(noisy, cache, width, gray):
    source = list(Sampler(noisy, SAMPLE_FPS, width, gray).frames())
    cached = Sampler(noisy, SAMPLE_FPS, width, gray, cache=True)
    filling = list(cached.frames())
    assert segments(cache)
    reading = list(cached.frames())
    for got in (filling, reading):
        assert [n for n, _ in got] == [n for n, _ in source]
        assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(got, source))


def test_stopping_early_keeps_no_partial_segment(noisy, cache):
    s = Sampler(noisy, SAMPLE_FPS, cache=True)
    frames = s.frames(0, 8)
    next(frames)
    frames.close()
    assert segments(cache) == []
    assert not list(cache.glob("*/*.tmp"))


def test_caching_can_be_turned_off(noisy, cache, monkeypatch):
    monkeypatch.setenv("ARTLAPSE_CACHE", "off")
    s = Sampler(noisy, SAMPLE_FPS, cache=True)
    assert s.cache is None
    list(s.frames())
    assert not cache.exists()
