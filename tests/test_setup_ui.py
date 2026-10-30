"""The setup page's server: what it saves and where."""
import json
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from artlapse import geometry as geo
from artlapse import pipeline
from artlapse.config import Config
from artlapse.setup_ui import PAGE_STATE, Run, _handler, import_setup

import synth


def start_server(work):
    run = Run(work)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(work, run))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, run


@pytest.fixture
def server(tmp_path):
    s = synth.make(tmp_path / "stream.mp4", count=40)
    Config(video=str(s.path)).save(tmp_path / pipeline.CONFIG)
    httpd, run = start_server(tmp_path)
    yield tmp_path, f"http://127.0.0.1:{httpd.server_address[1]}"
    run.stop()
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def ready(tmp_path):
    """A server for a work directory set up as the page would leave it, ready to make the timelapse."""
    s = synth.make(tmp_path / "stream.mp4", count=40)
    corners = geo.apply(s.views[s.reference], [[0, 0], list(synth.ART)])
    (x0, y0), (x1, y1) = corners.min(axis=0), corners.max(axis=0)
    cfg = Config(video=str(s.path), device="cpu", sample_fps=s.fps, workspace=list(synth.WORKSPACE), exclude=[list(synth.AVATAR)])
    cfg.reference.time = s.reference / s.fps
    cfg.reference.target = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
    cfg.canvas.long_side = 400
    cfg.render.duration = 2
    cfg.save(tmp_path / pipeline.CONFIG)
    httpd, run = start_server(tmp_path)
    yield tmp_path, f"http://127.0.0.1:{httpd.server_address[1]}"
    run.stop()
    httpd.shutdown()
    httpd.server_close()


def post(url, data=None):
    req = urllib.request.Request(url, json.dumps(data or {}).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def get(url):
    return json.loads(urllib.request.urlopen(url).read())


def finished(url, timeout=300):
    end = time.monotonic() + timeout
    while (st := get(f"{url}/run"))["state"] == "running":
        assert time.monotonic() < end, st
        time.sleep(0.5)
    return st


def test_names_go_beside_the_config_not_into_it(server):
    work, url = server
    post(f"{url}/config", {"workspace": [0, 0, 1600, 1200], "exclude": [[10, 20, 30, 40], [50, 60, 70, 80]],
                           "names": ["avatar", "chat"], "start": 0, "end": None,
                           "reference": {"time": 1.5, "target": [100, 100, 400, 300]}})
    cfg = Config.load(work / pipeline.CONFIG)
    assert cfg.exclude == [[10, 20, 30, 40], [50, 60, 70, 80]]
    assert cfg.reference.time == 1.5 and cfg.reference.target == [100, 100, 400, 300]
    assert "avatar" not in (work / pipeline.CONFIG).read_text()
    assert json.loads((work / PAGE_STATE).read_text())["names"] == ["avatar", "chat"]
    got = json.loads(urllib.request.urlopen(f"{url}/config").read())
    assert got["names"] == ["avatar", "chat"] and got["exclude"] == cfg.exclude


def test_a_work_directory_without_page_state_has_no_names(server):
    work, url = server
    assert not (work / PAGE_STATE).exists()
    assert json.loads(urllib.request.urlopen(f"{url}/config").read())["names"] == []


def test_the_page_makes_the_timelapse_and_tells_when_the_setup_changed_since(ready):
    work, url = ready
    assert post(f"{url}/run")["state"] == "running"
    st = finished(url)
    assert st["state"] == "done", st["lines"]
    assert st["stages"] == {"register": "done", "composite": "done", "render": "done"}
    assert not st["changed"] and st["timelapse"]
    video = urllib.request.urlopen(f"{url}/timelapse")
    assert video.headers["Content-Type"] == "video/mp4" and len(video.read()) == (work / pipeline.TIMELAPSE).stat().st_size
    assert "artlapse run, from the setup page" in (work / "setup-run.log").read_text()

    post(f"{url}/config", {"duration": 3})
    assert get(f"{url}/run")["changed"]
    assert Config.load(work / pipeline.CONFIG).render.duration == 3
    post(f"{url}/run")
    st = finished(url)
    assert st["stages"] == {"register": "skipped", "composite": "skipped", "render": "done"} and not st["changed"]


def test_saving_the_same_settings_does_not_count_as_a_change(ready):
    work, url = ready
    post(f"{url}/run")
    finished(url)
    post(f"{url}/config", {"duration": 2, "names": ["avatar"]})
    assert not get(f"{url}/run")["changed"]


def test_a_run_can_be_stopped(ready):
    work, url = ready
    post(f"{url}/run")
    st = post(f"{url}/run/stop")
    assert st["state"] == "stopped"
    time.sleep(0.5)
    assert get(f"{url}/run")["state"] == "stopped"


def test_the_page_and_its_example_figure_are_served(server):
    work, url = server
    page = urllib.request.urlopen(f"{url}/").read().decode()
    assert "How it works" in page and 'src="/example.svg"' in page
    figure = urllib.request.urlopen(f"{url}/example.svg")
    assert figure.headers["Content-Type"] == "image/svg+xml" and figure.read().startswith(b"<svg")


def test_a_new_work_directory_starts_without_a_video_and_loads_one(tmp_path):
    s = synth.make(tmp_path / "stream.mp4", count=40)
    work = tmp_path / "new"
    httpd, run = start_server(work)
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert get(f"{url}/config")["video"] is None
        listing = get(f"{url}/browse?dir={tmp_path}")
        assert {"name": "stream.mp4", "path": str(s.path), "dir": False, "size": s.path.stat().st_size} in listing["entries"]
        assert post(f"{url}/video", {"path": str(s.path)})["ok"]
        got = get(f"{url}/config")
        assert got["video"]["path"] == str(s.path.resolve()) and got["workspace"] == [0, 0, *synth.SCREEN]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_another_video_of_the_same_size_keeps_the_boxes_but_not_the_times(server):
    work, url = server
    other = synth.make(work / "other.mp4", count=30, seed=1)
    post(f"{url}/config", {"exclude": [[10, 20, 30, 40]], "names": ["avatar"], "start": 2, "cuts": [[5, 8]]})
    post(f"{url}/video", {"path": str(other.path)})
    cfg = Config.load(work / pipeline.CONFIG)
    assert cfg.video == str(other.path.resolve()) and cfg.exclude == [[10, 20, 30, 40]]
    assert (cfg.start, cfg.end, cfg.cuts) == (0.0, None, [])


def test_an_exported_setup_imports_back_with_its_names(server, tmp_path_factory):
    work, url = server
    post(f"{url}/config", {"exclude": [[10, 20, 30, 40]], "names": ["avatar"], "cuts": [[5, 8]]})
    exported = urllib.request.urlopen(f"{url}/export")
    assert "attachment" in exported.headers["Content-Disposition"]
    text = exported.read()

    # into a new work directory: the setup's own video comes with it
    httpd, run = start_server(tmp_path_factory.mktemp("elsewhere"))
    other = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        req = urllib.request.Request(f"{other}/import", text)
        assert json.loads(urllib.request.urlopen(req).read()) == {"ok": True, "other_video": False}
        got = get(f"{other}/config")
        assert got["names"] == ["avatar"] and got["exclude"] == [[10, 20, 30, 40]] and got["cuts"] == [[5, 8]]
        assert got["video"]["path"] == Config.load(work / pipeline.CONFIG).video
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_file_that_is_not_a_setup_is_refused_and_changes_nothing(server):
    work, url = server
    before = (work / pipeline.CONFIG).read_text()
    for body in (b"just text", b"colour: blue\n"):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(f"{url}/import", body))
        assert e.value.code == 400
    assert (work / pipeline.CONFIG).read_text() == before


def keys(d, path=""):
    return {path + k for k in d} | {x for k, v in d.items() if isinstance(v, dict) for x in keys(v, f"{path}{k}.")}


def test_a_setup_from_the_website_has_the_same_settings_and_imports(tmp_path):
    web = Path(__file__).parents[1] / "web"
    if not shutil.which("node") or not (web / "node_modules" / "js-yaml").is_dir():
        pytest.skip("needs Node and web/'s npm install")
    script = "import('./src/engine/config.ts').then(m => process.stdout.write(m.toYaml(m.defaults())))"
    text = subprocess.run(["node", "--experimental-strip-types", "--no-warnings", "-e", script],
                          cwd=web, capture_output=True, text=True, check=True).stdout
    Config().save(tmp_path / "python.yaml")
    assert keys(yaml.safe_load(text)) == keys(yaml.safe_load((tmp_path / "python.yaml").read_text()))
    import_setup(tmp_path / pipeline.CONFIG, text)
