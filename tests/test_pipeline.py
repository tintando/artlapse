"""End-to-end checks on a synthetic stream with known views and artwork."""
from pathlib import Path

import numpy as np
import pytest

from artlapse import geometry as geo
from artlapse import pipeline
from artlapse.config import Config
from artlapse.register import Status, Track

import synth

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


def set_device(work, device):
    cfg = Config.load(work / pipeline.CONFIG)
    cfg.device = device
    cfg.save(work / pipeline.CONFIG)


def setup_work(work, s, device="cpu"):
    v0 = s.views[s.reference]
    w, h = synth.ART
    corners = geo.apply(v0, [[0, 0], [w, h]])
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    cfg = Config(video=str(s.path), device=device, sample_fps=s.fps, workspace=list(synth.WORKSPACE),
                 exclude=[list(synth.AVATAR)])
    cfg.reference.time = s.reference / s.fps
    cfg.reference.target = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
    cfg.canvas.long_side = w  # canvas pixels == artwork pixels
    cfg.register.gpu_features = device == "cuda"  # register uses SiftGPU only when asked to
    cfg.save(work / pipeline.CONFIG)
    pipeline.run_register(work)
    set_device(work, "cpu")  # later stages choose their own device
    return work, s


def registered(tmp_path_factory, name, device, **stream):
    work = tmp_path_factory.mktemp(f"{name}_{device}")
    return setup_work(work, synth.make(work / "stream.mp4", **stream), device)


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Registered on the CPU; the stream the composite and render tests use."""
    return registered(tmp_path_factory, "work", "cpu")


@pytest.fixture(scope="module", params=DEVICES)
def work_reg(request, tmp_path_factory):
    """The same stream registered on each device."""
    if request.param == "cpu":
        return request.getfixturevalue("work")
    return registered(tmp_path_factory, "work", request.param)


@pytest.fixture(scope="module", params=DEVICES)
def work_transformed(request, tmp_path_factory):
    """The artist scales a large part of the drawing up a third of the way in."""
    return registered(tmp_path_factory, "work_transformed", request.param, transform_at=40)


@pytest.fixture(scope="module", params=DEVICES)
def work_panels(request, tmp_path_factory):
    """Chat and tool panels the user did not exclude sit fixed on screen over the workspace."""
    return registered(tmp_path_factory, "work_panels", request.param, with_panels=True)


@pytest.fixture(scope="module", params=DEVICES)
def work_visitor(request, tmp_path_factory):
    """A large figure the user did not exclude walks over the canvas while the view holds still."""
    return registered(tmp_path_factory, "work_visitor", request.param, with_visitor=True)


@pytest.fixture(scope="module", params=DEVICES)
def work_zooms(request, tmp_path_factory):
    """The view jumps between close-ups and the whole artwork."""
    return registered(tmp_path_factory, "work_zooms", request.param, big_zooms=True)


def corner_error(m_screen_to_canvas, view):
    """Max distance (canvas px) between where the estimate and the truth put the screen corners."""
    w, h = synth.SCREEN
    pts = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
    return float(np.max(np.linalg.norm(geo.apply(m_screen_to_canvas, pts) - geo.apply(np.linalg.inv(view), pts), axis=1)))


def test_registration_tracks_every_view(work_reg):
    work, s = work_reg
    track = Track.load(work / pipeline.TRACK)
    errors = []
    for n in range(s.count):
        if s.views[n] is None:
            continue
        assert track.ok(n), f"sample {n} not registered: {Status(track.status[n]).name}"
        errors.append(corner_error(track.m[n], s.views[n]))
        assert geo.Similarity.of(track.m[n]).flipped == geo.Similarity.of(s.views[n]).flipped
    assert np.median(errors) < 2.0
    assert max(errors) < 6.0, max(errors)


def test_registration_does_not_depend_on_what_ran_before(work):
    """OpenCV's FLANN builds its randomised trees from the process-wide RNG: a second register in the
    same process (a benchmark repeat) must give the same track as the first."""
    import cv2
    from artlapse.register import register

    work, s = work
    cfg = Config.load(work / pipeline.CONFIG)
    first = Track.load(work / pipeline.TRACK)
    cv2.randu(np.empty(1000), 0, 1)  # move the RNG on, as any earlier use of it does
    again = register(cfg)
    assert np.array_equal(again.status, first.status)
    assert np.array_equal(again.m, first.m)


def test_registration_over_part_of_the_range_reproduces_that_part(work):
    """Each sample draws from an RNG seeded by its index, so the forward pass does not depend on how long
    the backward pass before it was: a run over part of the range (a replay) gives the same samples."""
    from artlapse.register import register

    work, s = work
    cfg = Config.load(work / pipeline.CONFIG)
    ref = 40
    corners = geo.apply(s.views[ref], [[0, 0], list(synth.ART)])
    (x0, y0), (x1, y1) = corners.min(axis=0), corners.max(axis=0)
    cfg.reference.time, cfg.reference.target = ref / s.fps, [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
    whole = register(cfg)
    cfg.start = 20 / s.fps
    part = register(cfg)
    assert np.array_equal(part.status[20:], whole.status[20:])
    assert np.array_equal(part.m[20:], whole.m[20:])


def test_register_logs_how_it_decided(work):
    import gzip
    import json

    work, s = work
    records = [json.loads(line) for line in gzip.open(work / pipeline.DECISIONS, "rt")]
    assert records and all(r["time"].count(":") == 2 for r in records)
    matched = [r for r in records if r["status"] == "matched"]
    track = Track.load(work / pipeline.TRACK)
    assert {r["n"] for r in matched} == {n for n in range(s.count) if track.status[n] == Status.MATCHED and n != s.reference}
    for r in matched:
        assert r["views"] and r["won"] in [*range(len(r["views"])), "held"]
        assert all(v[0] < len(r["views"]) for v in r["votes"])
    assert {r["n"] for r in records if r["status"] == "lost"} >= set(s.cut)


def slipped(track: Track, s, shift=(300.0, 0.0)) -> tuple[Track, int]:
    """The track moved by `shift` canvas px from a sample on whose view held still since the sample before."""
    k = next(n for n in range(10, s.count) if track.ok(n - 1) and track.ok(n) and s.views[n] is not None
             and s.views[n - 1] is not None and np.array_equal(s.views[n], s.views[n - 1]))
    out = Track(track.fps, track.status.copy(), track.m.copy(), track.inliers.copy())
    out.status[k] = Status.MATCHED
    out.m[k:] = geo.scale_translate(1, *shift) @ out.m[k:]
    return out, k


def test_check_finds_a_slip(work):
    """A track that moves the view while the frames stay put is flagged where it happened, and disagrees
    with the canvas from there on; the true track is flagged nowhere."""
    from artlapse import check

    work, s = work
    cfg = Config.load(work / pipeline.CONFIG)
    track = Track.load(work / pipeline.TRACK)
    assert not [j.n for j in check.jumps(cfg, track) if j.suspect]
    bad, k = slipped(track, s)
    assert [j.n for j in check.jumps(cfg, bad) if j.suspect] == [k]

    canvas = cv2_bgr(s.final)
    samples = range(0, s.count, 3)
    good = check.agreement(cfg, track, canvas, samples)
    assert len(good) > 0.8 * sum(s.views[n] is not None for n in samples)
    assert not check.stretches(good), check.stretches(good)
    wrong = check.agreement(cfg, bad, canvas, samples)
    runs = check.stretches(wrong)
    assert len(runs) == 1 and k <= runs[0][0] < k + 3 and runs[0][1] >= max(a.n for a in wrong if a.error is not None) - 3


def test_a_cut_is_left_out_and_tracking_picks_up_after_it(work):
    """The samples in a cut are not registered; after it the view is found again, as after a lost stretch."""
    from artlapse.register import register

    work, s = work
    cfg = Config.load(work / pipeline.CONFIG)
    cfg.cuts = [[50 / s.fps, 80 / s.fps]]  # the break screen and the views around it
    track = register(cfg)
    assert (track.status[50:80] == Status.UNSEEN).all()
    errors = []
    for n in [*range(50), *range(80, s.count)]:
        if s.views[n] is not None:
            assert track.ok(n), f"sample {n} not registered: {Status(track.status[n]).name}"
            errors.append(corner_error(track.m[n], s.views[n]))
    assert max(errors) < 6.0, max(errors)


def test_the_reference_frame_cannot_be_cut(work):
    from artlapse.register import register

    work, s = work
    cfg = Config.load(work / pipeline.CONFIG)
    cfg.cuts = [[0, 5]]
    with pytest.raises(ValueError, match="outside the cuts"):
        register(cfg)


def test_a_cut_takes_no_time_in_the_timelapse():
    from artlapse.register import kept_samples

    cfg = Config(sample_fps=1, cuts=[[20, 80]])
    index = np.arange(100)
    picks = pipeline.pick_frames(kept_samples(cfg, index), np.zeros(100), 40, activity_weight=0.0)
    assert not ((picks > 20) & (picks < 80)).any()
    assert abs((picks <= 20).sum() - (picks >= 80).sum()) <= 2  # the 40 samples kept share the 40 frames


def cv2_bgr(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 3 else np.repeat(img[..., None], 3, axis=2)


def test_times_read_as_seconds_or_clock(tmp_path):
    from artlapse.config import clock, seconds

    assert seconds("3:39:37.5") == 13177.5 and seconds("1:30") == 90 and seconds(12) == 12.0 and seconds(None) is None
    assert clock(13177.5) == "3:39:37.5" and clock(59.96) == "0:01:00.0"
    path = tmp_path / "config.yaml"
    path.write_text('start: "0:16:40"\nend: 1:00:00\ncuts: [["0:20:00", 1500]]\nreference: {time: "0:30:00.5"}\n')
    cfg = Config.load(path)
    assert (cfg.start, cfg.end, cfg.reference.time) == (1000.0, 3600.0, 1800.5)
    assert cfg.cuts == [[1200.0, 1500.0]]


def test_registration_follows_document_through_layer_transform(work_transformed):
    work, s = work_transformed
    track = Track.load(work / pipeline.TRACK)
    errors = [corner_error(track.m[n], s.views[n]) for n in range(s.count) if s.views[n] is not None and track.ok(n)]
    registered = sum(track.ok(n) for n in range(s.count) if s.views[n] is not None)
    assert registered >= 0.95 * sum(v is not None for v in s.views)
    assert np.median(errors) < 2.0
    assert max(errors) < 6.0, max(errors)


def test_registration_ignores_panels_fixed_on_screen(work_panels):
    """After a zoom few canvas features survive; panels that stay put must not hold the old view."""
    work, s = work_panels
    track = Track.load(work / pipeline.TRACK)
    errors = [corner_error(track.m[n], s.views[n]) for n in range(s.count) if s.views[n] is not None and track.ok(n)]
    registered = sum(track.ok(n) for n in range(s.count) if s.views[n] is not None)
    assert registered >= 0.95 * sum(v is not None for v in s.views)
    assert np.median(errors) < 2.0
    assert max(errors) < 6.0, max(errors)


def test_registration_follows_big_zoom_jumps(work_zooms):
    """Zooming straight out from a close-up leaves the previous frame covering a small part of the next."""
    work, s = work_zooms
    track = Track.load(work / pipeline.TRACK)
    errors = []
    for n in range(s.count):
        if s.views[n] is None:
            continue
        assert track.ok(n), f"sample {n} not registered: {Status(track.status[n]).name}"
        errors.append(corner_error(track.m[n], s.views[n]))
    assert np.median(errors) < 2.0
    assert max(errors) < 6.0, max(errors)


def test_registration_ignores_a_large_figure_walking_over_a_still_canvas(work_visitor):
    """Its features enter the map with a keyframe; where it moves next, they support a view moved with it.
    That view explains the figure, but not the canvas around it that did not move."""
    work, s = work_visitor
    track = Track.load(work / pipeline.TRACK)
    errors = []
    for n in synth.VISITOR_SAMPLES:
        assert track.ok(n), f"sample {n} not registered: {Status(track.status[n]).name}"
        errors.append(corner_error(track.m[n], s.views[n]))
    assert max(errors) < 6.0, max(errors)


def test_scene_cut_is_skipped(work_reg):
    work, s = work_reg
    track = Track.load(work / pipeline.TRACK)
    assert all(not track.ok(n) for n in s.cut)


@pytest.mark.parametrize("device", DEVICES)
def test_composite_rebuilds_artwork(work, device):
    work, s = work
    set_device(work, device)
    pipeline.run_composite(work)
    set_device(work, "cpu")
    import cv2
    final = cv2.imread(str(work / pipeline.FINAL))
    assert final.shape[:2] == s.final.shape[:2]
    # the views are mostly zoomed out, so compare at the resolution they actually provide
    blur = lambda img: cv2.GaussianBlur(img, (0, 0), 2.0).astype(int)
    err = np.abs(blur(final) - blur(s.final)).max(axis=2)
    # some pixels are never in an unoccluded view; require the bulk to match closely
    assert np.median(err) < 8
    assert (err > 40).mean() < 0.03, (err > 40).mean()


def test_render_produces_video(work):
    work, s = work
    if not (work / pipeline.SNAPSHOTS).exists():
        pipeline.run_composite(work)
    cfg = Config.load(work / pipeline.CONFIG)
    cfg.render.duration, cfg.render.hold_end = 2, 1
    cfg.save(work / pipeline.CONFIG)
    out = pipeline.run_render(work)
    from artlapse.video import probe
    assert probe(out).duration == pytest.approx(3, abs=0.2)


def test_stages_are_recorded_with_their_outputs(work, monkeypatch):
    """Each stage logs its engine and config, and keeps its outputs where a later run cannot erase them."""
    work, s = work
    monkeypatch.chdir(work.parent)  # a work directory given relative, as on the command line
    rel = Path(work.name)
    if not (work / pipeline.SNAPSHOTS).exists():
        pipeline.run_composite(rel)
    pipeline.run_render(rel)
    import json
    runs = [json.loads(line) for line in (work / pipeline.RUNS).read_text().splitlines()]
    assert {"register", "composite", "render"} <= {r["stage"] for r in runs}
    for r in runs:
        assert r["engine"]["source"] and r["config"] and r["seconds"] >= 0
        kept = work / r["kept"]
        assert (kept / pipeline.CONFIG).exists()
    last = {r["stage"]: r for r in runs}
    assert (work / last["register"]["kept"] / pipeline.TRACK).exists()
    assert (work / last["composite"]["kept"] / pipeline.FINAL).exists()
    assert (work / last["render"]["kept"] / pipeline.TIMELAPSE).exists()
    assert last["register"]["result"]["matched"] > 0


def test_run_skips_stages_whose_inputs_did_not_change(work, tmp_path):
    """run redoes only the stages a changed setting reaches, and brings back an earlier register run's
    track when its settings come back."""
    import shutil
    work, s = work
    work = Path(shutil.copytree(work, tmp_path / "copy"))

    def ran():
        return [r["stage"] + ("*" if r.get("restored") else "") for r in pipeline.runs(work)]

    def change(edit):
        cfg = Config.load(work / pipeline.CONFIG)
        edit(cfg)
        cfg.save(work / pipeline.CONFIG)
        before = len(ran())
        pipeline.run_all(work)
        return ran()[before:]

    pipeline.run_all(work)
    assert change(lambda cfg: None) == []
    assert change(lambda cfg: setattr(cfg.render, "duration", 1.5)) == ["render"]
    assert change(lambda cfg: setattr(cfg.accumulate, "stable_seconds", 4.0)) == ["composite", "render"]
    first = Track.load(work / pipeline.TRACK)
    exclude = list(Config.load(work / pipeline.CONFIG).exclude)
    assert change(lambda cfg: cfg.exclude.append([0, 0, 40, 40])) == ["register", "composite", "render"]
    assert change(lambda cfg: setattr(cfg, "exclude", exclude)) == ["register*", "composite", "render"]
    assert np.array_equal(Track.load(work / pipeline.TRACK).m, first.m)
    (work / pipeline.CANVAS).touch()  # as a composite that failed part way leaves it
    assert change(lambda cfg: None) == ["composite", "render"]


def test_stage_inputs_follow_only_the_source_a_stage_runs(tmp_path, monkeypatch):
    """A change to the compositor's source redoes composite and render, not register."""
    import shutil
    src = Path(shutil.copytree(pipeline.SRC, tmp_path / "src"))
    monkeypatch.setattr(pipeline, "SRC", src)
    cfg = Config(video=str(tmp_path / "stream.mp4"), device="cpu")
    before = {s: pipeline.stage_inputs(cfg, s) for s in pipeline.STAGES}
    with open(src / "accumulate.py", "a") as f:
        f.write("# changed\n")
    after = {s: pipeline.stage_inputs(cfg, s) for s in pipeline.STAGES}
    assert [s for s in pipeline.STAGES if before[s] != after[s]] == ["composite", "render"]
    with open(src / "features.py", "a") as f:
        f.write("# changed\n")
    assert pipeline.stage_inputs(cfg, "register") != after["register"]


def test_pick_frames_is_monotonic_and_favours_activity():
    index = np.arange(100)
    changed = np.zeros(100)
    changed[40:50] = 1000  # all activity in a short burst
    picks = pipeline.pick_frames(index, changed, 30, activity_weight=1.0)
    assert np.all(np.diff(picks) >= 0)
    assert ((picks >= 40) & (picks < 50)).sum() >= 25


def test_incremental_matches_full_update(work, monkeypatch):
    """Updating only changed tiles must give the same canvas as updating everything every frame."""
    import cv2
    from artlapse.accumulate import Compositor

    work, s = work
    pipeline.run_composite(work)
    incremental = cv2.imread(str(work / pipeline.FINAL)).astype(int)
    monkeypatch.setattr(Compositor, "_same_view", staticmethod(lambda a, b: False))
    pipeline.run_composite(work)
    full = cv2.imread(str(work / pipeline.FINAL)).astype(int)
    diff = np.abs(incremental - full).max(axis=2)
    assert (diff > 30).mean() < 0.005, (diff > 30).mean()


@pytest.mark.gpu
def test_gpu_canvas_matches_cpu_canvas(work):
    """The GPU canvas applies the same rules as the CPU one, up to rounding."""
    import cv2

    work, s = work
    finals = {}
    for device in ("cpu", "cuda"):
        set_device(work, device)
        pipeline.run_composite(work)
        finals[device] = cv2.imread(str(work / pipeline.FINAL)).astype(int)
    set_device(work, "cpu")
    diff = np.abs(finals["cpu"] - finals["cuda"]).max(axis=2)
    assert (diff > 30).mean() < 0.005, (diff > 30).mean()


def make_canvas(device, size):
    from artlapse.config import AccumulateCfg

    if device == "cuda":
        from artlapse.accumulate_gpu import GpuCanvas as Canvas
    else:
        from artlapse.accumulate import Canvas
    return Canvas(size, AccumulateCfg(), (255, 255, 255), sample_fps=2.0)


@pytest.mark.parametrize("device", DEVICES)
def test_short_lived_overlay_never_commits_but_strokes_do(device):
    """A raid animation that sits still for 3 s stays off the canvas; a stroke that stays is committed."""
    from artlapse.accumulate import Compositor

    w, h = 256, 256
    canvas = make_canvas(device, (w, h))
    comp = Compositor(canvas, np.full((h, w), 255, np.uint8))
    m = np.eye(3)
    blank = np.full((h, w, 3), 255, np.uint8)
    overlay = blank.copy()
    overlay[20:80, 20:80] = (200, 120, 40)
    stroke = blank.copy()
    stroke[150:200, 150:200] = (30, 30, 30)
    for frame in [blank] * 4 + [overlay] * 6 + [blank] * 4 + [stroke] * 20:
        comp.add(frame, m)
        assert (canvas.image()[20:80, 20:80] == 255).all()
    assert (canvas.image()[150:200, 150:200] == 30).all()


@pytest.mark.parametrize("device", DEVICES)
def test_faint_layer_toggles_are_applied(device):
    """A faint layer (a light grey wash) shown and hidden again ends up as it is on screen, without ghosts."""
    from artlapse.accumulate import Compositor

    w, h = 256, 256
    canvas = make_canvas(device, (w, h))
    comp = Compositor(canvas, np.full((h, w), 255, np.uint8))
    blank = np.full((h, w, 3), 255, np.uint8)
    wash = blank.copy()
    wash[64:192, 64:192] = 238
    for frame in [blank] * 4 + [wash] * 20:
        comp.add(frame, np.eye(3))
    assert np.abs(canvas.image()[80:176, 80:176].astype(int) - 238).max() <= 2
    for frame in [blank] * 20:
        comp.add(frame, np.eye(3))
    assert (canvas.image()[80:176, 80:176] >= 253).all()


def test_register_uses_gpu_features_only_when_asked(monkeypatch):
    """On a GPU, composite moves to it by default; register only with register.gpu_features."""
    from artlapse import accumulate_gpu, features

    monkeypatch.setattr(accumulate_gpu, "available", lambda: True)
    monkeypatch.setattr(features, "gpu_available", lambda: True)
    cfg = Config(video="stream.mp4")
    assert (pipeline.resolve_device(cfg), pipeline.register_device(cfg)) == ("cuda", "cpu")
    cfg.register.gpu_features = True
    assert pipeline.register_device(cfg) == "cuda"
    cfg.device = "cpu"
    assert pipeline.register_device(cfg) == "cpu"


def sketch(size, contrast):
    """A rough sketch drawn close up: thin anti-aliased lines, each gone over three times, `contrast`
    darker than a grey background."""
    import cv2

    w, h = size
    img = np.full((h, w, 3), 150, np.uint8)
    rng = np.random.default_rng(1)
    for _ in range(40):
        (x0, y0), (x1, y1) = rng.integers(16, (w - 16, h - 16), (2, 2))
        for _ in range(3):
            dx, dy = rng.integers(-3, 4, 2)
            cv2.line(img, (int(x0 + dx), int(y0 + dy)), (int(x1 - dy), int(y1 + dx)), (150 - contrast,) * 3, 1,
                     cv2.LINE_AA)
    return img


def zoomed_out(art, scale):
    """The art as a view `scale` times further out shows it: one screen px covers `scale` canvas px."""
    import cv2

    return cv2.resize(art, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)


def ink(img):
    return float((150 - img.astype(float)).clip(0).sum())


@pytest.mark.parametrize("device", DEVICES)
def test_faded_sketch_seen_from_further_out_is_applied(device):
    """A sketch drawn close up and then faded (a layer's opacity lowered) is updated from a view further
    out than the one it was drawn in: its strokes lose their ink as a whole."""
    from artlapse.accumulate import Compositor

    size = (384, 384)
    art = sketch(size, 45)
    faded = (150 - (150 - art.astype(float)) * 0.2).round().astype(np.uint8)
    canvas = make_canvas(device, size)
    comp = Compositor(canvas, np.full((384, 384), 255, np.uint8))
    for _ in range(20):
        comp.add(art, np.eye(3))
    assert abs(ink(canvas.image()) - ink(art)) < 0.02 * ink(art)
    view = zoomed_out(faded, 2.5)
    comp.valid = np.full(view.shape[:2], 255, np.uint8)
    comp.valid_bool = comp.valid > 0
    for _ in range(20):
        comp.add(view, geo.scale_translate(2.5))
    assert ink(canvas.image()) < 0.3 * ink(art), ink(canvas.image()) / ink(art)  # 0.2 is what the view shows


@pytest.mark.parametrize("device", DEVICES)
def test_sharp_detail_seen_from_further_out_stays_sharp(device):
    """A view further out shows unchanged detail softer; the canvas keeps the sharp version."""
    from artlapse.accumulate import Compositor

    size = (384, 384)
    art = sketch(size, 45)
    canvas = make_canvas(device, size)
    comp = Compositor(canvas, np.full((384, 384), 255, np.uint8))
    for _ in range(20):
        comp.add(art, np.eye(3))
    sharp = canvas.image().astype(int)
    view = zoomed_out(art, 2.5)
    comp.valid = np.full(view.shape[:2], 255, np.uint8)
    comp.valid_bool = comp.valid > 0
    for _ in range(40):
        comp.add(view, geo.scale_translate(2.5))
    inner = (slice(8, -8), slice(8, -8))
    changed = np.abs(canvas.image().astype(int) - sharp).max(axis=2)[inner] > 4
    assert changed.mean() < 0.01, changed.mean()


@pytest.mark.parametrize("device", DEVICES)
def test_seeded_pixels_give_way_to_the_parts_own_views(device):
    """Pixels from an earlier part's canvas are replaced by the first view that agrees with them, and
    changed by one that differs only once the change has held for stable_seconds."""
    from artlapse.accumulate import Compositor

    w, h = 256, 256
    canvas = make_canvas(device, (w, h))
    known = np.zeros((h, w), bool)
    known[:, :192] = True
    canvas.seed(np.full((h, w, 3), 200, np.uint8), known)
    assert (canvas.image()[:, :192] == 200).all() and (canvas.image()[:, 192:] == 255).all()
    comp = Compositor(canvas, np.full((h, w), 255, np.uint8))
    rng = np.random.default_rng(0)
    frame = (200 + rng.integers(-6, 7, (h, w, 1))).astype(np.uint8).repeat(3, axis=2)
    comp.add(frame, np.eye(3))
    assert (canvas.image()[:, :184] == frame[:, :184]).all()  # next to the unseen part the blurred canvas differs
    stroke = frame.copy()
    stroke[100:150, 50:100] = 40
    for _ in range(canvas.stable - 1):
        comp.add(stroke, np.eye(3))
    assert (canvas.image()[100:150, 50:100] == frame[100:150, 50:100]).all()
    comp.add(stroke, np.eye(3))
    assert (canvas.image()[100:150, 50:100] == 40).all()


def test_placement_finds_the_rescaled_picture():
    """The earlier canvas is placed in this one through the frames that show it and their track."""
    import cv2
    from artlapse import continuation

    prev = synth.artwork(3)
    true = geo.to3(cv2.getRotationMatrix2D((0, 0), 2.0, 0.8)) @ geo.scale_translate(1, 60, 35)  # earlier -> this canvas
    track_m = np.stack([np.linalg.inv(synth.view(z, a, False, cx, cy)) for z, a, cx, cy in
                        [(1.5, 0, 400, 300), (0.7, 10, 800, 600), (2.5, -5, 1200, 900)]])  # screen -> this canvas
    w, h = synth.SCREEN
    frames = [(n, cv2.warpAffine(prev, (np.linalg.inv(m) @ true)[:2], (w, h), borderValue=(70, 70, 70)))
              for n, m in enumerate(track_m)]
    m, info = continuation.placement(prev, frames, track_m, np.full((h, w), 255, np.uint8))
    corners = np.array([[0, 0], [1600, 0], [0, 1200], [1600, 1200]], float)
    assert np.abs(geo.apply(m, corners) - geo.apply(true, corners)).max() < 1.0
    assert info["scale"] == pytest.approx(0.8, abs=1e-3)


def test_continued_part_starts_from_the_earlier_final_canvas(work, tmp_path):
    """With canvas.continues, composite starts from that part's final canvas: it is the first snapshot,
    with no activity of its own, and a new final canvas there redoes the composite."""
    import cv2
    import shutil
    work, s = work
    if not (work / pipeline.FINAL).exists():
        pipeline.run_composite(work)
    earlier = Path(shutil.copytree(work, tmp_path / "a"))
    later = tmp_path / "b"
    later.mkdir()
    shutil.copy2(work / pipeline.CONFIG, later / pipeline.CONFIG)
    shutil.copy2(work / pipeline.TRACK, later / pipeline.TRACK)
    cfg = Config.load(later / pipeline.CONFIG)
    cfg.canvas.continues = "../a"
    cfg.save(later / pipeline.CONFIG)
    before = pipeline.stage_inputs(cfg, "composite", later)
    pipeline.run_composite(later)
    snaps = np.load(later / pipeline.SNAPSHOTS)
    assert snaps["changed"][0] == 0
    first = cv2.VideoCapture(str(later / pipeline.CANVAS)).read()[1]
    cap = cv2.VideoCapture(str(earlier / pipeline.CANVAS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    last = cap.read()[1]  # encoded as the first one is
    blur = lambda img: cv2.GaussianBlur(img, (0, 0), 1.0).astype(int)  # placed to a fraction of a pixel
    assert (np.abs(blur(first) - blur(last)).max(axis=2) > 30).mean() < 0.001
    final = cv2.imread(str(earlier / pipeline.FINAL))
    placed = pipeline.runs(later)[-1]["result"]["continues"]
    assert placed["scale"] == pytest.approx(1, abs=2e-3) and np.abs(placed["shift"]).max() < 1
    cv2.imwrite(str(earlier / pipeline.FINAL), cv2.flip(final, 1))
    assert pipeline.stage_inputs(cfg, "composite", later) != before
    assert pipeline.stage_inputs(cfg, "register", later) == pipeline.stage_inputs(cfg, "register")
