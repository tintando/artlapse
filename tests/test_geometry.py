import math

import numpy as np
import pytest

from artlapse import geometry as geo
from artlapse.config import Config


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("angle", [0.0, 37.0, -120.0])
def test_similarity_roundtrip(flip, angle):
    s, tx, ty = 1.7, 12.0, -40.0
    c, sn = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    m = np.array([[c * s, -sn * s, tx], [sn * s, c * s, ty], [0, 0, 1]])
    if flip:
        m = m @ np.diag([-1.0, 1.0, 1.0])
    d = geo.Similarity.of(m)
    assert d.flipped == flip
    assert d.scale == pytest.approx(s)
    assert d.angle_deg == pytest.approx(angle)
    assert geo.is_similarity(m)


def test_mirror_is_involution():
    m = geo.mirror_x(960)
    assert np.allclose(m @ m, np.eye(3))
    assert np.allclose(geo.apply(m, [[0, 5]]), [[959, 5]])


def test_config_roundtrip(tmp_path):
    cfg = Config(video="x.mp4", exclude=[[1, 2, 3, 4]])
    cfg.register.min_inliers = 7
    cfg.save(tmp_path / "c.yaml")
    back = Config.load(tmp_path / "c.yaml")
    assert back == cfg


def test_config_rejects_unknown_keys(tmp_path):
    (tmp_path / "c.yaml").write_text("video: x\nregister:\n  bogus: 1\n")
    with pytest.raises(ValueError, match="bogus"):
        Config.load(tmp_path / "c.yaml")
