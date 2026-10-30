"""Feature backends: keypoints where the features are, in the same pixel convention on both."""
import numpy as np
import pytest

from artlapse import features


def blobs(h=540, w=960, n=120, seed=1):
    """Gaussian blobs at known sub-pixel centres (pixel centres at integer coordinates)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.full((h, w), 128, np.float32)
    rng = np.random.default_rng(seed)
    centres = rng.uniform([40, 40], [w - 40, h - 40], (n, 2))
    for (cx, cy), sigma, sign in zip(centres, rng.uniform(2.5, 6, n), rng.choice([-1, 1], n)):
        img += sign * 90 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return np.clip(img, 0, 255).astype(np.uint8), centres


@pytest.mark.parametrize("device, bias", [("cpu", 0.25), pytest.param("cuda", 0.0, marks=pytest.mark.gpu)])
def test_keypoints_are_where_the_features_are(device, bias):
    """SiftGPU is unbiased once COLMAP's +0.5 pixel convention is undone; OpenCV is 0.25 px off."""
    if device == "cuda" and not features.gpu_available():
        pytest.skip("pycolmap without CUDA")
    img, centres = blobs()
    (pts, desc), = features.backend(device, 1500).detect([(img, np.full(img.shape, 255, np.uint8))])
    nearest = pts[np.argmin(np.linalg.norm(pts[None] - centres[:, None], axis=2), axis=1)]
    offsets = nearest - centres
    offsets = offsets[np.linalg.norm(offsets, axis=1) < 1]
    assert len(offsets) > 60
    assert np.abs(np.median(offsets, axis=0) - bias).max() < 0.05, np.median(offsets, axis=0)
    assert desc.dtype == np.float32 and desc.shape == (len(pts), 128)
