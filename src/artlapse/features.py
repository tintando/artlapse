"""SIFT feature extraction for registration, on the CPU (OpenCV) or a CUDA GPU (COLMAP's SiftGPU).

A run uses one backend throughout: the feature map mixes descriptors from every keyframe, and the
two backends' descriptors are not interchangeable. Both return keypoint positions in image pixels
and float32 descriptors for the FLANN matcher.
"""
from __future__ import annotations

import cv2
import numpy as np

try:
    import pycolmap
except ImportError:  # the gpu extra is not installed
    pycolmap = None


def gpu_available() -> bool:
    """Whether pycolmap is installed with CUDA support (the device itself is checked by CuPy)."""
    return pycolmap is not None and pycolmap.has_cuda


class CpuSift:
    """cv2.SIFT, restricted to the mask. Its keypoints sit 0.25 px right of and below the features
    (OpenCV halves positions found on the image upsampled 2×); correcting that made no difference to
    accuracy on the synthetic streams, so they are used as they are."""

    def __init__(self, max_features: int):
        self.sift = cv2.SIFT_create(nfeatures=max_features)

    def detect(self, images: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray | None]]:
        """For each (grey image, uint8 mask): keypoints (N, 2) and descriptors (N, 128), or None for none."""
        out = []
        for gray, mask in images:
            kps, desc = self.sift.detectAndCompute(gray, mask)
            out.append((np.float32([k.pt for k in kps]).reshape(-1, 2), desc))
        return out


class GpuSift:
    """COLMAP's SiftGPU through pycolmap. It takes no mask: pixels outside the mask are set to flat grey,
    so overlays do not use up the feature budget and give no features, and keypoints within EDGE_PX of
    the mask edge, whose descriptors see the grey, are dropped. Both images of a call are extracted side
    by side in one pass, separated by a grey gap.

    The peak threshold is raised from COLMAP's default so that frames give about as many keypoints as
    cv2.SIFT finds on them. Descriptors are L2-normalised and scaled to bytes, as OpenCV's are."""

    EDGE_PX = 3
    GAP = 64  # grey between images, wider than any descriptor patch at the scales that matter
    PEAK_THRESHOLD = 0.03  # COLMAP's default is 0.0067

    def __init__(self, max_features: int):
        if not gpu_available():
            raise RuntimeError("GPU SIFT needs pycolmap with CUDA (uv sync --extra gpu)")
        self.max_features = max_features
        self._sift: dict[int, object] = {}  # per batch size, since the feature budget is per image
        self._edge: dict[bytes, np.ndarray] = {}

    def _extractor(self, n: int):
        if n not in self._sift:
            o = pycolmap.FeatureExtractionOptions()
            o.use_gpu = True
            o.sift.max_num_features = n * self.max_features
            o.sift.peak_threshold = self.PEAK_THRESHOLD
            o.sift.normalization = pycolmap.Normalization.L2
            self._sift[n] = pycolmap.FeatureExtractor.create(o, pycolmap.Device.cuda)
        return self._sift[n]

    def _inner(self, mask: np.ndarray) -> np.ndarray:
        """The mask shrunk by EDGE_PX (cached: register uses the same mask and its mirror image)."""
        key = mask.tobytes()
        if key not in self._edge:
            k = 2 * self.EDGE_PX + 1
            self._edge[key] = cv2.erode(mask, np.ones((k, k), np.uint8), borderValue=0) > 0
        return self._edge[key]

    def detect(self, images: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray | None]]:
        h, w = images[0][0].shape
        n = len(images)
        sheet = np.full((h, n * w + (n - 1) * self.GAP), 128, np.uint8)
        for i, (gray, mask) in enumerate(images):
            x0 = i * (w + self.GAP)
            sheet[:, x0:x0 + w] = np.where(mask > 0, gray, 128)
        kp, desc = self._extractor(n).extract_from_uint8_array(sheet)
        # COLMAP puts pixel centres at +0.5, OpenCV (and the rest of artlapse) at 0
        kp = np.float32([(k.x - 0.5, k.y - 0.5) for k in kp]).reshape(-1, 2)
        desc = desc.data
        out = []
        for i, (_, mask) in enumerate(images):
            x0 = i * (w + self.GAP)
            pts = kp - np.float32([x0, 0])
            xy = np.round(pts).astype(int)
            inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            keep = np.zeros(len(pts), bool)
            keep[inside] = self._inner(mask)[xy[inside, 1], xy[inside, 0]]
            out.append((pts[keep], np.float32(desc[keep]) if keep.any() else None))
        return out


def backend(device: str, max_features: int) -> CpuSift | GpuSift:
    return GpuSift(max_features) if device == "cuda" else CpuSift(max_features)
