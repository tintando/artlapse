"""The compositing canvas on a CUDA GPU (CuPy), with the same rules as `accumulate.Canvas`.

The canvas arrays live on the GPU. Each frame is uploaded once, warped into the canvas box it
covers, blurred and compared there, and only a two-number summary comes back. The GPU updates
the whole visible box every frame, which costs less than finding the tiles that changed, so the
Compositor skips its tile bookkeeping for this canvas. `image()` downloads the canvas.

The warp, blurs and rules follow the OpenCV calls of the CPU canvas (bilinear warp with a
replicated border, nearest-neighbour mask warp, Gaussian kernels of the same size, reflected
borders, results rounded to 8 bits), so the two agree up to rounding.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import geometry as geo
from . import accumulate
from .accumulate import INK_SIGMA, MARGIN, SEEDED, Canvas, FrameResult
from .config import AccumulateCfg

try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndi
except ImportError:  # the gpu extra is not installed
    cp = None


def available() -> bool:
    """Whether CuPy is installed and sees a CUDA device."""
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


_SOURCE = r"""
extern "C" __global__ void warp(const unsigned char* src, const unsigned char* valid, int sw, int sh,
                                float a, float b, float c, float d, float e, float f,
                                float* out, unsigned char* vout, int ew, int eh) {
    // out: the frame bilinearly sampled at (sx, sy) = (a x + b y + c, d x + e y + f), border replicated;
    // vout: the screen mask, nearest neighbour, 0 outside the screen
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ew * eh) return;
    int x = i % ew, y = i / ew;
    float sx = a * x + b * y + c, sy = d * x + e * y + f;
    int nx = (int)floorf(sx + 0.5f), ny = (int)floorf(sy + 0.5f);
    vout[i] = (nx >= 0 && nx < sw && ny >= 0 && ny < sh) ? valid[ny * sw + nx] : 0;
    float fx0 = floorf(sx), fy0 = floorf(sy);
    float fx = sx - fx0, fy = sy - fy0;
    int x0 = min(max((int)fx0, 0), sw - 1), x1 = min(max((int)fx0 + 1, 0), sw - 1);
    int y0 = min(max((int)fy0, 0), sh - 1), y1 = min(max((int)fy0 + 1, 0), sh - 1);
    const unsigned char* r0 = src + (size_t)y0 * sw * 3;
    const unsigned char* r1 = src + (size_t)y1 * sw * 3;
    for (int k = 0; k < 3; k++) {
        float top = (1 - fx) * r0[3 * x0 + k] + fx * r0[3 * x1 + k];
        float bot = (1 - fx) * r1[3 * x0 + k] + fx * r1[3 * x1 + k];
        out[3 * i + k] = rintf((1 - fy) * top + fy * bot);
    }
}

extern "C" __global__ void rules(const float* fb, const float* cb, const float* f, const unsigned char* v,
                                 const unsigned char* ink, int has_ink, unsigned char* color, float* quality, unsigned char* pending, unsigned char* count,
                                 int ew, int ix, int iy, int iw, int ih, int cw, int x0, int y0,
                                 float q, float qg, int faint_t, int diff_t, float gain, int stable,
                                 unsigned long long* res) {
    // the per-pixel rules of accumulate.Canvas._update; res[0] += committed, res[1] += visible
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool vis = false, committed = false;
    if (i < iw * ih) {
        int x = i % iw, y = i / iw;
        size_t e = (size_t)(y + iy) * ew + (x + ix);
        size_t c = (size_t)(y + y0) * cw + (x + x0);
        vis = v[e] != 0;
        float qual = quality[c];
        int d = 0, dp = 0;
        for (int k = 0; k < 3; k++) {
            d = max(d, abs((int)fb[3 * e + k] - (int)cb[3 * e + k]));
            dp = max(dp, abs((int)f[3 * e + k] - (int)pending[3 * c + k]));
        }
        bool faint = d >= faint_t && (qg >= qual || (has_ink && ink[e]));
        bool agree = qual > 0 && d < diff_t && !faint;
        bool differ = vis && !agree;
        agree = agree && vis;
        unsigned char cnt = count[c];
        bool take = false;
        if (agree) {
            take = q > qual * gain;  // a sharper view of the same thing
            cnt = 0;
        } else if (differ) {
            if (cnt > 0 && dp < diff_t) {
                cnt++;
            } else {
                for (int k = 0; k < 3; k++) pending[3 * c + k] = (unsigned char)f[3 * e + k];
                cnt = 1;
            }
            if (cnt >= stable) {
                take = committed = true;
                cnt = 0;
            }
        }
        if (take) {
            for (int k = 0; k < 3; k++) color[3 * c + k] = (unsigned char)f[3 * e + k];
            quality[c] = q;
        }
        if (agree || differ) count[c] = cnt;
    }
    int n_vis = __syncthreads_count(vis), n_com = __syncthreads_count(committed);
    if (threadIdx.x == 0) {
        if (n_com) atomicAdd(&res[0], (unsigned long long)n_com);
        if (n_vis) atomicAdd(&res[1], (unsigned long long)n_vis);
    }
}
"""

_BLOCK = 256


def _gauss_kernel(sigma: float) -> np.ndarray:
    """The kernel cv2.GaussianBlur uses on 8-bit images for this sigma."""
    ksize = int(round(sigma * 6 + 1)) | 1
    return cv2.getGaussianKernel(ksize, sigma).ravel().astype(np.float32)


class GpuCanvas:
    incremental = False  # the Compositor updates the whole visible box every frame

    def __init__(self, size: tuple[int, int], cfg: AccumulateCfg, background: tuple[int, int, int], sample_fps: float):
        if not available():
            raise RuntimeError("GpuCanvas needs CuPy and a CUDA device (uv sync --extra gpu)")
        w, h = size
        self.size = size
        self.cfg = cfg
        self.stable = int(np.clip(round(cfg.stable_seconds * sample_fps), 1, 255))
        self.background = np.array(background, np.uint8)
        self.color = cp.empty((h, w, 3), cp.uint8)
        self.color[:] = cp.asarray(self.background)
        self.quality = cp.zeros((h, w), cp.float32)
        self.pending = cp.zeros((h, w, 3), cp.uint8)
        self.count = cp.zeros((h, w), cp.uint8)
        module = cp.RawModule(code=_SOURCE)
        self._warp = module.get_function("warp")
        self._rules = module.get_function("rules")
        self._res = cp.zeros(2, cp.uint64)
        self._no_ink = cp.zeros(1, cp.uint8)
        self._frame: cp.ndarray | None = None
        self._pinned: np.ndarray | None = None
        self._valid_host: np.ndarray | None = None
        self._valid: cp.ndarray | None = None

    visible_box = Canvas.visible_box
    _clip = Canvas._clip

    def image(self) -> np.ndarray:
        return cp.asnumpy(self.color)

    def seed(self, image: np.ndarray, known: np.ndarray) -> None:
        known = cp.asarray(known)
        self.color[known] = cp.asarray(image)[known]
        self.quality[known] = SEEDED

    def _upload(self, frame: np.ndarray, valid: np.ndarray) -> None:
        if self._pinned is None or self._pinned.shape != frame.shape:
            mem = cp.cuda.alloc_pinned_memory(frame.nbytes)
            self._pinned = np.frombuffer(mem, np.uint8, frame.size).reshape(frame.shape)
            self._frame = cp.empty(frame.shape, cp.uint8)
        np.copyto(self._pinned, frame)
        self._frame.set(self._pinned)
        if valid is not self._valid_host:  # the Compositor passes the same mask every frame
            self._valid_host, self._valid = valid, cp.asarray(np.ascontiguousarray(valid))

    @staticmethod
    def _blur(img: cp.ndarray, sigma: float) -> cp.ndarray:
        """cv2.GaussianBlur of an 8-bit image (held as float32), rounded back to 8-bit values."""
        k = cp.asarray(_gauss_kernel(sigma))
        out = ndi.correlate1d(img, k, axis=0, mode="mirror")
        out = ndi.correlate1d(out, k, axis=1, mode="mirror")
        return cp.clip(cp.rint(out), 0, 255)

    def add(self, frame: np.ndarray, valid: np.ndarray, m: np.ndarray, boxes=None) -> FrameResult:
        """Merge a full-resolution BGR frame; `valid` is the screen mask, `m` maps screen -> canvas.
        Everything the frame shows is updated (`boxes` is accepted for the Canvas interface only)."""
        box = self.visible_box(valid, m)
        if box is None:
            return FrameResult(0, 0)
        self._upload(frame, valid)
        cfg = self.cfg
        w, h = self.size
        scale = geo.Similarity.of(m).scale
        src = self._frame
        if scale < 0.6:  # zoomed in further than the canvas resolution: low-pass before shrinking
            src = self._blur(src.astype(cp.float32), 0.4 / scale).astype(cp.uint8)
        x0, y0, x1, y1 = box
        ex0, ey0, ex1, ey1 = max(x0 - MARGIN, 0), max(y0 - MARGIN, 0), min(x1 + MARGIN, w), min(y1 + MARGIN, h)
        ew, eh = ex1 - ex0, ey1 - ey0
        inv = np.linalg.inv(geo.scale_translate(1, -ex0, -ey0) @ m)  # box pixel -> screen pixel
        f_ex = cp.empty((eh, ew, 3), cp.float32)
        v_ex = cp.empty((eh, ew), cp.uint8)
        sh, sw = frame.shape[:2]
        n = ew * eh
        self._warp(((n + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
                   (src, self._valid, np.int32(sw), np.int32(sh), *(np.float32(v) for v in inv[:2].ravel()),
                    f_ex, v_ex, np.int32(ew), np.int32(eh)))
        q = min(1.0 / scale, 4.0)
        sigma = max(0.8, 0.5 * scale)
        fb = self._blur(f_ex, sigma)
        cb = self._blur(self.color[ey0:ey1, ex0:ex1].astype(cp.float32), sigma)
        iw, ih = x1 - x0, y1 - y0
        ink = None
        if cfg.ink_threshold > 0 and bool((self.quality[y0:y1, x0:x1] > q * cfg.refine_gain).any()):
            ink = self._ink_changed(f_ex, v_ex, (ex0, ey0, ex1, ey1), scale)
        self._res.fill(0)
        self._rules(((iw * ih + _BLOCK - 1) // _BLOCK,), (_BLOCK,),
                    (fb, cb, f_ex, v_ex, self._no_ink if ink is None else ink, np.int32(ink is not None), self.color, self.quality, self.pending, self.count,
                     np.int32(ew), np.int32(x0 - ex0), np.int32(y0 - ey0), np.int32(iw), np.int32(ih), np.int32(w),
                     np.int32(x0), np.int32(y0), np.float32(q), np.float32(q * cfg.refine_gain),
                     np.int32(cfg.faint_threshold), np.int32(cfg.diff_threshold), np.float32(cfg.refine_gain),
                     np.int32(self.stable), self._res))
        changed, processed = (int(x) for x in self._res.get())
        return FrameResult(changed, processed)

    def _ink_changed(self, f_ex: cp.ndarray, v_ex: cp.ndarray, ebox: tuple[int, int, int, int], scale: float) -> cp.ndarray:
        """accumulate.ink_changed over the extended box, as uint8."""
        ex0, ey0, ex1, ey1 = ebox
        s = INK_SIGMA * scale
        k = cp.asarray(_gauss_kernel(s))

        def blur(a):
            return ndi.correlate1d(ndi.correlate1d(a, k, axis=0, mode="mirror"), k, axis=1, mode="mirror")

        wv = (v_ex > 0).astype(cp.float32)
        norm = blur(wv)
        c_ex = self.color[ey0:ey1, ex0:ex1].astype(cp.float32)
        d = cp.abs(blur(f_ex * wv[..., None]) - blur(c_ex * wv[..., None])).max(axis=2) / cp.maximum(norm, 1e-6)
        changed = ((d >= self.cfg.ink_threshold) & (wv > 0) & (norm > 0.25)).astype(cp.uint8)
        r = 2 * max(1, round(scale)) + 1
        changed = ndi.maximum_filter(ndi.minimum_filter(changed, size=r, mode="constant", cval=1), size=r, mode="constant", cval=0)
        g = math.ceil(accumulate.INK_GROW * scale)
        return cp.ascontiguousarray(ndi.maximum_filter(changed, size=2 * g + 1, mode="constant", cval=0) if g else changed)
