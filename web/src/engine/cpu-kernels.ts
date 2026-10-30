/**
 * The processor fallback's kernels: each one does its part of a pass, rows (or keypoints, or queries) i0
 * to i1, so a team of workers (team.ts) can share a pass. They read and write typed arrays on shared memory
 * that every worker of the team sees; what they return is small (candidates, counts) and is gathered in
 * order. The kernels follow the WGSL ones in sift.ts, match.ts, frames.ts and composite.ts step by step.
 */
import { BORDER, CAP, CONTRAST, EDGE, LAYERS, SIGMA, THRESHOLD } from './sift';

/** WGSL's round (and cvRound): halves to even. */
export function rne(v: number): number {
  const r = Math.round(v);
  return r - v === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export function reflect101(i: number, n: number): number {
  if (n === 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
}

/** atan2 in degrees, in [0, 360) as cv::fastAtan2. */
function deg(dy: number, dx: number): number {
  let a = Math.atan2(dy, dx) * (180 / Math.PI);
  if (a < 0) a += 360;
  if (a >= 360) a = 0;
  return a;
}

const clamp8 = (v: number) => Math.min(Math.max(rne(v), 0), 255);

/** A level of the SIFT pyramid: its offset in the pyramid's buffer and its size. */
export interface Level { off: number; w: number; h: number }

type F32 = Float32Array;
type U8 = Uint8Array | Uint8ClampedArray;

export const kernels = {
  // ---- SIFT (sift.ts) ----

  /** The grey frame doubled (cv::resize INTER_LINEAR), mirrored for a flipped state; rows of the result. */
  upsample(a: { gray: Uint8Array; w: number; h: number; flip: boolean; dst: F32 }, y0: number, y1: number) {
    const { gray, w, h, flip, dst } = a, W = 2 * w;
    // source coordinate (d + 0.5) / 2 - 0.5, clamped to the image
    const tap = (d: number, n: number): [number, number, number] => {
      let fr = Math.fround((d + 0.5) * 0.5 - 0.5);
      let s = Math.floor(fr);
      fr -= s;
      if (s < 0) { fr = 0; s = 0; }
      if (s >= n - 1) { fr = 0; s = n - 1; }
      return [s, Math.min(s + 1, n - 1), fr];
    };
    const xs = Array.from({ length: W }, (_, x) => tap(x, w));
    const px = (x: number, y: number) => gray[y * w + (flip ? w - 1 - x : x)];
    for (let y = y0; y < y1; y++) {
      const [ya, yb, fy] = tap(y, h);
      for (let x = 0; x < W; x++) {
        const [xa, xb, fx] = xs[x];
        const top = px(xa, ya) * (1 - fx) + px(xb, ya) * fx;
        const bot = px(xa, yb) * (1 - fx) + px(xb, yb) * fx;
        dst[y * W + x] = top * (1 - fy) + bot * fy;
      }
    }
  },

  /** Blur along x of a float image (w wide) at sOff, into dst at dOff; rows y0 to y1. The kernel is symmetric:
   * each pair of taps around the centre is summed before it is weighted. */
  blurH(a: { src: F32; sOff: number; dst: F32; dOff: number; w: number; k: F32 }, y0: number, y1: number) {
    const { src, sOff, dst, dOff, w, k } = a, half = (k.length - 1) / 2, kc = k[half];
    for (let y = y0; y < y1; y++) {
      const row = sOff + y * w, out = dOff + y * w;
      for (let x = 0; x < w; x++) {
        const c = row + x;
        let s = kc * src[c];
        if (x >= half && x < w - half) {
          for (let t = 1; t <= half; t++) s += k[half - t] * (src[c - t] + src[c + t]);
        } else {
          for (let t = 1; t <= half; t++) s += k[half - t] * (src[row + reflect101(x - t, w)] + src[row + reflect101(x + t, w)]);
        }
        dst[out + x] = s;
      }
    }
  },

  /** Blur along y (an image h high), as blurH. */
  blurV(a: { src: F32; sOff: number; dst: F32; dOff: number; w: number; h: number; k: F32 }, y0: number, y1: number) {
    const { src, sOff, dst, dOff, w, h, k } = a, half = (k.length - 1) / 2, kc = k[half];
    const acc = new Float64Array(w);
    for (let y = y0; y < y1; y++) {
      const row = sOff + y * w;
      for (let x = 0; x < w; x++) acc[x] = kc * src[row + x];
      for (let t = 1; t <= half; t++) {
        const kt = k[half - t], p = sOff + reflect101(y - t, h) * w, q = sOff + reflect101(y + t, h) * w;
        for (let x = 0; x < w; x++) acc[x] += kt * (src[p + x] + src[q + x]);
      }
      const out = dOff + y * w;
      for (let x = 0; x < w; x++) dst[out + x] = acc[x];
    }
  },

  /** cv::resize INTER_NEAREST from one level to the next octave's base: source pixel floor(d * src / dst). */
  downsample(a: { pyr: F32; src: Level; dst: Level }, y0: number, y1: number) {
    const { pyr, src: s, dst: d } = a;
    for (let y = y0; y < y1; y++) {
      const sy = Math.floor((y * s.h) / d.h);
      for (let x = 0; x < d.w; x++) pyr[d.off + y * d.w + x] = pyr[s.off + sy * s.w + Math.floor((x * s.w) / d.w)];
    }
  },

  /** The differences of an octave's Gaussian levels (levels[l + 1] - levels[l] into dog[l]); rows y0 to y1. */
  dog(a: { pyr: F32; levels: Level[]; dog: F32; dogs: number[] }, y0: number, y1: number) {
    const { pyr, levels, dog, dogs } = a, w = levels[0].w;
    for (let l = 0; l < dogs.length; l++) {
      const p = levels[l].off, q = levels[l + 1].off, d = dogs[l];
      for (let i = y0 * w; i < y1 * w; i++) dog[d + i] = pyr[q + i] - pyr[p + i];
    }
  },

  /** The candidates of one octave in rows y0 to y1: extrema refined as adjustLocalExtrema does, 8 numbers each
   * (x, y, response, size, octave | layer << 8, row, column, layer offset). */
  extrema(a: { dog: F32; dogs: number[]; w: number; h: number; o: number }, y0: number, y1: number): number[] {
    const { dog, dogs, w, h, o } = a, out: number[] = [];
    const D = (l: number, r: number, c: number) => dog[dogs[l] + r * w + c];
    const imgScale = 1 / 255, ds = imgScale * 0.5, s2 = imgScale, cs = imgScale * 0.25;
    const det3 = (a0: number, b: number, c: number, d: number, e: number, f: number, g: number, hh: number, k: number) =>
      a0 * (e * k - f * hh) - b * (d * k - f * g) + c * (d * hh - e * g);
    for (let layer0 = 1; layer0 <= LAYERS; layer0++) {
      const cur = dogs[layer0], prev = dogs[layer0 - 1], next = dogs[layer0 + 1];
      for (let r0 = Math.max(y0, BORDER); r0 < Math.min(y1, h - BORDER); r0++) {
        for (let c0 = BORDER; c0 < w - BORDER; c0++) {
          const i0 = r0 * w + c0, val = dog[cur + i0];
          if (Math.abs(val) <= THRESHOLD) continue;
          let ext = true;
          for (let dr = -1; dr <= 1 && ext; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
              const j = i0 + dr * w + dc;
              const p = dog[prev + j], n = dog[next + j], c = dog[cur + j];
              if (val > 0 ? (val < p || val < n || (j !== i0 && val < c)) : (val > p || val > n || (j !== i0 && val > c))) { ext = false; break; }
            }
          }
          if (!ext) continue;
          let r = r0, c = c0, layer = layer0, xi = 0, xr = 0, xc = 0, i = 0, ok = true;
          for (;;) {
            if (i >= 5) { ok = false; break; }
            const l = layer, v2 = D(l, r, c) * 2;
            const dx = (D(l, r, c + 1) - D(l, r, c - 1)) * ds, dy = (D(l, r + 1, c) - D(l, r - 1, c)) * ds;
            const dsg = (D(l + 1, r, c) - D(l - 1, r, c)) * ds;
            const dxx = (D(l, r, c + 1) + D(l, r, c - 1) - v2) * s2, dyy = (D(l, r + 1, c) + D(l, r - 1, c) - v2) * s2;
            const dss = (D(l + 1, r, c) + D(l - 1, r, c) - v2) * s2;
            const dxy = (D(l, r + 1, c + 1) - D(l, r + 1, c - 1) - D(l, r - 1, c + 1) + D(l, r - 1, c - 1)) * cs;
            const dxs = (D(l + 1, r, c + 1) - D(l + 1, r, c - 1) - D(l - 1, r, c + 1) + D(l - 1, r, c - 1)) * cs;
            const dys = (D(l + 1, r + 1, c) - D(l + 1, r - 1, c) - D(l - 1, r + 1, c) + D(l - 1, r - 1, c)) * cs;
            // H X = dD by Cramer's rule (H symmetric)
            const det = det3(dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss);
            let X0 = 0, X1 = 0, X2 = 0;
            if (det !== 0) {
              X0 = det3(dx, dxy, dxs, dy, dyy, dys, dsg, dys, dss) / det;
              X1 = det3(dxx, dx, dxs, dxy, dy, dys, dxs, dsg, dss) / det;
              X2 = det3(dxx, dxy, dx, dxy, dyy, dy, dxs, dys, dsg) / det;
            }
            xi = -X2; xr = -X1; xc = -X0;
            if (Math.abs(xi) < 0.5 && Math.abs(xr) < 0.5 && Math.abs(xc) < 0.5) break;
            if (Math.abs(xi) > 7e8 || Math.abs(xr) > 7e8 || Math.abs(xc) > 7e8) { ok = false; break; }
            c += rne(xc); r += rne(xr); layer += rne(xi);
            if (layer < 1 || layer > LAYERS || c < BORDER || c >= w - BORDER || r < BORDER || r >= h - BORDER) { ok = false; break; }
            i++;
          }
          if (!ok) continue;
          const l = layer;
          const dx = (D(l, r, c + 1) - D(l, r, c - 1)) * ds, dy = (D(l, r + 1, c) - D(l, r - 1, c)) * ds;
          const dsg = (D(l + 1, r, c) - D(l - 1, r, c)) * ds;
          const contr = D(l, r, c) * imgScale + (dx * xc + dy * xr + dsg * xi) * 0.5;
          if (Math.abs(contr) * LAYERS < CONTRAST) continue;
          const v2 = D(l, r, c) * 2;
          const dxx = (D(l, r, c + 1) + D(l, r, c - 1) - v2) * s2, dyy = (D(l, r + 1, c) + D(l, r - 1, c) - v2) * s2;
          const dxy = (D(l, r + 1, c + 1) - D(l, r + 1, c - 1) - D(l, r - 1, c + 1) + D(l, r - 1, c - 1)) * cs;
          const tr = dxx + dyy, det = dxx * dyy - dxy * dxy;
          if (det <= 0 || tr * tr * EDGE >= (EDGE + 1) * (EDGE + 1) * det) continue;
          if (out.length / 8 >= CAP) return out;
          out.push(c + xc, r + xr, Math.abs(contr), SIGMA * 2 ** ((layer + xi) / LAYERS) * (1 << o) * 2, o | (layer << 8), r, c, xi);
        }
      }
    }
    return out;
  },

  /** The orientations of candidates i0 to i1: a keypoint for every histogram peak within 80% of the highest,
   * 6 numbers each (x, y, response, size, angle, octave | layer << 8). */
  orient(a: { pyr: F32; levels: Level[][]; cands: Float64Array }, i0: number, i1: number): number[] {
    const { pyr, levels, cands } = a, out: number[] = [];
    const hist = new Float64Array(36), t = new Float64Array(40), hs = new Float64Array(36);
    for (let ci = i0; ci < i1; ci++) {
      const b = ci * 8, kx = cands[b], ky = cands[b + 1], resp = cands[b + 2], size = cands[b + 3], olay = cands[b + 4];
      const kr = cands[b + 5], kc = cands[b + 6];
      const o = olay & 255, { off, w, h } = levels[o][olay >> 8];
      const scl = size * 0.5 / (1 << o);
      const radius = rne(4.5 * scl), sig = 1.5 * scl, es = -1 / (2 * sig * sig);
      hist.fill(0);
      for (let i = -radius; i <= radius; i++) {
        const y = kr + i;
        if (y <= 0 || y >= h - 1) continue;
        for (let j = -radius; j <= radius; j++) {
          const x = kc + j;
          if (x <= 0 || x >= w - 1) continue;
          const p = off + y * w + x;
          const dx = pyr[p + 1] - pyr[p - 1], dy = pyr[p - w] - pyr[p + w];
          const wt = Math.exp((i * i + j * j) * es);
          let bin = rne((36 / 360) * deg(dy, dx));
          if (bin >= 36) bin -= 36;
          if (bin < 0) bin += 36;
          hist[bin] += wt * Math.sqrt(dx * dx + dy * dy);
        }
      }
      for (let i = 0; i < 36; i++) t[i + 2] = hist[i];
      t[0] = t[36]; t[1] = t[37]; t[38] = t[2]; t[39] = t[3];
      let omax = 0;
      for (let i = 0; i < 36; i++) {
        hs[i] = (t[i] + t[i + 4]) * (1 / 16) + (t[i + 1] + t[i + 3]) * (4 / 16) + t[i + 2] * (6 / 16);
        omax = Math.max(omax, hs[i]);
      }
      const thr = omax * 0.8;
      for (let j = 0; j < 36; j++) {
        const l = j > 0 ? j - 1 : 35, r2 = j < 35 ? j + 1 : 0;
        if (hs[j] > hs[l] && hs[j] > hs[r2] && hs[j] >= thr) {
          let bin = j + 0.5 * (hs[l] - hs[r2]) / (hs[l] - 2 * hs[j] + hs[r2]);
          if (bin < 0) bin += 36; else if (bin >= 36) bin -= 36;
          let angle = 360 - 10 * bin;
          if (Math.abs(angle - 360) < 1.1920929e-7) angle = 0;
          out.push(kx, ky, resp, size, angle, olay);
        }
      }
    }
    return out;
  },

  /** 4x4x8 descriptors of keypoints i0 to i1 (the descriptor stage's input records, sift.ts) into out. */
  describe(a: { pyr: F32; levels: Level[][]; input: ArrayBuffer; out: Uint8Array }, i0: number, i1: number) {
    const { pyr, levels, out } = a, f = new Float32Array(a.input), u = new Uint32Array(a.input);
    const hist = new Float64Array(360), raw = new Float64Array(128);
    for (let ki = i0; ki < i1; ki++) {
      const kx = f[ki * 8], ky = f[ki * 8 + 1], ori = f[ki * 8 + 2], kscl = f[ki * 8 + 3], olay = u[ki * 8 + 4];
      const { off, w, h } = levels[olay & 255][olay >> 8];
      const px = rne(kx), py = rne(ky);
      const histWidth = 3 * kscl;
      let radius = rne(histWidth * Math.SQRT2 * 5 * 0.5);
      radius = Math.min(radius, Math.trunc(Math.sqrt(w * w + h * h)));
      const cosT = Math.cos(ori * 0.017453292519943295) / histWidth, sinT = Math.sin(ori * 0.017453292519943295) / histWidth;
      hist.fill(0);
      for (let i = -radius; i <= radius; i++) {
        const r = py + i;
        if (r <= 0 || r >= h - 1) continue;
        for (let j = -radius; j <= radius; j++) {
          const cRot = j * cosT - i * sinT, rRot = j * sinT + i * cosT;
          let rbin = rRot + 2 - 0.5, cbin = cRot + 2 - 0.5;
          const c = px + j;
          if (!(rbin > -1 && rbin < 4 && cbin > -1 && cbin < 4 && c > 0 && c < w - 1)) continue;
          const p = off + r * w + c;
          const dx = pyr[p + 1] - pyr[p - 1], dy = pyr[p - w] - pyr[p + w];
          let obin = (deg(dy, dx) - ori) * (8 / 360);
          const mag = Math.sqrt(dx * dx + dy * dy) * Math.exp((cRot * cRot + rRot * rRot) * (-1 / 8));
          const r0 = Math.floor(rbin), c0 = Math.floor(cbin);
          let o0 = Math.floor(obin);
          rbin -= r0; cbin -= c0; obin -= o0;
          if (o0 < 0) o0 += 8;
          if (o0 >= 8) o0 -= 8;
          const vr1 = mag * rbin, vr0 = mag - vr1;
          const v11 = vr1 * cbin, v10 = vr1 - v11, v01 = vr0 * cbin, v00 = vr0 - v01;
          const v111 = v11 * obin, v101 = v10 * obin, v011 = v01 * obin, v001 = v00 * obin;
          const idx = ((r0 + 1) * 6 + c0 + 1) * 10 + o0;
          hist[idx] += v00 - v001; hist[idx + 1] += v001;
          hist[idx + 10] += v01 - v011; hist[idx + 11] += v011;
          hist[idx + 60] += v10 - v101; hist[idx + 61] += v101;
          hist[idx + 70] += v11 - v111; hist[idx + 71] += v111;
        }
      }
      for (let t = 0; t < 128; t++) {
        const cell = t >> 3, ob = t & 7;
        const idx = (((cell >> 2) + 1) * 6 + (cell & 3) + 1) * 10;
        raw[t] = hist[idx + ob] + (ob < 2 ? hist[idx + 8 + ob] : 0);
      }
      let nrm2 = 0;
      for (let t = 0; t < 128; t++) nrm2 += raw[t] * raw[t];
      const thr = Math.sqrt(nrm2) * 0.2;
      nrm2 = 0;
      for (let t = 0; t < 128; t++) { const v = Math.min(raw[t], thr); raw[t] = v; nrm2 += v * v; }
      const scale = 512 / Math.max(Math.sqrt(nrm2), 1.1920929e-7);
      for (let t = 0; t < 128; t++) out[ki * 128 + t] = clamp8(raw[t] * scale);
    }
  },

  // ---- matching (match.ts) ----

  /** For queries q0 to q1: [best index, best squared distance, second index, second squared distance] into
   * out, exactly as match.ts's shader finds them (the first of equal distances wins). */
  knn(a: { query: Uint8Array; map: Uint8Array; nm: number; out: Uint32Array }, q0: number, q1: number) {
    const { query, map, nm, out } = a;
    for (let q = q0; q < q1; q++) {
      const qo = q * 128;
      let b1 = Infinity, b2 = Infinity, i1 = 0, i2 = 0;
      for (let j = 0; j < nm; j++) {
        const mo = j * 128;
        let d = 0;
        // a partial sum already at the second best can't change either: stop early
        for (let k = 0; k < 128 && d < b2; k += 32) {
          for (let t = k; t < k + 32; t++) { const e = query[qo + t] - map[mo + t]; d += e * e; }
        }
        if (d < b1) { b2 = b1; i2 = i1; b1 = d; i1 = j; }
        else if (d < b2) { b2 = d; i2 = j; }
      }
      out[q * 4] = i1; out[q * 4 + 1] = Math.min(b1, 0xffffffff);
      out[q * 4 + 2] = i2; out[q * 4 + 3] = Math.min(b2, 0xffffffff);
    }
  },

  // ---- analysis frames (frames.ts) ----

  /** Luma of source rows y0 to y1 of an RGBA frame, shrunk along x by area into rows (w wide). */
  grayH(a: { rgba: U8; sw: number; w: number; k: number[]; rows: F32 }, y0: number, y1: number) {
    const { rgba, sw, w, rows } = a, [kr, kg, kb] = a.k, f = sw / w;
    for (let y = y0; y < y1; y++) {
      for (let ox = 0; ox < w; ox++) {
        const xa = ox * f, xb = xa + f;
        let s = 0;
        for (let sx = Math.floor(xa); sx < xb && sx < sw; sx++) {
          const p = (y * sw + sx) * 4;
          s += (Math.min(xb, sx + 1) - Math.max(xa, sx)) * (rgba[p] * kr + rgba[p + 1] * kg + rgba[p + 2] * kb) / 255;
        }
        rows[y * w + ox] = s;
      }
    }
  },

  /** Output rows y0 to y1 of the grey frame: grayH's rows shrunk along y by area, scaled to bytes. */
  grayV(a: { rows: F32; sw: number; sh: number; w: number; h: number; out: Uint8Array }, y0: number, y1: number) {
    const { rows, sw, sh, w, h, out } = a, f = sh / h, norm = 255 / ((sw / w) * f);
    for (let oy = y0; oy < y1; oy++) {
      const ya = oy * f, yb = ya + f;
      for (let ox = 0; ox < w; ox++) {
        let s = 0;
        for (let sy = Math.floor(ya); sy < yb && sy < sh; sy++) s += (Math.min(yb, sy + 1) - Math.max(ya, sy)) * rows[sy * w + ox];
        out[oy * w + ox] = clamp8(s * norm);
      }
    }
  },

  // ---- compositing (composite.ts) ----

  /** Blur along y of an RGB region (w x h at x0, y0 of an RGBA image `stride` wide) into tmp (3 floats a pixel);
   * rows r0 to r1 of the region. */
  blur8Y(a: { src: U8; stride: number; x0: number; y0: number; w: number; h: number; tmp: F32; k: F32 }, r0: number, r1: number) {
    const { src, stride, x0, y0, w, h, tmp, k } = a, n = k.length, half = (n - 1) / 2;
    for (let y = r0; y < r1; y++) {
      for (let x = 0; x < w; x++) {
        let r = 0, g = 0, b = 0;
        for (let t = 0; t < n; t++) {
          const s = ((y0 + reflect101(y - half + t, h)) * stride + x0 + x) * 4, kt = k[t];
          r += kt * src[s]; g += kt * src[s + 1]; b += kt * src[s + 2];
        }
        const o = (y * w + x) * 3;
        tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b;
      }
    }
  },

  /** Then along x, rounded, into dst (at dx0, dy0 of an RGBA image `dstride` wide). */
  blur8X(a: { tmp: F32; dst: U8; dstride: number; dx0: number; dy0: number; w: number; k: F32 }, r0: number, r1: number) {
    const { tmp, dst, dstride, dx0, dy0, w, k } = a, n = k.length, half = (n - 1) / 2;
    for (let y = r0; y < r1; y++) {
      for (let x = 0; x < w; x++) {
        let r = 0, g = 0, b = 0;
        for (let t = 0; t < n; t++) {
          const s = (y * w + reflect101(x - half + t, w)) * 3, kt = k[t];
          r += kt * tmp[s]; g += kt * tmp[s + 1]; b += kt * tmp[s + 2];
        }
        const o = ((dy0 + y) * dstride + dx0 + x) * 4;
        dst[o] = clamp8(r); dst[o + 1] = clamp8(g); dst[o + 2] = clamp8(b); dst[o + 3] = 255;
      }
    }
  },

  /** Rows r0 to r1 of the box (ew wide): the frame warped bilinearly with a replicated border into fex, the
   * valid mask by nearest neighbour into vex. `m` maps box pixels to screen pixels. */
  warp(a: { src: U8; sw: number; sh: number; valid: Uint8Array; m: number[]; ew: number; fex: U8; vex: Uint8Array }, r0: number, r1: number) {
    const { src, sw, sh, valid, ew, fex, vex } = a, [ma, mb, mc, md, me, mf] = a.m;
    for (let y = r0; y < r1; y++) {
      for (let x = 0; x < ew; x++) {
        const i = y * ew + x;
        const sx = ma * x + mb * y + mc, sy = md * x + me * y + mf;
        const nx = Math.floor(sx + 0.5), ny = Math.floor(sy + 0.5);
        vex[i] = nx >= 0 && nx < sw && ny >= 0 && ny < sh && valid[ny * sw + nx] ? 1 : 0;
        const fx0 = Math.floor(sx), fy0 = Math.floor(sy), fx = sx - fx0, fy = sy - fy0;
        const xa = Math.min(Math.max(fx0, 0), sw - 1), xb = Math.min(Math.max(fx0 + 1, 0), sw - 1);
        const ya = Math.min(Math.max(fy0, 0), sh - 1), yb = Math.min(Math.max(fy0 + 1, 0), sh - 1);
        const p00 = (ya * sw + xa) * 4, p01 = (ya * sw + xb) * 4, p10 = (yb * sw + xa) * 4, p11 = (yb * sw + xb) * 4;
        for (let ch = 0; ch < 3; ch++) {
          const top = (1 - fx) * src[p00 + ch] + fx * src[p01 + ch], bot = (1 - fx) * src[p10 + ch] + fx * src[p11 + ch];
          fex[i * 4 + ch] = clamp8((1 - fy) * top + fy * bot);
        }
        fex[i * 4 + 3] = 255;
      }
    }
  },

  /** ink_changed, along y: blurs of the visible mask, and of frame and canvas weighted by it (ta: 4 floats a
   * pixel, tb: 3). */
  ink1(a: { fex: U8; vex: Uint8Array; color: U8; cw: number; ex0: number; ey0: number; ew: number; eh: number; k: F32; ta: F32; tb: F32 }, r0: number, r1: number) {
    const { fex, vex, color, cw, ex0, ey0, ew, eh, k, ta, tb } = a, n = k.length, half = (n - 1) / 2;
    for (let y = r0; y < r1; y++) {
      for (let x = 0; x < ew; x++) {
        let a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0;
        for (let t = 0; t < n; t++) {
          const yy = reflect101(y - half + t, eh), j = yy * ew + x, kt = k[t];
          if (!vex[j]) continue;
          const c = ((ey0 + yy) * cw + ex0 + x) * 4;
          a0 += kt; a1 += kt * fex[j * 4]; a2 += kt * fex[j * 4 + 1]; a3 += kt * fex[j * 4 + 2];
          b0 += kt * color[c]; b1 += kt * color[c + 1]; b2 += kt * color[c + 2];
        }
        const i = y * ew + x;
        ta[i * 4] = a0; ta[i * 4 + 1] = a1; ta[i * 4 + 2] = a2; ta[i * 4 + 3] = a3;
        tb[i * 3] = b0; tb[i * 3 + 1] = b1; tb[i * 3 + 2] = b2;
      }
    }
  },

  /** Then along x, and the change test, into out. */
  ink2(a: { ta: F32; tb: F32; vex: Uint8Array; ew: number; k: F32; thr: number; out: Uint8Array }, r0: number, r1: number) {
    const { ta, tb, vex, ew, k, thr, out } = a, n = k.length, half = (n - 1) / 2;
    for (let y = r0; y < r1; y++) {
      for (let x = 0; x < ew; x++) {
        let a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0;
        for (let t = 0; t < n; t++) {
          const j = y * ew + reflect101(x - half + t, ew), kt = k[t];
          a0 += kt * ta[j * 4]; a1 += kt * ta[j * 4 + 1]; a2 += kt * ta[j * 4 + 2]; a3 += kt * ta[j * 4 + 3];
          b0 += kt * tb[j * 3]; b1 += kt * tb[j * 3 + 1]; b2 += kt * tb[j * 3 + 2];
        }
        const i = y * ew + x;
        const d = Math.max(Math.abs(a1 - b0), Math.abs(a2 - b1), Math.abs(a3 - b2)) / Math.max(a0, 1e-6);
        out[i] = d >= thr && vex[i] !== 0 && a0 > 0.25 ? 1 : 0;
      }
    }
  },

  /** A min or max filter of radius r along one axis; outside the box counts as `outside`. */
  morph(a: { src: Uint8Array; dst: Uint8Array; ew: number; eh: number; r: number; axis: number; opMax: boolean; outside: number }, r0: number, r1: number) {
    const { src, dst, ew, eh, r, axis, opMax, outside } = a;
    for (let y = r0; y < r1; y++) {
      for (let x = 0; x < ew; x++) {
        let v = opMax ? 0 : 1;
        for (let t = -r; t <= r; t++) {
          const xx = axis === 0 ? x + t : x, yy = axis === 0 ? y : y + t;
          const s = xx >= 0 && xx < ew && yy >= 0 && yy < eh ? src[yy * ew + xx] : outside;
          v = opMax ? Math.max(v, s) : Math.min(v, s);
        }
        dst[y * ew + x] = v;
      }
    }
  },

  /** The rules, on rows r0 to r1 of the updated box (iw wide at ix, iy of the warped box, at x0, y0 of the
   * canvas); returns [pixels committed, pixels visible]. */
  rules(a: {
    fb: U8; cb: U8; fex: U8; vex: Uint8Array; ink: Uint8Array; hasInk: boolean;
    color: U8; quality: F32; pending: Uint8Array; count: Uint16Array;
    ew: number; ix: number; iy: number; iw: number; cw: number; x0: number; y0: number;
    q: number; qg: number; faint: number; diff: number; gain: number; stable: number;
  }, r0: number, r1: number): [number, number] {
    const { fb, cb, fex, vex, ink, hasInk, color, quality, pending, count, ew, ix, iy, iw, cw, x0, y0, q, qg, faint: faintT, diff, gain, stable } = a;
    let committed = 0, visible = 0;
    const maxdiff = (p: ArrayLike<number>, i: number, r: ArrayLike<number>, j: number) =>
      Math.max(Math.abs(p[i] - r[j]), Math.abs(p[i + 1] - r[j + 1]), Math.abs(p[i + 2] - r[j + 2]));
    for (let y = r0; y < r1; y++) {
      for (let x = 0; x < iw; x++) {
        const ei = (y + iy) * ew + x + ix, ci = (y + y0) * cw + x + x0;
        const vis = vex[ei] !== 0;
        const qual = quality[ci];
        const d = maxdiff(fb, ei * 4, cb, ei * 4);
        const dp = maxdiff(fex, ei * 4, pending, ci * 3);
        const faint = d >= faintT && (qg >= qual || (hasInk && ink[ei] !== 0));
        let agree = qual > 0 && d < diff && !faint;
        const differ = vis && !agree;
        agree = agree && vis;
        let cnt = count[ci], take = false;
        if (agree) {
          take = q > qual * gain; // a sharper view of the same thing
          cnt = 0;
        } else if (differ) {
          if (cnt > 0 && dp < diff) cnt++;
          else {
            pending[ci * 3] = fex[ei * 4]; pending[ci * 3 + 1] = fex[ei * 4 + 1]; pending[ci * 3 + 2] = fex[ei * 4 + 2];
            cnt = 1;
          }
          if (cnt >= stable) { take = true; cnt = 0; committed++; }
        }
        if (take) {
          color[ci * 4] = fex[ei * 4]; color[ci * 4 + 1] = fex[ei * 4 + 1]; color[ci * 4 + 2] = fex[ei * 4 + 2];
          quality[ci] = q;
        }
        if (agree || differ) count[ci] = cnt;
        if (vis) visible++;
      }
    }
    return [committed, visible];
  },
};

export type Kernel = keyof typeof kernels;
