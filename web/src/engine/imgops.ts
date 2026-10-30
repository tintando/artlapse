/** The few OpenCV image operations register's pixel votes use, on grey images in typed arrays. */
import { M3 } from './geometry';

/** cv2.warpAffine(src, m, WARP_INVERSE_MAP | INTER_LINEAR), constant border 0: out(x, y) = src(m (x, y)). */
export function warpLinear(src: Uint8Array, w: number, h: number, m: M3, out = new Uint8Array(w * h)): Uint8Array {
  const [a, b, c, d, e, f] = m;
  for (let y = 0; y < h; y++) {
    let sx = b * y + c, sy = e * y + f;
    for (let x = 0; x < w; x++, sx += a, sy += d) {
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      if (x0 < -1 || y0 < -1 || x0 >= w || y0 >= h) { out[y * w + x] = 0; continue; }
      const fx = sx - x0, fy = sy - y0;
      const in00 = x0 >= 0 && y0 >= 0, in10 = x0 + 1 < w && y0 >= 0, in01 = x0 >= 0 && y0 + 1 < h, in11 = x0 + 1 < w && y0 + 1 < h;
      const i = y0 * w + x0;
      const v00 = in00 ? src[i] : 0, v10 = in10 ? src[i + 1] : 0, v01 = in01 ? src[i + w] : 0, v11 = in11 ? src[i + w + 1] : 0;
      const v = (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy;
      out[y * w + x] = Math.round(v);
    }
  }
  return out;
}

/** cv2.warpAffine(mask, m, WARP_INVERSE_MAP | INTER_NEAREST) > 0, constant border 0. */
export function warpNearest(src: Uint8Array, w: number, h: number, m: M3, out = new Uint8Array(w * h)): Uint8Array {
  const [a, b, c, d, e, f] = m;
  for (let y = 0; y < h; y++) {
    let sx = b * y + c + 0.5, sy = e * y + f + 0.5;
    for (let x = 0; x < w; x++, sx += a, sy += d) {
      const X = Math.floor(sx), Y = Math.floor(sy);
      out[y * w + x] = X >= 0 && Y >= 0 && X < w && Y < h && src[Y * w + X] ? 1 : 0;
    }
  }
  return out;
}

function reflect101(i: number, n: number): number {
  if (n === 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
}

/** cv2.boxFilter(src, -1, (k, k)): the mean over k x k, reflected border (BORDER_REFLECT_101). */
export function boxFilter(src: Float32Array, w: number, h: number, k: number): Float32Array {
  const r = k >> 1;
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let s = 0;
    for (let t = -r; t <= r; t++) s += src[row + reflect101(t, w)];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = s;
      s += src[row + reflect101(x + r + 1, w)] - src[row + reflect101(x - r, w)];
    }
  }
  const inv = 1 / (k * k);
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let t = -r; t <= r; t++) s += tmp[reflect101(t, h) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = s * inv;
      s += tmp[reflect101(y + r + 1, h) * w + x] - tmp[reflect101(y - r, h) * w + x];
    }
  }
  return out;
}

/** cv2.erode(src > 0, ones((k, k)), borderValue=0) > 0: set where all of the k x k neighbourhood is set
 * and inside the image. */
export function erode(src: Uint8Array, w: number, h: number, k: number): Uint8Array {
  const r = k >> 1;
  // run lengths of set pixels: horizontally, then vertically over the horizontal result
  const hor = new Uint8Array(w * h), out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    let run = 0;
    const row = y * w;
    const right = new Int32Array(w); // set pixels from x rightwards
    for (let x = w - 1; x >= 0; x--) { run = src[row + x] ? run + 1 : 0; right[x] = run; }
    for (let x = 0; x < w; x++) hor[row + x] = x - r >= 0 && right[x - r] >= k ? 1 : 0;
  }
  for (let x = 0; x < w; x++) {
    let run = 0;
    const down = new Int32Array(h);
    for (let y = h - 1; y >= 0; y--) { run = hor[y * w + x] ? run + 1 : 0; down[y] = run; }
    for (let y = 0; y < h; y++) out[y * w + x] = y - r >= 0 && down[y - r] >= k ? 1 : 0;
  }
  return out;
}

/** Area of the convex hull of (N, 2) points (cv2.contourArea(cv2.convexHull(pts))). */
export function hullArea(pts: Float64Array | Float32Array, idx: number[]): number {
  if (idx.length < 3) return 0;
  const p = idx.map(i => [pts[2 * i], pts[2 * i + 1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: number[], a: number[], b: number[]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: number[][] = [], upper: number[][] = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
  let area = 0;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i], b = hull[(i + 1) % hull.length];
    area += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(area) / 2;
}

/** Python's round(): halves to even. */
export function pyRound(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}
