/**
 * cv2.estimateAffinePartial2D with RANSAC: a similarity (rotation, uniform scale, translation) from
 * point pairs. Two-point samples, OpenCV's adaptive iteration count, and a least-squares fit on the
 * inliers at the end (OpenCV refines with Levenberg-Marquardt, which reaches the same optimum: the
 * model is linear in its parameters). The inlier mask is RANSAC's, as OpenCV returns it.
 */
import { M3, mat } from './geometry';

/** A small seeded generator (mulberry32), so each sample's draws do not depend on what ran before. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function updateIters(p: number, ep: number, modelPoints: number, maxIters: number): number {
  const num = Math.log(Math.max(1 - p, Number.MIN_VALUE));
  const denom1 = 1 - Math.pow(1 - ep, modelPoints);
  if (denom1 < Number.MIN_VALUE) return 0;
  const denom = Math.log(denom1);
  return denom >= 0 || -num >= maxIters * -denom ? maxIters : Math.round(num / denom);
}

/** Least-squares similarity src -> dst over the points where `use` is set: [a, b, tx, ty] with
 * x' = a x - b y + tx, y' = b x + a y + ty. */
function lsq(src: Float64Array, dst: Float64Array, use: Uint8Array | null): number[] | null {
  let n = 0, sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < src.length / 2; i++) {
    if (use && !use[i]) continue;
    n++;
    sx += src[2 * i]; sy += src[2 * i + 1]; dx += dst[2 * i]; dy += dst[2 * i + 1];
  }
  if (n < 2) return null;
  sx /= n; sy /= n; dx /= n; dy /= n;
  let ss = 0, sa = 0, sb = 0;
  for (let i = 0; i < src.length / 2; i++) {
    if (use && !use[i]) continue;
    const x = src[2 * i] - sx, y = src[2 * i + 1] - sy, u = dst[2 * i] - dx, v = dst[2 * i + 1] - dy;
    ss += x * x + y * y;
    sa += x * u + y * v;
    sb += x * v - y * u;
  }
  if (ss <= 0) return null;
  const a = sa / ss, b = sb / ss;
  return [a, b, dx - (a * sx - b * sy), dy - (b * sx + a * sy)];
}

/** The similarity src -> dst ((N, 2) points, flat) and RANSAC's inlier mask, or null. */
export function estimateSimilarity(src: Float64Array, dst: Float64Array, threshold: number, random: () => number,
  maxIters = 4000, confidence = 0.995): { m: M3; inliers: Uint8Array } | null {
  const n = src.length / 2;
  if (n < 2) return null;
  const thr2 = threshold * threshold;
  let best: number[] | null = null;
  let bestCount = 0;
  let mask = new Uint8Array(n);
  const cur = new Uint8Array(n);
  let iters = maxIters;
  for (let it = 0; it < iters; it++) {
    let i = 0, j = 0, tries = 0;
    let ddx = 0, ddy = 0;
    do {
      i = Math.floor(random() * n);
      j = Math.floor(random() * n);
      ddx = src[2 * j] - src[2 * i];
      ddy = src[2 * j + 1] - src[2 * i + 1];
    } while ((i === j || ddx * ddx + ddy * ddy < 1e-12) && ++tries < 1000);
    if (tries >= 1000) break;
    // (a + ib) = (q2 - q1) / (p2 - p1) as complex numbers
    const qx = dst[2 * j] - dst[2 * i], qy = dst[2 * j + 1] - dst[2 * i + 1];
    const den = ddx * ddx + ddy * ddy;
    const a = (qx * ddx + qy * ddy) / den, b = (qy * ddx - qx * ddy) / den;
    const tx = dst[2 * i] - (a * src[2 * i] - b * src[2 * i + 1]);
    const ty = dst[2 * i + 1] - (b * src[2 * i] + a * src[2 * i + 1]);
    let count = 0;
    for (let k = 0; k < n; k++) {
      const x = src[2 * k], y = src[2 * k + 1];
      const ex = a * x - b * y + tx - dst[2 * k], ey = b * x + a * y + ty - dst[2 * k + 1];
      const inl = ex * ex + ey * ey <= thr2 ? 1 : 0;
      cur[k] = inl;
      count += inl;
    }
    if (count > Math.max(bestCount, 1)) {
      bestCount = count;
      best = [a, b, tx, ty];
      mask = cur.slice();
      iters = updateIters(confidence, (n - count) / n, 2, iters);
    }
  }
  if (!best) return null;
  const refined = lsq(src, dst, mask) ?? best;
  const [a, b, tx, ty] = refined;
  return { m: mat(a, -b, tx, b, a, ty), inliers: mask };
}
