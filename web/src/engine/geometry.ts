/**
 * 2D similarity transforms with optional mirroring, as 3x3 homogeneous matrices stored row-major in
 * arrays of 9 numbers (artlapse's geometry.py).
 */
export type M3 = Float64Array;

export function eye(): M3 {
  return Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
}

export function mat(a: number, b: number, c: number, d: number, e: number, f: number): M3 {
  return Float64Array.of(a, b, c, d, e, f, 0, 0, 1);
}

export function scaleTranslate(s: number, tx = 0, ty = 0): M3 {
  return mat(s, 0, tx, 0, s, ty);
}

/** Horizontal mirror of an image `width` pixels wide (pixel centres stay on the grid). */
export function mirrorX(width: number): M3 {
  return mat(-1, 0, width - 1, 0, 1, 0);
}

export function mul(a: M3, b: M3): M3 {
  const o = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
  }
  return o;
}

export function inv(m: M3): M3 {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  const o = Float64Array.of(A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d);
  for (let k = 0; k < 9; k++) o[k] /= det;
  return o;
}

export function apply(m: M3, x: number, y: number): [number, number] {
  return [m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]];
}

export interface Similarity { scale: number; angle: number; flipped: boolean; tx: number; ty: number }

export function similarity(m: M3): Similarity {
  const det = m[0] * m[4] - m[1] * m[3];
  const flipped = det < 0;
  const [cx, cy] = flipped ? [-m[0], -m[3]] : [m[0], m[3]];
  return { scale: Math.sqrt(Math.abs(det)), angle: (Math.atan2(cy, cx) * 180) / Math.PI, flipped, tx: m[2], ty: m[5] };
}

export function isSimilarity(m: M3, tol = 1e-3): boolean {
  const [a, b, , c, d] = m;
  const s2 = Math.abs(a * d - b * c);
  const t = tol * Math.max(s2, 1e-9);
  // a @ a.T == s2 * I
  return Math.abs(a * a + b * b - s2) <= t && Math.abs(c * c + d * d - s2) <= t && Math.abs(a * c + b * d) <= t;
}

/** How much a screen->canvas transform moved: (relative scale change, max corner shift in canvas units)
 * over a nominal 1000x1000 screen patch. */
export function viewChange(prev: M3, cur: M3): [number, number] {
  let shift = 0;
  for (const [x, y] of [[0, 0], [1000, 0], [0, 1000], [1000, 1000]]) {
    const [ax, ay] = apply(prev, x, y), [bx, by] = apply(cur, x, y);
    shift = Math.max(shift, Math.hypot(ax - bx, ay - by));
  }
  return [Math.abs(Math.log(similarity(cur).scale / similarity(prev).scale)), shift];
}
