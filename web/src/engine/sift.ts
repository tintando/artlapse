/**
 * SIFT on the GPU, following OpenCV's implementation (sift.simd.cpp) step by step, so that it finds the
 * same keypoints and descriptors as cv2.SIFT_create(nfeatures) up to float rounding:
 *
 * - the image doubled (bilinear) and blurred to sigma 1.6, 3 layers per octave, 6 Gaussian images each;
 * - extrema of the difference of Gaussians over 26 neighbours, refined to sub-pixel and sub-scale, with
 *   OpenCV's contrast (0.04) and edge (10) thresholds;
 * - a 36-bin orientation histogram per extremum, a keypoint for every peak within 80% of the highest;
 * - the strongest `maxFeatures` kept (with ties), and only then the mask applied, as OpenCV does;
 * - 4x4x8 descriptors, clamped at 0.2, scaled by 512 and rounded to bytes.
 *
 * Histograms are summed with integer atomics in fixed point, so results do not depend on the order
 * the GPU runs threads in. The keypoint list is sorted on the CPU for the same reason. Two flip states
 * of one frame run in one pass: the mirrored one reads the frame right to left.
 */
import { Gpu } from './gpu';

export const LAYERS = 3; // nOctaveLayers
export const PER_OCTAVE = LAYERS + 3;
export const SIGMA = 1.6;
export const CONTRAST = 0.04;
export const EDGE = 10;
export const BORDER = 5; // SIFT_IMG_BORDER
const MAX_OCTAVES = 10;
export const CAP = 32768; // candidates and keypoints per image
const ORI_FIX = 1024; // fixed-point scale of the orientation histogram
const DESC_FIX = 4096; // ... and of the descriptor histogram

const COMMON = /* wgsl */ `
struct Level { off: u32, w: u32, h: u32, _p: u32 }
struct Pyr { lv: array<Level, ${MAX_OCTAVES * PER_OCTAVE}> }
fn reflect101(i: i32, n: i32) -> i32 {
  if (n == 1) { return 0; }
  var j = i;
  loop {
    if (j < 0) { j = -j; } else if (j >= n) { j = 2 * n - 2 - j; } else { break; }
  }
  return j;
}
fn deg(dy: f32, dx: f32) -> f32 {  // atan2 in degrees, in [0, 360) as cv::fastAtan2
  var a = degrees(atan2(dy, dx));
  if (a < 0.0) { a = a + 360.0; }
  if (a >= 360.0) { a = 0.0; }
  return a;
}
struct Cand { x: f32, y: f32, resp: f32, size: f32, olay: u32, r: i32, c: i32, xi: f32 }
struct Kp { x: f32, y: f32, resp: f32, size: f32, angle: f32, olay: u32, _a: u32, _b: u32 }
`;

const UPSAMPLE = /* wgsl */ `
struct U { w: u32, h: u32, flip: u32, _p: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> img: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
fn px(x: i32, y: i32) -> f32 {
  var xx = x;
  if (u.flip == 1u) { xx = i32(u.w) - 1 - x; }
  let i = u32(y) * u.w + u32(xx);
  return f32((img[i >> 2u] >> ((i & 3u) * 8u)) & 255u);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let W = 2u * u.w;
  let H = 2u * u.h;
  if (g.x >= W || g.y >= H) { return; }
  // cv::resize INTER_LINEAR: source coordinate (d + 0.5) / 2 - 0.5, clamped to the image
  var fx = (f32(g.x) + 0.5) * 0.5 - 0.5;
  var sx = i32(floor(fx));
  fx = fx - f32(sx);
  if (sx < 0) { fx = 0.0; sx = 0; }
  if (sx >= i32(u.w) - 1) { fx = 0.0; sx = i32(u.w) - 1; }
  var fy = (f32(g.y) + 0.5) * 0.5 - 0.5;
  var sy = i32(floor(fy));
  fy = fy - f32(sy);
  if (sy < 0) { fy = 0.0; sy = 0; }
  if (sy >= i32(u.h) - 1) { fy = 0.0; sy = i32(u.h) - 1; }
  let sx1 = min(sx + 1, i32(u.w) - 1);
  let sy1 = min(sy + 1, i32(u.h) - 1);
  let top = px(sx, sy) * (1.0 - fx) + px(sx1, sy) * fx;
  let bot = px(sx, sy1) * (1.0 - fx) + px(sx1, sy1) * fx;
  dst[g.y * W + g.x] = top * (1.0 - fy) + bot * fy;
}
`;

const BLUR = /* wgsl */ `
struct B { w: u32, h: u32, src: u32, dst: u32, half: u32, _a: u32, _b: u32, _c: u32, k: array<vec4f, 8> }
@group(0) @binding(0) var<uniform> b: B;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
fn kv(i: u32) -> f32 { return b.k[i >> 2u][i & 3u]; }
@compute @workgroup_size(16, 16)
fn h(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= b.w || g.y >= b.h) { return; }
  let row = b.src + g.y * b.w;
  var s = 0.0;
  let half = i32(b.half);
  for (var t = -half; t <= half; t++) {
    s += kv(u32(t + half)) * src[row + u32(reflect101(i32(g.x) + t, i32(b.w)))];
  }
  dst[b.dst + g.y * b.w + g.x] = s;
}
@compute @workgroup_size(16, 16)
fn v(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= b.w || g.y >= b.h) { return; }
  var s = 0.0;
  let half = i32(b.half);
  for (var t = -half; t <= half; t++) {
    s += kv(u32(t + half)) * src[b.src + u32(reflect101(i32(g.y) + t, i32(b.h))) * b.w + g.x];
  }
  dst[b.dst + g.y * b.w + g.x] = s;
}
`;

const DOWNSAMPLE = /* wgsl */ `
struct D { src: u32, sw: u32, sh: u32, dst: u32, dw: u32, dh: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> d: D;
@group(0) @binding(1) var<storage, read_write> pyr: array<f32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= d.dw || g.y >= d.dh) { return; }
  // cv::resize INTER_NEAREST: source pixel floor(d * src / dst)
  let sx = (g.x * d.sw) / d.dw;
  let sy = (g.y * d.sh) / d.dh;
  pyr[d.dst + g.y * d.dw + g.x] = pyr[d.src + sy * d.sw + sx];
}
`;

const EXTREMA = COMMON + /* wgsl */ `
struct E { o: u32, layer: u32, threshold: f32, sigma: f32 }
@group(0) @binding(0) var<uniform> e: E;
@group(0) @binding(1) var<uniform> pyr: Pyr;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read_write> cand: array<Cand>;
@group(0) @binding(4) var<storage, read_write> count: array<atomic<u32>>;
fn G(l: u32, r: i32, c: i32) -> f32 { let L = pyr.lv[e.o * ${PER_OCTAVE}u + l]; return g[L.off + u32(r) * L.w + u32(c)]; }
fn D(l: u32, r: i32, c: i32) -> f32 { return G(l + 1u, r, c) - G(l, r, c); }
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let L = pyr.lv[e.o * ${PER_OCTAVE}u];
  let w = i32(L.w);
  let h = i32(L.h);
  var c = i32(gid.x) + ${BORDER};
  var r = i32(gid.y) + ${BORDER};
  if (c >= w - ${BORDER} || r >= h - ${BORDER}) { return; }
  let val = D(e.layer, r, c);
  if (abs(val) <= e.threshold) { return; }
  for (var dl = -1; dl <= 1; dl++) {
    for (var dr = -1; dr <= 1; dr++) {
      for (var dc = -1; dc <= 1; dc++) {
        if (dl == 0 && dr == 0 && dc == 0) { continue; }
        let n = D(u32(i32(e.layer) + dl), r + dr, c + dc);
        if ((val > 0.0 && val < n) || (val < 0.0 && val > n)) { return; }
      }
    }
  }
  // adjustLocalExtrema
  let img_scale = 1.0 / 255.0;
  let ds = img_scale * 0.5;
  let s2 = img_scale;
  let cs = img_scale * 0.25;
  var layer = i32(e.layer);
  var xi = 0.0;
  var xr = 0.0;
  var xc = 0.0;
  var i = 0;
  loop {
    if (i >= 5) { return; }
    let l = u32(layer);
    let v2 = D(l, r, c) * 2.0;
    let dD = vec3f((D(l, r, c + 1) - D(l, r, c - 1)) * ds, (D(l, r + 1, c) - D(l, r - 1, c)) * ds,
                   (D(l + 1u, r, c) - D(l - 1u, r, c)) * ds);
    let dxx = (D(l, r, c + 1) + D(l, r, c - 1) - v2) * s2;
    let dyy = (D(l, r + 1, c) + D(l, r - 1, c) - v2) * s2;
    let dss = (D(l + 1u, r, c) + D(l - 1u, r, c) - v2) * s2;
    let dxy = (D(l, r + 1, c + 1) - D(l, r + 1, c - 1) - D(l, r - 1, c + 1) + D(l, r - 1, c - 1)) * cs;
    let dxs = (D(l + 1u, r, c + 1) - D(l + 1u, r, c - 1) - D(l - 1u, r, c + 1) + D(l - 1u, r, c - 1)) * cs;
    let dys = (D(l + 1u, r + 1, c) - D(l + 1u, r - 1, c) - D(l - 1u, r + 1, c) + D(l - 1u, r - 1, c)) * cs;
    let H = mat3x3f(vec3f(dxx, dxy, dxs), vec3f(dxy, dyy, dys), vec3f(dxs, dys, dss));
    let det = determinant(H);
    var X = vec3f(0.0);
    if (det != 0.0) {
      // Cramer's rule
      X = vec3f(determinant(mat3x3f(dD, H[1], H[2])), determinant(mat3x3f(H[0], dD, H[2])),
                determinant(mat3x3f(H[0], H[1], dD))) / det;
    }
    xi = -X.z;
    xr = -X.y;
    xc = -X.x;
    if (abs(xi) < 0.5 && abs(xr) < 0.5 && abs(xc) < 0.5) { break; }
    if (abs(xi) > 7.0e8 || abs(xr) > 7.0e8 || abs(xc) > 7.0e8) { return; }
    c += i32(round(xc));
    r += i32(round(xr));
    layer += i32(round(xi));
    if (layer < 1 || layer > ${LAYERS} || c < ${BORDER} || c >= w - ${BORDER} || r < ${BORDER} || r >= h - ${BORDER}) { return; }
    i++;
  }
  let l = u32(layer);
  let dD = vec3f((D(l, r, c + 1) - D(l, r, c - 1)) * ds, (D(l, r + 1, c) - D(l, r - 1, c)) * ds,
                 (D(l + 1u, r, c) - D(l - 1u, r, c)) * ds);
  let t = dot(dD, vec3f(xc, xr, xi));
  let contr = D(l, r, c) * img_scale + t * 0.5;
  if (abs(contr) * ${LAYERS}.0 < ${CONTRAST}) { return; }
  let v2 = D(l, r, c) * 2.0;
  let dxx = (D(l, r, c + 1) + D(l, r, c - 1) - v2) * s2;
  let dyy = (D(l, r + 1, c) + D(l, r - 1, c) - v2) * s2;
  let dxy = (D(l, r + 1, c + 1) - D(l, r + 1, c - 1) - D(l, r - 1, c + 1) + D(l, r - 1, c - 1)) * cs;
  let tr = dxx + dyy;
  let det = dxx * dyy - dxy * dxy;
  if (det <= 0.0 || tr * tr * ${EDGE}.0 >= ${(EDGE + 1) * (EDGE + 1)}.0 * det) { return; }
  let k = atomicAdd(&count[0], 1u);
  if (k >= ${CAP}u) { return; }
  let po = f32(1u << e.o);
  cand[k] = Cand(f32(c) + xc, f32(r) + xr, abs(contr), e.sigma * pow(2.0, (f32(layer) + xi) / ${LAYERS}.0) * po * 2.0,
                 e.o | (u32(layer) << 8u), r, c, xi);
}
`;

const ARGS = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> count: array<u32>;
@group(0) @binding(1) var<storage, read_write> args: array<u32>;
@compute @workgroup_size(1)
fn main() { args[0] = min(count[0], ${CAP}u); args[1] = 1u; args[2] = 1u; }
`;

const ORIENTATION = COMMON + /* wgsl */ `
@group(0) @binding(0) var<uniform> pyr: Pyr;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> cand: array<Cand>;
@group(0) @binding(3) var<storage, read_write> kps: array<Kp>;
@group(0) @binding(4) var<storage, read_write> count: array<atomic<u32>>;
var<workgroup> hist: array<atomic<u32>, 36>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li < 36u) { atomicStore(&hist[li], 0u); }
  workgroupBarrier();
  let k = cand[wg.x];
  let o = k.olay & 255u;
  let L = pyr.lv[o * ${PER_OCTAVE}u + (k.olay >> 8u)];
  let w = i32(L.w);
  let h = i32(L.h);
  let scl = k.size * 0.5 / f32(1u << o);
  let radius = i32(round(4.5 * scl));
  let sig = 1.5 * scl;
  let es = -1.0 / (2.0 * sig * sig);
  let side = 2 * radius + 1;
  for (var s = i32(li); s < side * side; s += 64) {
    let i = s / side - radius;
    let j = s % side - radius;
    let y = k.r + i;
    let x = k.c + j;
    if (y <= 0 || y >= h - 1 || x <= 0 || x >= w - 1) { continue; }
    let dx = g[L.off + u32(y * w + x + 1)] - g[L.off + u32(y * w + x - 1)];
    let dy = g[L.off + u32((y - 1) * w + x)] - g[L.off + u32((y + 1) * w + x)];
    let wt = exp(f32(i * i + j * j) * es);
    var bin = i32(round(36.0 / 360.0 * deg(dy, dx)));
    if (bin >= 36) { bin -= 36; }
    if (bin < 0) { bin += 36; }
    atomicAdd(&hist[bin], u32(round(wt * sqrt(dx * dx + dy * dy) * ${ORI_FIX}.0)));
  }
  workgroupBarrier();
  if (li != 0u) { return; }
  var t: array<f32, 40>;
  for (var i = 0; i < 36; i++) { t[i + 2] = f32(atomicLoad(&hist[i])) / ${ORI_FIX}.0; }
  t[0] = t[36]; t[1] = t[37]; t[38] = t[2]; t[39] = t[3];
  var hs: array<f32, 36>;
  var omax = 0.0;
  for (var i = 0; i < 36; i++) {
    hs[i] = (t[i] + t[i + 4]) * (1.0 / 16.0) + (t[i + 1] + t[i + 3]) * (4.0 / 16.0) + t[i + 2] * (6.0 / 16.0);
    omax = max(omax, hs[i]);
  }
  let thr = omax * 0.8;
  for (var j = 0; j < 36; j++) {
    let l = select(35, j - 1, j > 0);
    let r2 = select(0, j + 1, j < 35);
    if (hs[j] > hs[l] && hs[j] > hs[r2] && hs[j] >= thr) {
      var bin = f32(j) + 0.5 * (hs[l] - hs[r2]) / (hs[l] - 2.0 * hs[j] + hs[r2]);
      if (bin < 0.0) { bin += 36.0; } else if (bin >= 36.0) { bin -= 36.0; }
      var angle = 360.0 - 10.0 * bin;
      if (abs(angle - 360.0) < 1.1920929e-7) { angle = 0.0; }
      let n = atomicAdd(&count[1], 1u);
      if (n < ${CAP}u) { kps[n] = Kp(k.x, k.y, k.resp, k.size, angle, k.olay, 0u, 0u); }
    }
  }
}
`;

const DESCRIPTOR = COMMON + /* wgsl */ `
struct In { x: f32, y: f32, ori: f32, scl: f32, olay: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> pyr: Pyr;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> kin: array<In>;
@group(0) @binding(3) var<storage, read_write> desc: array<u32>;
var<workgroup> hist: array<atomic<u32>, 360>;
var<workgroup> raw: array<f32, 128>;
var<workgroup> scale: f32;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let ki = wg.x;
  for (var i = li; i < 360u; i += 64u) { atomicStore(&hist[i], 0u); }
  workgroupBarrier();
  let k = kin[ki];
  let L = pyr.lv[(k.olay & 255u) * ${PER_OCTAVE}u + (k.olay >> 8u)];
  let w = i32(L.w);
  let h = i32(L.h);
  let px = i32(round(k.x));
  let py = i32(round(k.y));
  let hist_width = 3.0 * k.scl;
  var radius = i32(round(hist_width * 1.4142135623730951 * 5.0 * 0.5));
  radius = min(radius, i32(sqrt(f32(w) * f32(w) + f32(h) * f32(h))));
  let cos_t = cos(k.ori * 0.017453292519943295) / hist_width;
  let sin_t = sin(k.ori * 0.017453292519943295) / hist_width;
  let side = 2 * radius + 1;
  for (var s = i32(li); s < side * side; s += 64) {
    let i = s / side - radius;
    let j = s % side - radius;
    let c_rot = f32(j) * cos_t - f32(i) * sin_t;
    let r_rot = f32(j) * sin_t + f32(i) * cos_t;
    var rbin = r_rot + 2.0 - 0.5;
    var cbin = c_rot + 2.0 - 0.5;
    let r = py + i;
    let c = px + j;
    if (!(rbin > -1.0 && rbin < 4.0 && cbin > -1.0 && cbin < 4.0 && r > 0 && r < h - 1 && c > 0 && c < w - 1)) { continue; }
    let dx = g[L.off + u32(r * w + c + 1)] - g[L.off + u32(r * w + c - 1)];
    let dy = g[L.off + u32((r - 1) * w + c)] - g[L.off + u32((r + 1) * w + c)];
    var obin = (deg(dy, dx) - k.ori) * (8.0 / 360.0);
    let mag = sqrt(dx * dx + dy * dy) * exp((c_rot * c_rot + r_rot * r_rot) * (-1.0 / 8.0));
    let r0 = i32(floor(rbin));
    let c0 = i32(floor(cbin));
    var o0 = i32(floor(obin));
    rbin -= f32(r0);
    cbin -= f32(c0);
    obin -= f32(o0);
    if (o0 < 0) { o0 += 8; }
    if (o0 >= 8) { o0 -= 8; }
    let v_r1 = mag * rbin;
    let v_r0 = mag - v_r1;
    let v_rc11 = v_r1 * cbin;
    let v_rc10 = v_r1 - v_rc11;
    let v_rc01 = v_r0 * cbin;
    let v_rc00 = v_r0 - v_rc01;
    let v_rco111 = v_rc11 * obin;
    let v_rco110 = v_rc11 - v_rco111;
    let v_rco101 = v_rc10 * obin;
    let v_rco100 = v_rc10 - v_rco101;
    let v_rco011 = v_rc01 * obin;
    let v_rco010 = v_rc01 - v_rco011;
    let v_rco001 = v_rc00 * obin;
    let v_rco000 = v_rc00 - v_rco001;
    let idx = u32(((r0 + 1) * 6 + c0 + 1) * 10 + o0);
    atomicAdd(&hist[idx], u32(round(v_rco000 * ${DESC_FIX}.0)));
    atomicAdd(&hist[idx + 1u], u32(round(v_rco001 * ${DESC_FIX}.0)));
    atomicAdd(&hist[idx + 10u], u32(round(v_rco010 * ${DESC_FIX}.0)));
    atomicAdd(&hist[idx + 11u], u32(round(v_rco011 * ${DESC_FIX}.0)));
    atomicAdd(&hist[idx + 60u], u32(round(v_rco100 * ${DESC_FIX}.0)));
    atomicAdd(&hist[idx + 61u], u32(round(v_rco101 * ${DESC_FIX}.0)));
    atomicAdd(&hist[idx + 70u], u32(round(v_rco110 * ${DESC_FIX}.0)));
    atomicAdd(&hist[idx + 71u], u32(round(v_rco111 * ${DESC_FIX}.0)));
  }
  workgroupBarrier();
  for (var t = li; t < 128u; t += 64u) {
    let cell = t / 8u;
    let o = t % 8u;
    let idx = (((cell / 4u) + 1u) * 6u + (cell % 4u) + 1u) * 10u;
    var v = f32(atomicLoad(&hist[idx + o]));
    if (o < 2u) { v += f32(atomicLoad(&hist[idx + 8u + o])); }
    raw[t] = v / ${DESC_FIX}.0;
  }
  workgroupBarrier();
  if (li == 0u) {
    var nrm2 = 0.0;
    for (var t = 0; t < 128; t++) { nrm2 += raw[t] * raw[t]; }
    let thr = sqrt(nrm2) * 0.2;
    nrm2 = 0.0;
    for (var t = 0; t < 128; t++) { let v = min(raw[t], thr); raw[t] = v; nrm2 += v * v; }
    scale = 512.0 / max(sqrt(nrm2), 1.1920929e-7);
  }
  workgroupBarrier();
  if (li < 32u) {
    var word = 0u;
    for (var b = 0u; b < 4u; b++) {
      word |= u32(clamp(round(raw[li * 4u + b] * scale), 0.0, 255.0)) << (8u * b);
    }
    desc[ki * 32u + li] = word;
  }
}
`;

/** One octave's size and its Gaussian images' offsets in the pyramid buffer (in floats). */
export interface Octave { w: number; h: number; off: number[] }

export interface Features {
  pts: Float32Array; // (N, 2) keypoint positions in analysis pixels (mirrored image for a flipped state)
  desc: Uint8Array; // (N, 128)
}

/** What register needs of SIFT: this GPU one, or SiftCpu's pool (sift-cpu.ts). */
export interface Detector {
  /** Keypoints and descriptors of a grey analysis frame (w x h bytes) in each flip state; `masks[i]` (255
   * where features may be kept) is in that state's coordinates, mirrored for a flipped state. */
  detect(gray: Uint8Array, flips: boolean[], masks: Uint8Array[]): Promise<Features[]>;
}

/** The octaves of the doubled w x h image, each with its Gaussian images' offsets in one buffer, and
 * that buffer's length (in floats). */
export function octaves(w: number, h: number): [Octave[], number] {
  const W = 2 * w, H = 2 * h;
  const n = Math.min(MAX_OCTAVES, Math.round(Math.log2(Math.min(W, H)) - 2) + 1);
  const out: Octave[] = [];
  let off = 0, ow = W, oh = H;
  for (let o = 0; o < n && ow > 2 * BORDER && oh > 2 * BORDER; o++) {
    const offs = [];
    for (let l = 0; l < PER_OCTAVE; l++) { offs.push(off); off += ow * oh; }
    out.push({ w: ow, h: oh, off: offs });
    ow = Math.floor(ow / 2);
    oh = Math.floor(oh / 2);
  }
  return [out, off];
}

/** The blur from each level of an octave to the next (buildGaussianPyramid); [0] is the base's sigma. */
export function levelSigmas(): number[] {
  const sig = [SIGMA];
  const k = 2 ** (1 / LAYERS);
  for (let i = 1; i < PER_OCTAVE; i++) {
    const prev = k ** (i - 1) * SIGMA, total = prev * k;
    sig.push(Math.sqrt(total * total - prev * prev));
  }
  return sig;
}

/** The blur that takes the doubled image to the pyramid's base. */
export const BASE_SIGMA = Math.sqrt(Math.max(SIGMA * SIGMA - 0.25 * 4, 0.01));

/** The DoG threshold extrema are first tested against. */
export const THRESHOLD = Math.floor(0.5 * CONTRAST / LAYERS * 255);

/**
 * removeDuplicatedSorted, retainBest(maxFeatures), the firstOctave = -1 rescaling and the mask, on `n`
 * oriented keypoints (Kp records of 8 words: x, y, response, size, angle, octave | layer << 8); returns
 * their positions and the descriptor stage's input (8 words each: x, y, angle, scale, octave | layer << 8).
 */
export function selectKeypoints(n: number, list: ArrayBuffer, mask: Uint8Array, w: number, h: number, maxFeatures: number) {
  const f = new Float32Array(list), u = new Uint32Array(list);
  let idx = Array.from({ length: n }, (_, i) => i);
  // x, y, size, angle, response
  const key = (i: number, j: number) => f[i * 8 + j];
  idx.sort((a, b) => key(a, 0) - key(b, 0) || key(a, 1) - key(b, 1) || key(a, 3) - key(b, 3) || key(a, 4) - key(b, 4) || key(b, 2) - key(a, 2));
  idx = idx.filter((a, i) => {
    if (!i) return true;
    const b = idx[i - 1];
    return !(key(a, 0) === key(b, 0) && key(a, 1) === key(b, 1) && key(a, 3) === key(b, 3) && key(a, 4) === key(b, 4));
  });
  if (maxFeatures > 0 && idx.length > maxFeatures) {
    const resp = idx.map(i => key(i, 2)).sort((a, b) => b - a);
    const cut = resp[maxFeatures - 1];
    idx = idx.filter(i => key(i, 2) >= cut);
  }
  const pts: number[] = [];
  const input: number[] = [];
  for (const i of idx) {
    const olay = u[i * 8 + 5], o = olay & 255;
    // kpt.pt = (c + xc) * 2^o, then halved for the doubled image
    const x = Math.fround(Math.fround(key(i, 0) * (1 << o)) * 0.5), y = Math.fround(Math.fround(key(i, 1) * (1 << o)) * 0.5);
    const mx = Math.floor(x + 0.5), my = Math.floor(y + 0.5);
    if (mx < 0 || my < 0 || mx >= w || my >= h || !mask[my * w + mx]) continue;
    const size = Math.fround(key(i, 3) * 0.5);
    const scale = o >= 1 ? 1 / (1 << (o - 1)) : 2; // unpackOctave of octave o - 1
    let angle = 360 - key(i, 4);
    if (Math.abs(angle - 360) < 1.1920929e-7) angle = 0;
    pts.push(x, y);
    input.push(x * scale, y * scale, angle, size * scale * 0.5, olay, 0, 0, 0);
  }
  const buf = new ArrayBuffer(input.length * 4);
  const fb = new Float32Array(buf), ub = new Uint32Array(buf);
  for (let i = 0; i < input.length; i += 8) {
    fb.set(input.slice(i, i + 4), i);
    ub[i + 4] = input[i + 4];
  }
  return { n: pts.length / 2, pts: new Float32Array(pts), input: buf };
}

/** cv::getGaussianKernel for a float image (ksize from sigma as cv::GaussianBlur picks it). */
export function gaussKernel(sigma: number, depth8u = false): Float64Array {
  const n = Math.round(sigma * (depth8u ? 3 : 4) * 2 + 1) | 1;
  const k = new Float64Array(n);
  const s2 = -0.5 / (sigma * sigma);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const x = i - (n - 1) * 0.5;
    k[i] = Math.exp(s2 * x * x);
    sum += k[i];
  }
  for (let i = 0; i < n; i++) k[i] /= sum;
  return k;
}

interface State {
  pyr: GPUBuffer;
  cand: GPUBuffer;
  kps: GPUBuffer;
  counts: GPUBuffer; // [candidates, keypoints]
  args: GPUBuffer;
  upU: GPUBuffer;
  descIn: GPUBuffer;
  desc: GPUBuffer;
  descCap: number;
}

export class Sift implements Detector {
  readonly octaves: Octave[];
  private img: GPUBuffer;
  private tmp: GPUBuffer;
  private tmp2: GPUBuffer;
  private pyrU: GPUBuffer;
  private states: State[] = [];
  private blurU = new Map<string, GPUBuffer>();
  private extremaU: GPUBuffer[][] = [];
  private downU: GPUBuffer[] = [];
  private zeros = new Uint32Array(4);

  constructor(private gpu: Gpu, readonly w: number, readonly h: number, readonly maxFeatures: number) {
    const W = 2 * w, H = 2 * h;
    const [octs, off] = octaves(w, h);
    this.octaves = octs;
    const lv = new Uint32Array(MAX_OCTAVES * PER_OCTAVE * 4);
    this.octaves.forEach((oc, o) => oc.off.forEach((of, l) => lv.set([of, oc.w, oc.h, 0], (o * PER_OCTAVE + l) * 4)));
    this.pyrU = gpu.uniform(lv);
    this.img = gpu.buffer(w * h);
    this.tmp = gpu.buffer(W * H * 4);
    this.tmp2 = gpu.buffer(W * H * 4);
    for (const flip of [0, 1]) {
      this.states.push({
        pyr: gpu.buffer(off * 4),
        cand: gpu.buffer(CAP * 32),
        kps: gpu.buffer(CAP * 32),
        counts: gpu.buffer(16),
        args: gpu.buffer(16, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT),
        upU: gpu.uniform(new Uint32Array([w, h, flip, 0])),
        descIn: gpu.buffer(0),
        desc: gpu.buffer(0),
        descCap: 0,
      });
    }
    this.sig = levelSigmas();
    this.octaves.forEach((oc, o) => {
      this.extremaU.push([0, 1, 2, 3].map(l => {
        const b = new ArrayBuffer(16);
        new Uint32Array(b, 0, 2).set([o, l]);
        new Float32Array(b, 8, 2).set([THRESHOLD, SIGMA]);
        return gpu.uniform(b);
      }));
      if (o > 0) {
        const p = this.octaves[o - 1];
        this.downU.push(gpu.uniform(new Uint32Array([p.off[LAYERS], p.w, p.h, oc.off[0], oc.w, oc.h, 0, 0])));
      }
    });
  }

  private sig: number[];

  private blurUniform(w: number, h: number, src: number, dst: number, sigma: number): GPUBuffer {
    const key = `${w},${h},${src},${dst},${sigma}`;
    let u = this.blurU.get(key);
    if (!u) {
      const k = gaussKernel(sigma);
      const b = new ArrayBuffer(32 + 128);
      new Uint32Array(b, 0, 5).set([w, h, src, dst, (k.length - 1) / 2]);
      new Float32Array(b, 32, 32).set(k);
      u = this.gpu.uniform(b);
      this.blurU.set(key, u);
    }
    return u;
  }

  private blur(enc: GPUCommandEncoder, src: GPUBuffer, srcOff: number, dst: GPUBuffer, dstOff: number, w: number, h: number, sigma: number) {
    const g = this.gpu;
    const gx = Math.ceil(w / 16), gy = Math.ceil(h / 16);
    g.dispatch(enc, g.pipeline(COMMON + BLUR, 'h'), [this.blurUniform(w, h, srcOff, 0, sigma), src, this.tmp], gx, gy);
    g.dispatch(enc, g.pipeline(COMMON + BLUR, 'v'), [this.blurUniform(w, h, 0, dstOff, sigma), this.tmp, dst], gx, gy);
  }

  /** Record the pyramid, extrema and orientations of one flip state. */
  private encodeKeypoints(enc: GPUCommandEncoder, st: State) {
    const g = this.gpu;
    const W = 2 * this.w, H = 2 * this.h;
    g.dispatch(enc, g.pipeline(UPSAMPLE, 'main'), [st.upU, this.img, this.tmp2], Math.ceil(W / 16), Math.ceil(H / 16));
    this.blur(enc, this.tmp2, 0, st.pyr, this.octaves[0].off[0], W, H, BASE_SIGMA);
    this.octaves.forEach((oc, o) => {
      if (o > 0) g.dispatch(enc, g.pipeline(DOWNSAMPLE, 'main'), [this.downU[o - 1], st.pyr], Math.ceil(oc.w / 16), Math.ceil(oc.h / 16));
      for (let l = 1; l < PER_OCTAVE; l++) this.blur(enc, st.pyr, oc.off[l - 1], st.pyr, oc.off[l], oc.w, oc.h, this.sig[l]);
    });
    const ex = g.pipeline(EXTREMA, 'main');
    this.octaves.forEach((oc, o) => {
      for (let l = 1; l <= LAYERS; l++) {
        g.dispatch(enc, ex, [this.extremaU[o][l], this.pyrU, st.pyr, st.cand, st.counts],
          Math.ceil((oc.w - 2 * BORDER) / 16), Math.ceil((oc.h - 2 * BORDER) / 16));
      }
    });
    g.dispatch(enc, g.pipeline(ARGS, 'main'), [st.counts, st.args], 1);
    g.dispatchIndirect(enc, g.pipeline(ORIENTATION, 'main'), [this.pyrU, st.pyr, st.cand, st.kps, st.counts], st.args);
  }

  async detect(gray: Uint8Array, flips: boolean[], masks: Uint8Array[]): Promise<Features[]> {
    const g = this.gpu, q = g.device.queue;
    q.writeBuffer(this.img, 0, gray.buffer, gray.byteOffset, Math.ceil(gray.byteLength / 4) * 4);
    const enc = g.device.createCommandEncoder();
    const used = flips.map(f => this.states[f ? 1 : 0]);
    for (const st of used) {
      q.writeBuffer(st.counts, 0, this.zeros);
      this.encodeKeypoints(enc, st);
    }
    // the keypoint counter is the second word of counts; read both, and the lists
    const parts: [GPUBuffer, number, number][] = [];
    for (const st of used) parts.push([st.counts, 0, 16], [st.kps, 0, CAP * 32]);
    const raw = await g.read(enc, parts);
    const chosen = used.map((_, i) => selectKeypoints(Math.min(new Uint32Array(raw[2 * i])[1], CAP), raw[2 * i + 1], masks[i], this.w, this.h, this.maxFeatures));
    // descriptors
    const enc2 = g.device.createCommandEncoder();
    used.forEach((st, i) => {
      const n = chosen[i].n;
      if (!n) return;
      if (st.descCap < n) {
        st.descIn.destroy();
        st.desc.destroy();
        st.descCap = Math.max(n, 2048);
        st.descIn = g.buffer(st.descCap * 32);
        st.desc = g.buffer(st.descCap * 128);
      }
      q.writeBuffer(st.descIn, 0, chosen[i].input);
      g.dispatch(enc2, g.pipeline(DESCRIPTOR, 'main'), [this.pyrU, st.pyr, st.descIn, st.desc], n);
    });
    const dparts: [GPUBuffer, number, number][] = used.map((st, i) => [st.desc, 0, chosen[i].n * 128]);
    const draw = await g.read(enc2, dparts);
    return chosen.map((c, i) => ({ pts: c.pts, desc: new Uint8Array(draw[i]) }));
  }
}
