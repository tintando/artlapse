/**
 * Compositing on the GPU: artlapse's accumulate_gpu.GpuCanvas (the rules of accumulate.Canvas) in WGSL.
 * Each placed frame is imported from its VideoFrame, warped into the canvas box it covers, blurred and
 * compared there; only two counts come back. The warp, blurs and rules follow the CUDA kernels: bilinear
 * warp with a replicated border, nearest-neighbour mask warp, Gaussian kernels of OpenCV's size for 8-bit
 * images, reflected borders, results rounded to 8 bits.
 */
import { Config } from './config';
import * as geo from './geometry';
import { M3 } from './geometry';
import { Gpu, groups } from './gpu';
import { erode, pyRound } from './imgops';
import { gaussKernel } from './sift';

export const MARGIN = 8; // canvas px of context around the updated box
export const INK_SIGMA = 2.0;
export const INK_GROW = 1.0;
const WG = 256;

const PACK = /* wgsl */ `
fn unpack(v: u32) -> vec3f { return vec3f(f32(v & 255u), f32((v >> 8u) & 255u), f32((v >> 16u) & 255u)); }
fn pack(c: vec3f) -> u32 {
  let q = vec3u(clamp(round(c), vec3f(0.0), vec3f(255.0)));
  return q.x | (q.y << 8u) | (q.z << 16u) | (255u << 24u);
}
fn reflect101(i: i32, n: i32) -> i32 {
  if (n == 1) { return 0; }
  var j = i;
  loop {
    if (j < 0) { j = -j; } else if (j >= n) { j = 2 * n - 2 - j; } else { break; }
  }
  return j;
}
fn lin(g: vec3u) -> u32 { return g.x + g.y * ${WG * 65535}u; }
`;

const UPLOAD = PACK + /* wgsl */ `
@group(0) @binding(0) var<uniform> p: vec4u;
@group(0) @binding(1) var tex: texture_external;
@group(0) @binding(2) var<storage, read_write> frame: array<u32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = lin(g);
  if (i >= p.x * p.y) { return; }
  frame[i] = pack(textureLoad(tex, vec2u(i % p.x, i / p.x)).rgb * 255.0);
}
`;

// separable blur of an 8-bit RGB region: pass 1 (along y) from packed pixels to floats, pass 2 (along x)
// back to rounded packed pixels
const BLUR = PACK + /* wgsl */ `
struct B { w: u32, h: u32, stride: u32, x0: u32, y0: u32, half: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> b: B;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<u32>;
@group(0) @binding(3) var<storage, read_write> tmp: array<vec4f>;
@compute @workgroup_size(${WG})
fn along_y(@builtin(global_invocation_id) g: vec3u) {
  let i = lin(g);
  if (i >= b.w * b.h) { return; }
  let x = i % b.w;
  let y = i32(i / b.w);
  var s = vec3f(0.0);
  let half = i32(b.half);
  for (var t = -half; t <= half; t++) {
    let yy = u32(reflect101(y + t, i32(b.h)));
    s += k[u32(t + half)] * unpack(src[(b.y0 + yy) * b.stride + b.x0 + x]);
  }
  tmp[i] = vec4f(s, 0.0);
}
`;

const BLUR2 = PACK + /* wgsl */ `
struct B { w: u32, h: u32, stride: u32, x0: u32, y0: u32, half: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> b: B;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> tmp: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(${WG})
fn along_x(@builtin(global_invocation_id) g: vec3u) {
  let i = lin(g);
  if (i >= b.w * b.h) { return; }
  let x = i32(i % b.w);
  let y = i / b.w;
  var s = vec3f(0.0);
  let half = i32(b.half);
  for (var t = -half; t <= half; t++) {
    s += k[u32(t + half)] * tmp[y * b.w + u32(reflect101(x + t, i32(b.w)))].xyz;
  }
  dst[(b.y0 + y) * b.stride + b.x0 + u32(x)] = pack(s);
}
`;

const WARP = PACK + /* wgsl */ `
struct W { a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, sw: u32, sh: u32, ew: u32, eh: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> w: W;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read> valid: array<u32>;
@group(0) @binding(3) var<storage, read_write> fex: array<u32>;
@group(0) @binding(4) var<storage, read_write> vex: array<u32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = lin(g);
  if (i >= w.ew * w.eh) { return; }
  let x = f32(i % w.ew);
  let y = f32(i / w.ew);
  let sx = w.a * x + w.b * y + w.c;
  let sy = w.d * x + w.e * y + w.f;
  let sw = i32(w.sw);
  let sh = i32(w.sh);
  let nx = i32(floor(sx + 0.5));
  let ny = i32(floor(sy + 0.5));
  var v = 0u;
  if (nx >= 0 && nx < sw && ny >= 0 && ny < sh) {
    let j = u32(ny * sw + nx);
    v = (valid[j >> 2u] >> ((j & 3u) * 8u)) & 255u;
  }
  vex[i] = select(0u, 1u, v != 0u);
  let fx0 = floor(sx);
  let fy0 = floor(sy);
  let fx = sx - fx0;
  let fy = sy - fy0;
  let x0 = clamp(i32(fx0), 0, sw - 1);
  let x1 = clamp(i32(fx0) + 1, 0, sw - 1);
  let y0 = clamp(i32(fy0), 0, sh - 1);
  let y1 = clamp(i32(fy0) + 1, 0, sh - 1);
  let top = (1.0 - fx) * unpack(src[y0 * sw + x0]) + fx * unpack(src[y0 * sw + x1]);
  let bot = (1.0 - fx) * unpack(src[y1 * sw + x0]) + fx * unpack(src[y1 * sw + x1]);
  fex[i] = pack((1.0 - fy) * top + fy * bot);
}
`;

// ink_changed: blurs of the visible mask, and of frame and canvas weighted by it, then the change test
const INK1 = PACK + /* wgsl */ `
struct I { w: u32, h: u32, stride: u32, x0: u32, y0: u32, half: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> b: I;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> fex: array<u32>;
@group(0) @binding(3) var<storage, read> vex: array<u32>;
@group(0) @binding(4) var<storage, read> color: array<u32>;
@group(0) @binding(5) var<storage, read_write> ta: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> tb: array<vec4f>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = lin(g);
  if (i >= b.w * b.h) { return; }
  let x = i % b.w;
  let y = i32(i / b.w);
  var sa = vec4f(0.0);
  var sb = vec3f(0.0);
  let half = i32(b.half);
  for (var t = -half; t <= half; t++) {
    let yy = u32(reflect101(y + t, i32(b.h)));
    let j = yy * b.w + x;
    let wv = f32(vex[j]);
    let kt = k[u32(t + half)];
    sa += kt * vec4f(wv, unpack(fex[j]) * wv);
    sb += kt * unpack(color[(b.y0 + yy) * b.stride + b.x0 + x]) * wv;
  }
  ta[i] = sa;
  tb[i] = vec4f(sb, 0.0);
}
`;

const INK2 = PACK + /* wgsl */ `
struct I { w: u32, h: u32, half: u32, threshold: f32 }
@group(0) @binding(0) var<uniform> b: I;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> ta: array<vec4f>;
@group(0) @binding(3) var<storage, read> tb: array<vec4f>;
@group(0) @binding(4) var<storage, read> vex: array<u32>;
@group(0) @binding(5) var<storage, read_write> changed: array<u32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = lin(g);
  if (i >= b.w * b.h) { return; }
  let x = i32(i % b.w);
  let y = i / b.w;
  var sa = vec4f(0.0);
  var sb = vec3f(0.0);
  let half = i32(b.half);
  for (var t = -half; t <= half; t++) {
    let j = y * b.w + u32(reflect101(x + t, i32(b.w)));
    let kt = k[u32(t + half)];
    sa += kt * ta[j];
    sb += kt * tb[j].xyz;
  }
  let norm = sa.x;
  let diff = abs(sa.yzw - sb);
  let d = max(diff.x, max(diff.y, diff.z)) / max(norm, 1e-6);
  changed[i] = select(0u, 1u, d >= b.threshold && vex[i] != 0u && norm > 0.25);
}
`;

// a square min or max filter, one axis at a time; outside the box counts as `outside`
const MORPH = /* wgsl */ `
struct M { w: u32, h: u32, r: u32, axis: u32, op_max: u32, outside: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> m: M;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + g.y * ${WG * 65535}u;
  if (i >= m.w * m.h) { return; }
  let x = i32(i % m.w);
  let y = i32(i / m.w);
  let r = i32(m.r);
  var v = select(1u, 0u, m.op_max == 1u);
  for (var t = -r; t <= r; t++) {
    var xx = x;
    var yy = y;
    if (m.axis == 0u) { xx = x + t; } else { yy = y + t; }
    var s = m.outside;
    if (xx >= 0 && xx < i32(m.w) && yy >= 0 && yy < i32(m.h)) { s = src[u32(yy) * m.w + u32(xx)]; }
    if (m.op_max == 1u) { v = max(v, s); } else { v = min(v, s); }
  }
  dst[i] = v;
}
`;

const RULES = PACK + /* wgsl */ `
struct R { ew: u32, ix: u32, iy: u32, iw: u32, ih: u32, cw: u32, x0: u32, y0: u32,
           q: f32, qg: f32, faint: i32, diff: i32, gain: f32, stable: u32, has_ink: u32, _a: u32 }
@group(0) @binding(0) var<uniform> r: R;
@group(0) @binding(1) var<storage, read> fb: array<u32>;
@group(0) @binding(2) var<storage, read> cb: array<u32>;
@group(0) @binding(3) var<storage, read> fex: array<u32>;
@group(0) @binding(4) var<storage, read> vex: array<u32>;
@group(0) @binding(5) var<storage, read> ink: array<u32>;
@group(0) @binding(6) var<storage, read_write> color: array<u32>;
@group(0) @binding(7) var<storage, read_write> quality: array<f32>;
@group(0) @binding(8) var<storage, read_write> pending: array<u32>;
@group(0) @binding(9) var<storage, read_write> count: array<u32>;
@group(0) @binding(10) var<storage, read_write> res: array<atomic<u32>>;
var<workgroup> n_vis: atomic<u32>;
var<workgroup> n_com: atomic<u32>;
fn maxdiff(a: vec3f, b: vec3f) -> i32 { let d = abs(a - b); return i32(max(d.x, max(d.y, d.z))); }
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) { atomicStore(&n_vis, 0u); atomicStore(&n_com, 0u); }
  workgroupBarrier();
  let i = lin(g);
  if (i < r.iw * r.ih) {
    let x = i % r.iw;
    let y = i / r.iw;
    let e = (y + r.iy) * r.ew + x + r.ix;
    let c = (y + r.y0) * r.cw + x + r.x0;
    let vis = vex[e] != 0u;
    let qual = quality[c];
    let f = unpack(fex[e]);
    let d = maxdiff(unpack(fb[e]), unpack(cb[e]));
    let dp = maxdiff(f, unpack(pending[c]));
    let faint = d >= r.faint && (r.qg >= qual || (r.has_ink == 1u && ink[e] != 0u));
    var agree = qual > 0.0 && d < r.diff && !faint;
    let differ = vis && !agree;
    agree = agree && vis;
    var cnt = count[c];
    var take = false;
    if (agree) {
      take = r.q > qual * r.gain;  // a sharper view of the same thing
      cnt = 0u;
    } else if (differ) {
      if (cnt > 0u && dp < r.diff) {
        cnt += 1u;
      } else {
        pending[c] = pack(f);
        cnt = 1u;
      }
      if (cnt >= r.stable) {
        take = true;
        cnt = 0u;
        atomicAdd(&n_com, 1u);
      }
    }
    if (take) {
      color[c] = pack(f);
      quality[c] = r.q;
    }
    if (agree || differ) { count[c] = cnt; }
    if (vis) { atomicAdd(&n_vis, 1u); }
  }
  workgroupBarrier();
  if (li == 0u) {
    let a = atomicLoad(&n_com);
    let v = atomicLoad(&n_vis);
    if (a > 0u) { atomicAdd(&res[0], a); }
    if (v > 0u) { atomicAdd(&res[1], v); }
  }
}
`;

const SHOW = /* wgsl */ `
@group(0) @binding(0) var<storage, read> color: array<u32>;
@group(0) @binding(1) var<uniform> size: vec4u;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let v = color[u32(pos.y) * size.x + u32(pos.x)];
  return vec4f(f32(v & 255u), f32((v >> 8u) & 255u), f32((v >> 16u) & 255u), 255.0) / 255.0;
}
`;

export interface FrameResult { changed: number; processed: number }
export type Box = [number, number, number, number];

/** The valid screen region: the screen mask shrunk by `erodePx` (the image edge does not shrink it). */
export function validMask(mask: Uint8Array, w: number, h: number, erodePx: number): Uint8Array {
  if (erodePx <= 0) return mask;
  // pad by the radius with set pixels, so only the mask's own edges erode
  const r = erodePx, pw = w + 2 * r, ph = h + 2 * r;
  const padded = new Uint8Array(pw * ph).fill(1);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) padded[(y + r) * pw + x + r] = mask[y * w + x] ? 1 : 0;
  const e = erode(padded, pw, ph, 2 * r + 1);
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = e[(y + r) * pw + x + r] ? 255 : 0;
  return out;
}

/** What the GPU canvas and the processor's (composite-cpu.ts) share: the settings, and which part of the
 * canvas a frame can reach. */
export abstract class Canvas {
  readonly stable: number;
  protected cfg: Config['accumulate'];
  private validBox: [number, number, number, number] | null; // bounding box of valid[::8, ::8], x0 y0 x1 y1

  constructor(readonly size: [number, number], cfg: Config, readonly sw: number, readonly sh: number, valid: Uint8Array) {
    this.cfg = cfg.accumulate;
    this.stable = Math.min(Math.max(pyRound(cfg.accumulate.stable_seconds * cfg.sample_fps), 1), 255);
    let x0 = Infinity, y0 = Infinity, x1 = -1, y1 = -1;
    for (let y = 0; y < sh; y += 8) for (let x = 0; x < sw; x += 8) {
      if (valid[y * sw + x]) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    }
    this.validBox = x1 < 0 ? null : [x0, y0, x1, y1];
  }

  private clip(pts: [number, number][], pad: number): Box | null {
    const [w, h] = this.size;
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const x0 = Math.max(Math.floor(Math.min(...xs) - pad), 0), y0 = Math.max(Math.floor(Math.min(...ys) - pad), 0);
    const x1 = Math.min(Math.ceil(Math.max(...xs) + pad), w), y1 = Math.min(Math.ceil(Math.max(...ys) + pad), h);
    return x1 > x0 && y1 > y0 ? [x0, y0, x1, y1] : null;
  }

  /** Canvas bounding box of the valid screen region under `m`. */
  visibleBox(m: M3): Box | null {
    if (!this.validBox) return null;
    const [x0, y0, x1, y1] = this.validBox;
    const pts = [[x0, y0], [x1, y0], [x0, y1], [x1, y1]].map(([x, y]) => geo.apply(m, x, y));
    return this.clip(pts, 8 * geo.similarity(m).scale + 8);
  }

  /** Merge a frame; `m` maps screen -> canvas. */
  abstract add(frame: VideoFrame, m: M3): Promise<FrameResult>;
  /** The canvas as an ImageBitmap. */
  abstract image(): ImageBitmap;
  /** The canvas as RGBA bytes. */
  abstract pixels(): Promise<Uint8ClampedArray>;
}

export class GpuCanvas extends Canvas {
  private color: GPUBuffer;
  private quality: GPUBuffer;
  private pending: GPUBuffer;
  private count: GPUBuffer;
  private frame: GPUBuffer;
  private frameBlur: GPUBuffer;
  private valid: GPUBuffer;
  private fex!: GPUBuffer;
  private vex!: GPUBuffer;
  private fb!: GPUBuffer;
  private cb!: GPUBuffer;
  private tmp: GPUBuffer;
  private ta: GPUBuffer;
  private tb: GPUBuffer;
  private ink0!: GPUBuffer;
  private ink1!: GPUBuffer;
  private res: GPUBuffer;
  private pool: GPUBuffer[] = [];
  private used = 0;

  constructor(private gpu: Gpu, size: [number, number], cfg: Config, sw: number, sh: number, valid: Uint8Array) {
    super(size, cfg, sw, sh, valid);
    const [w, h] = size;
    const n = w * h;
    this.color = gpu.buffer(n * 4);
    const [bb, bg, br] = cfg.canvas.background;
    gpu.device.queue.writeBuffer(this.color, 0, new Uint32Array(n).fill(br | (bg << 8) | (bb << 16) | (255 << 24)));
    this.quality = gpu.buffer(n * 4);
    this.pending = gpu.buffer(n * 4);
    this.count = gpu.buffer(n * 4);
    this.frame = gpu.buffer(sw * sh * 4);
    this.frameBlur = gpu.buffer(sw * sh * 4);
    this.valid = gpu.buffer(sw * sh);
    gpu.device.queue.writeBuffer(this.valid, 0, valid.buffer, valid.byteOffset, Math.ceil(valid.length / 4) * 4);
    for (const name of ['fex', 'vex', 'fb', 'cb', 'ink0', 'ink1'] as const) this[name] = gpu.buffer(n * 4);
    this.tmp = gpu.buffer(Math.max(n, sw * sh) * 16);
    this.ta = gpu.buffer(n * 16);
    this.tb = gpu.buffer(n * 16);
    this.res = gpu.buffer(8);
  }

  /** A storage or uniform buffer from this frame's pool, holding `data`. */
  private data(data: ArrayBufferView, usage: number): GPUBuffer {
    const size = Math.max(64, Math.ceil(data.byteLength / 64) * 64);
    let b = this.pool[this.used];
    if (!b || b.size < size || b.usage !== (usage | GPUBufferUsage.COPY_DST)) {
      b?.destroy();
      b = this.gpu.device.createBuffer({ size, usage: usage | GPUBufferUsage.COPY_DST });
      this.pool[this.used] = b;
    }
    this.used++;
    this.gpu.device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    return b;
  }
  private u(...words: number[]): GPUBuffer {
    return this.data(new Uint32Array(words), GPUBufferUsage.UNIFORM);
  }
  private kernel(sigma: number): [GPUBuffer, number] {
    const k = gaussKernel(sigma, true);
    return [this.data(new Float32Array(k), GPUBufferUsage.STORAGE), (k.length - 1) / 2];
  }

  private run(enc: GPUCommandEncoder, code: string, entry: string, bind: (GPUBuffer | GPUExternalTexture)[], n: number) {
    const [gx, gy] = groups(n, WG);
    this.gpu.dispatch(enc, this.gpu.pipeline(code, entry), bind, gx, gy);
  }

  /** Blur an 8-bit region (w x h at x0, y0 of a buffer `stride` wide) into `dst` (at dx0, dy0 of a buffer
   * `dstride` wide), rounded. */
  private blur(enc: GPUCommandEncoder, src: GPUBuffer, stride: number, x0: number, y0: number, w: number, h: number,
    dst: GPUBuffer, dstride: number, dx0: number, dy0: number, sigma: number) {
    const [k, half] = this.kernel(sigma);
    this.run(enc, BLUR, 'along_y', [this.u(w, h, stride, x0, y0, half, 0, 0), k, src, this.tmp], w * h);
    this.run(enc, BLUR2, 'along_x', [this.u(w, h, dstride, dx0, dy0, half, 0, 0), k, this.tmp, dst], w * h);
  }

  async add(frame: VideoFrame, m: M3): Promise<FrameResult> {
    const box = this.visibleBox(m);
    if (!box) return { changed: 0, processed: 0 };
    this.used = 0;
    const g = this.gpu, cfg = this.cfg;
    const [w, h] = this.size;
    const { sw, sh } = this;
    const enc = g.device.createCommandEncoder();
    const tex = g.device.importExternalTexture({ source: frame });
    this.run(enc, UPLOAD, 'main', [this.u(sw, sh, 0, 0), tex, this.frame], sw * sh);
    const scale = geo.similarity(m).scale;
    let src = this.frame;
    if (scale < 0.6) { // zoomed in further than the canvas resolution: low-pass before shrinking
      this.blur(enc, this.frame, sw, 0, 0, sw, sh, this.frameBlur, sw, 0, 0, 0.4 / scale);
      src = this.frameBlur;
    }
    const [x0, y0, x1, y1] = box;
    const ex0 = Math.max(x0 - MARGIN, 0), ey0 = Math.max(y0 - MARGIN, 0), ex1 = Math.min(x1 + MARGIN, w), ey1 = Math.min(y1 + MARGIN, h);
    const ew = ex1 - ex0, eh = ey1 - ey0;
    const inv = geo.inv(geo.mul(geo.scaleTranslate(1, -ex0, -ey0), m)); // box pixel -> screen pixel
    const wu = new ArrayBuffer(48);
    new Float32Array(wu, 0, 6).set([inv[0], inv[1], inv[2], inv[3], inv[4], inv[5]]);
    new Uint32Array(wu, 24, 4).set([sw, sh, ew, eh]);
    this.run(enc, WARP, 'main', [this.data(new Uint8Array(wu), GPUBufferUsage.UNIFORM), src, this.valid, this.fex, this.vex], ew * eh);
    const q = Math.min(1 / scale, 4.0);
    const sigma = Math.max(0.8, 0.5 * scale);
    this.blur(enc, this.fex, ew, 0, 0, ew, eh, this.fb, ew, 0, 0, sigma);
    this.blur(enc, this.color, w, ex0, ey0, ew, eh, this.cb, ew, 0, 0, sigma);
    // ink only matters where the canvas holds pixels sharper than this view; elsewhere the rules ignore it
    const hasInk = cfg.ink_threshold > 0;
    if (hasInk) {
      const s = INK_SIGMA * scale;
      const [k, half] = this.kernel(s);
      this.run(enc, INK1, 'main', [this.u(ew, eh, w, ex0, ey0, half, 0, 0), k, this.fex, this.vex, this.color, this.ta, this.tb], ew * eh);
      const iu = new ArrayBuffer(16);
      new Uint32Array(iu, 0, 3).set([ew, eh, half]);
      new Float32Array(iu, 12, 1).set([cfg.ink_threshold]);
      this.run(enc, INK2, 'main', [this.data(new Uint8Array(iu), GPUBufferUsage.UNIFORM), k, this.ta, this.tb, this.vex, this.ink0], ew * eh);
      // opening by a (2a+1) square, then growing back by g: max filters compose, so min a, max a + g
      const a = Math.max(1, pyRound(scale)), grow = Math.ceil(INK_GROW * scale);
      const morph = (srcB: GPUBuffer, dst: GPUBuffer, r: number, axis: number, opMax: number, outside: number) =>
        this.run(enc, MORPH, 'main', [this.u(ew, eh, r, axis, opMax, outside, 0, 0), srcB, dst], ew * eh);
      morph(this.ink0, this.ink1, a, 0, 0, 1);
      morph(this.ink1, this.ink0, a, 1, 0, 1);
      morph(this.ink0, this.ink1, a + grow, 0, 1, 0);
      morph(this.ink1, this.ink0, a + grow, 1, 1, 0);
    }
    enc.clearBuffer(this.res);
    const iw = x1 - x0, ih = y1 - y0;
    const ru = new ArrayBuffer(64);
    const ruU = new Uint32Array(ru), ruF = new Float32Array(ru), ruI = new Int32Array(ru);
    ruU.set([ew, x0 - ex0, y0 - ey0, iw, ih, w, x0, y0]);
    ruF[8] = q; ruF[9] = q * cfg.refine_gain; ruI[10] = cfg.faint_threshold; ruI[11] = cfg.diff_threshold;
    ruF[12] = cfg.refine_gain; ruU[13] = this.stable; ruU[14] = hasInk ? 1 : 0;
    this.run(enc, RULES, 'main', [this.data(ruU, GPUBufferUsage.UNIFORM), this.fb, this.cb, this.fex, this.vex, this.ink0,
      this.color, this.quality, this.pending, this.count, this.res], iw * ih);
    const [r] = await g.read(enc, [[this.res, 0, 8]]);
    const [changed, processed] = new Uint32Array(r);
    return { changed, processed };
  }

  private show?: { canvas: OffscreenCanvas; ctx: GPUCanvasContext; pipe: GPURenderPipeline; bind: GPUBindGroup };

  /** Drawn on the GPU into an OffscreenCanvas. */
  image(): ImageBitmap {
    const g = this.gpu;
    if (!this.show) {
      const [w, h] = this.size;
      const canvas = new OffscreenCanvas(w, h);
      const ctx = canvas.getContext('webgpu')!;
      const format = navigator.gpu.getPreferredCanvasFormat();
      ctx.configure({ device: g.device, format, alphaMode: 'opaque' });
      const module = g.device.createShaderModule({ code: SHOW });
      const pipe = g.device.createRenderPipeline({ layout: 'auto', vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format }] } });
      const bind = g.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.color } }, { binding: 1, resource: { buffer: g.uniform(new Uint32Array([w, h, 0, 0])) } }] });
      this.show = { canvas, ctx, pipe, bind };
    }
    const { canvas, ctx, pipe, bind } = this.show;
    const enc = g.device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bind);
    pass.draw(3);
    pass.end();
    g.device.queue.submit([enc.finish()]);
    return canvas.transferToImageBitmap();
  }

  async pixels(): Promise<Uint8ClampedArray> {
    const [w, h] = this.size;
    const [raw] = await this.gpu.read(this.gpu.device.createCommandEncoder(), [[this.color, 0, w * h * 4]]);
    return new Uint8ClampedArray(raw);
  }
}

/** Feeds frames into the canvas; while the view holds still (under a screen pixel over 1000 px), keeps the
 * earlier transform, ignoring sub-pixel re-estimation jitter (accumulate.Compositor for a GPU canvas). */
export class Compositor {
  private prev: M3 | null = null;
  full = 0;

  constructor(readonly canvas: Canvas) {}

  async add(frame: VideoFrame, m: M3): Promise<FrameResult> {
    if (this.prev) {
      const [ds, shift] = geo.viewChange(this.prev, m);
      if (ds < 0.003 && shift < geo.similarity(m).scale) m = this.prev;
    }
    this.prev = m;
    this.full++;
    return this.canvas.add(frame, m);
  }
}
