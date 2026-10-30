/**
 * Nearest and second-nearest map descriptor (L2) for each query descriptor, by brute force on the GPU.
 * artlapse uses FLANN's approximate kd-trees; the map holds a few thousand descriptors at most, so the
 * exact search costs little here. Descriptors are SIFT bytes, packed four to a word.
 */
import { Gpu } from './gpu';

const WG = 64;

function shader(dot4: boolean): string {
  const dot = dot4
    ? 'fn dot4(a: u32, b: u32) -> u32 { return dot4U8Packed(a, b); }'
    : `fn dot4(a: u32, b: u32) -> u32 {
  var s = 0u;
  for (var k = 0u; k < 32u; k += 8u) { s += ((a >> k) & 255u) * ((b >> k) & 255u); }
  return s;
}`;
  return /* wgsl */ `
${dot}
struct P { nq: u32, nm: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> query: array<u32>;
@group(0) @binding(2) var<storage, read> map: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4u>;
var<workgroup> tile: array<u32, ${WG * 32}>;
var<workgroup> tnorm: array<u32, ${WG}>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let qi = gid.x;
  var q: array<u32, 32>;
  var qn = 0u;
  if (qi < p.nq) {
    for (var k = 0u; k < 32u; k++) { q[k] = query[qi * 32u + k]; qn += dot4(q[k], q[k]); }
  }
  var b1 = 0xffffffffu;
  var b2 = 0xffffffffu;
  var i1 = 0u;
  var i2 = 0u;
  for (var t0 = 0u; t0 < p.nm; t0 += ${WG}u) {
    let mi = t0 + li;
    var n = 0u;
    for (var k = 0u; k < 32u; k++) {
      var v = 0u;
      if (mi < p.nm) { v = map[mi * 32u + k]; }
      tile[li * 32u + k] = v;
      n += dot4(v, v);
    }
    tnorm[li] = n;
    workgroupBarrier();
    let count = min(${WG}u, p.nm - t0);
    for (var j = 0u; j < count; j++) {
      var d = 0u;
      for (var k = 0u; k < 32u; k++) { d += dot4(q[k], tile[j * 32u + k]); }
      let dist = qn + tnorm[j] - 2u * d;
      if (dist < b1) { b2 = b1; i2 = i1; b1 = dist; i1 = t0 + j; }
      else if (dist < b2) { b2 = dist; i2 = t0 + j; }
    }
    workgroupBarrier();
  }
  if (qi < p.nq) { out[qi] = vec4u(i1, b1, i2, b2); }
}
`;
}

/** What register needs of matching: this GPU one, or the processor's (cpu.ts). */
export interface Knn {
  /** Replace the map's descriptors ((N, 128) bytes). */
  setMap(desc: Uint8Array, n: number): void;
  /** For each query set: per query [best index, best squared distance, second index, second squared
   * distance] (Uint32Array of 4 per query). */
  knn2(queries: Uint8Array[]): Promise<Uint32Array[]>;
}

export class Matcher implements Knn {
  private map: GPUBuffer;
  private mapCap = 0;
  private nm = 0;
  private query: GPUBuffer;
  private out: GPUBuffer;
  private qCap = 0;
  private code: string;
  private params: GPUBuffer;

  constructor(private gpu: Gpu) {
    this.code = shader(navigator.gpu.wgslLanguageFeatures?.has('packed_4x8_integer_dot_product') ?? false);
    this.map = gpu.buffer(0);
    this.query = gpu.buffer(0);
    this.out = gpu.buffer(0);
    this.params = gpu.uniform(new Uint32Array(4));
  }

  /** Replace the map's descriptors ((N, 128) bytes). */
  setMap(desc: Uint8Array, n: number): void {
    if (n > this.mapCap) {
      this.map.destroy();
      this.mapCap = Math.max(n, 2 * this.mapCap, 1024);
      this.map = this.gpu.buffer(this.mapCap * 128);
    }
    if (n) this.gpu.device.queue.writeBuffer(this.map, 0, desc.buffer, desc.byteOffset, n * 128);
    this.nm = n;
  }

  /** For each query set: per query [best index, best squared distance, second index, second squared
   * distance] (Uint32Array of 4 per query). */
  async knn2(queries: Uint8Array[]): Promise<Uint32Array[]> {
    const counts = queries.map(q => q.length / 128);
    const total = counts.reduce((a, b) => a + b, 0);
    if (!total || this.nm < 2) return counts.map(() => new Uint32Array(0));
    if (total > this.qCap) {
      this.query.destroy();
      this.out.destroy();
      this.qCap = Math.max(total, 4096);
      this.query = this.gpu.buffer(this.qCap * 128);
      this.out = this.gpu.buffer(this.qCap * 16);
    }
    let at = 0;
    for (const q of queries) {
      if (q.length) this.gpu.device.queue.writeBuffer(this.query, at * 128, q.buffer, q.byteOffset, q.length);
      at += q.length / 128;
    }
    const enc = this.gpu.device.createCommandEncoder();
    const p = this.gpu.pipeline(this.code, 'main');
    this.gpu.device.queue.writeBuffer(this.params, 0, new Uint32Array([total, this.nm, 0, 0]));
    this.gpu.dispatch(enc, p, [this.params, this.query, this.map, this.out], Math.ceil(total / WG));
    const [raw] = await this.gpu.read(enc, [[this.out, 0, total * 16]]);
    const all = new Uint32Array(raw);
    const res: Uint32Array[] = [];
    at = 0;
    for (const c of counts) {
      res.push(all.subarray(at * 4, (at + c) * 4));
      at += c;
    }
    return res;
  }
}
