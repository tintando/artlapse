/**
 * Registration: for every sampled frame, where the canvas sits on screen (artlapse's register.py, whose
 * docstrings explain the method). SIFT and matching run on the GPU; RANSAC, the feature map and the
 * pixel votes between views on the CPU.
 */
import { Config } from './config';
import * as geo from './geometry';
import { M3 } from './geometry';
import { boxFilter, erode, hullArea, pyRound, warpLinear, warpNearest } from './imgops';
import { Matcher } from './match';
import { estimateSimilarity, rng } from './ransac';
import { Features, Sift } from './sift';

export const PATCH = 7;
const PATCH_CONTRAST = 16.0;
const PATCH_ERROR = 8.0;
export const READ_AHEAD = 240;

export enum Status { UNSEEN = 0, MATCHED = 1, STATIC = 2, LOST = 3 }

export class Track {
  constructor(readonly fps: number, readonly status: Uint8Array, readonly m: Float64Array, readonly inliers: Int32Array) {}

  static empty(count: number, fps: number): Track {
    const m = new Float64Array(count * 9);
    for (let i = 0; i < count; i++) m.set(geo.eye(), i * 9);
    return new Track(fps, new Uint8Array(count), m, new Int32Array(count));
  }

  get length(): number { return this.status.length; }
  ok(n: number): boolean { return this.status[n] === Status.MATCHED || this.status[n] === Status.STATIC; }
  at(n: number): M3 { return this.m.slice(n * 9, n * 9 + 9); }
  set(n: number, m: M3): void { this.m.set(m, n * 9); }

  toJSON() {
    return { fps: this.fps, status: Array.from(this.status), m: Array.from(this.m), inliers: Array.from(this.inliers) };
  }
}

/** 255 where the art program's canvas view may be seen, at `scale` x full resolution. */
export function screenMask(cfg: Config, width: number, height: number, scale: number): Uint8Array {
  const mask = new Uint8Array(width * height);
  const fill = (rect: number[], value: number) => {
    const [x, y, w, h] = rect.map(v => pyRound(v * scale));
    const y0 = Math.min(Math.max(y, 0), height), y1 = Math.min(Math.max(y + h, 0), height);
    const x0 = Math.min(Math.max(x, 0), width), x1 = Math.min(Math.max(x + w, 0), width);
    for (let r = y0; r < y1; r++) mask.fill(value, r * width + x0, r * width + x1);
  };
  fill(cfg.workspace, 255);
  for (const r of cfg.exclude) fill(r, 0);
  return mask;
}

/** Screen->canvas transform of the reference frame, and the canvas size [w, h], both even. */
export function referenceTransform(cfg: Config): [M3, [number, number]] {
  const [x, y, w, h] = cfg.reference.target;
  if (w <= 0 || h <= 0) throw new Error('the target must be a non-empty rectangle');
  const k = cfg.canvas.long_side / Math.max(w, h);
  return [geo.scaleTranslate(k, -x * k, -y * k), [2 * pyRound((w * k) / 2), 2 * pyRound((h * k) / 2)]];
}

export function sampleRange(cfg: Config, count: number): [number, number] {
  const lo = Math.max(0, pyRound(cfg.start * cfg.sample_fps));
  const hi = cfg.end === null ? count : Math.min(count, pyRound(cfg.end * cfg.sample_fps));
  return [lo, hi];
}

/** The stretches [a, b) of sample indices that are processed: start to end, without the cuts. */
export function sampleSpans(cfg: Config, count: number): [number, number][] {
  let [lo, hi] = sampleRange(cfg, count);
  const spans: [number, number][] = [];
  const cuts = cfg.cuts.map(([a, b]) => [pyRound(a * cfg.sample_fps), pyRound(b * cfg.sample_fps)]).sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  for (const [c0, c1] of cuts) {
    if (c0 > lo) spans.push([lo, Math.min(c0, hi)]);
    lo = Math.max(lo, c1);
    if (lo >= hi) break;
  }
  if (lo < hi) spans.push([lo, hi]);
  return spans.filter(([a, b]) => a < b);
}

/** How many processed samples come before each sample index: its stream time with the cuts left out. */
export function keptSamples(cfg: Config, index: number[]): number[] {
  const spans = sampleSpans(cfg, Math.max(0, ...index) + 1);
  return index.map(n => spans.reduce((s, [a, b]) => s + Math.min(Math.max(n - a, 0), b - a), 0));
}

/** Canvas-space features holding the latest appearance of every canvas area (register.py's FeatureMap). */
class FeatureMap {
  slots = new Map<string, number>();
  pts: number[] = []; // x, y
  desc: Uint8Array[] = [];
  level: number[] = [];
  alive: boolean[] = [];
  indexRows: number[] = []; // map row of each descriptor on the GPU
  dirty = true;
  aliveCount = 0;

  get size(): number { return this.aliveCount; }

  add(canvasPts: Float64Array, desc: Uint8Array, cell: number, seen: ((x: number, y: number) => boolean) | null): void {
    const level = pyRound(Math.log2(cell));
    if (seen && this.aliveCount) {
      for (let r = 0; r < this.alive.length; r++) {
        if (!this.alive[r] || this.level[r] < level - 1 || !seen(this.pts[2 * r], this.pts[2 * r + 1])) continue;
        const size = 2 ** this.level[r];
        this.slots.delete(`${this.level[r]},${Math.floor(this.pts[2 * r] / size)},${Math.floor(this.pts[2 * r + 1] / size)}`);
        this.alive[r] = false;
        this.aliveCount--;
      }
    }
    const size = 2 ** level;
    const n = canvasPts.length / 2;
    for (let i = 0; i < n; i++) {
      const x = canvasPts[2 * i], y = canvasPts[2 * i + 1];
      const d = desc.subarray(i * 128, i * 128 + 128);
      const key = `${level},${Math.floor(x / size)},${Math.floor(y / size)}`;
      const slot = this.slots.get(key);
      if (slot === undefined) {
        this.slots.set(key, this.alive.length);
        this.pts.push(x, y);
        this.desc.push(d);
        this.level.push(level);
        this.alive.push(true);
        this.aliveCount++;
      } else {
        this.pts[2 * slot] = x;
        this.pts[2 * slot + 1] = y;
        this.desc[slot] = d;
      }
    }
    if (this.alive.length > 2 * Math.max(this.aliveCount, 1000)) this.compact();
    this.indexRows = [];
    for (let r = 0; r < this.alive.length; r++) if (this.alive[r]) this.indexRows.push(r);
    this.dirty = true;
  }

  private compact(): void {
    const keep: number[] = [];
    const remap = new Map<number, number>();
    for (let r = 0; r < this.alive.length; r++) if (this.alive[r]) { remap.set(r, keep.length); keep.push(r); }
    const slots = new Map<string, number>();
    for (const [k, v] of this.slots) if (remap.has(v)) slots.set(k, remap.get(v)!);
    this.slots = slots;
    this.pts = keep.flatMap(r => [this.pts[2 * r], this.pts[2 * r + 1]]);
    this.desc = keep.map(r => this.desc[r]);
    this.level = keep.map(r => this.level[r]);
    this.alive = keep.map(() => true);
  }

  /** The alive descriptors, packed for the GPU. */
  packed(): Uint8Array {
    const out = new Uint8Array(this.indexRows.length * 128);
    this.indexRows.forEach((r, i) => out.set(this.desc[r], i * 128));
    return out;
  }
}

interface Frame { m: M3; gray: Uint8Array }
type View = [M3, number, boolean]; // screen(full) -> canvas, inliers, flipped

export interface Decision {
  matches: number[]; map: number; views: Record<string, unknown>[]; ref: string | null;
  votes: unknown[][]; won?: number | string; keyframe?: boolean;
}

export class Registrar {
  readonly r: number;
  readonly aw: number;
  readonly ah: number;
  readonly mask: Uint8Array;
  readonly maskFlipped: Uint8Array;
  private maskCount: number;
  private mirror: M3;
  private feat: { gray: Uint8Array | null; f: Map<boolean, Features> } = { gray: null, f: new Map() };
  private tex: { gray: Uint8Array | null; t: Uint8Array | null } = { gray: null, t: null };
  map = new FeatureMap();
  flipped = false;
  last: Frame | null = null;
  anchor: Frame | null = null;
  lastKey: Frame | null = null;
  decision: Decision | null = null;
  random: () => number = rng(0);

  constructor(readonly cfg: Config, fullWidth: number, fullHeight: number, private sift: Sift, private matcher: Matcher) {
    const rc = cfg.register;
    this.r = rc.analysis_width / fullWidth;
    this.aw = rc.analysis_width;
    this.ah = pyRound((fullHeight * this.r) / 2) * 2;
    this.mask = screenMask(cfg, this.aw, this.ah, this.r);
    this.maskFlipped = new Uint8Array(this.mask.length);
    for (let y = 0; y < this.ah; y++) for (let x = 0; x < this.aw; x++) this.maskFlipped[y * this.aw + x] = this.mask[y * this.aw + this.aw - 1 - x];
    this.maskCount = this.mask.reduce((s, v) => s + (v ? 1 : 0), 0);
    this.mirror = geo.mirrorX(this.aw);
  }

  private a2c(m: M3): M3 { return geo.mul(m, geo.scaleTranslate(1 / this.r)); }
  private c2full(m: M3): M3 { return geo.mul(m, geo.scaleTranslate(this.r)); }

  /** Extract features of `gray` in the flip states not yet extracted for it. */
  private async detect(gray: Uint8Array, states: boolean[]): Promise<void> {
    if (gray !== this.feat.gray) this.feat = { gray, f: new Map() };
    const todo = states.filter(s => !this.feat.f.has(s));
    if (!todo.length) return;
    const found = await this.sift.detect(gray, todo, todo.map(f => (f ? this.maskFlipped : this.mask)));
    todo.forEach((s, i) => this.feat.f.set(s, found[i]));
  }

  private async features(gray: Uint8Array, flipped: boolean): Promise<Features> {
    await this.detect(gray, [flipped]);
    return this.feat.f.get(flipped)!;
  }

  async seed(gray: Uint8Array, m: M3): Promise<void> {
    this.map = new FeatureMap();
    this.flipped = geo.similarity(m).flipped;
    const frame = { m, gray };
    await this.addKeyframe(frame);
    this.last = this.anchor = frame;
  }

  private async addKeyframe(frame: Frame): Promise<void> {
    const flipped = geo.similarity(frame.m).flipped;
    const { pts, desc } = await this.features(frame.gray, flipped);
    const n = pts.length / 2;
    if (!n) return;
    let ma = this.a2c(frame.m);
    if (flipped) ma = geo.mul(ma, this.mirror);
    const canvas = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) [canvas[2 * i], canvas[2 * i + 1]] = geo.apply(ma, pts[2 * i], pts[2 * i + 1]);
    const cell = this.cfg.register.map_cell_px * geo.similarity(ma).scale;
    const toScreen = geo.inv(this.a2c(frame.m));
    const seen = (x: number, y: number) => {
      const [sx, sy] = geo.apply(toScreen, x, y);
      const ix = pyRound(sx), iy = pyRound(sy);
      return ix >= 0 && ix < this.aw && iy >= 0 && iy < this.ah && this.mask[iy * this.aw + ix] > 0;
    };
    this.map.add(canvas, desc, cell, seen);
    this.lastKey = frame;
  }

  /** Ratio-test matches of several descriptor sets against the map: per set, (query indices, canvas points). */
  private async match(descs: Uint8Array[]): Promise<[number[], number[]][]> {
    if (this.map.dirty) {
      this.matcher.setMap(this.map.packed(), this.map.indexRows.length);
      this.map.dirty = false;
    }
    if (this.map.indexRows.length < 2) return descs.map(() => [[], []]);
    const res = await this.matcher.knn2(descs.map(d => (d.length >= 256 ? d : new Uint8Array(0))));
    const ratio = this.cfg.register.ratio;
    return res.map(r => {
      const qi: number[] = [], pts: number[] = [];
      for (let q = 0; q < r.length / 4; q++) {
        const best = this.map.indexRows[r[4 * q]], second = this.map.indexRows[r[4 * q + 2]];
        const db = Math.sqrt(r[4 * q + 1]), ds = Math.sqrt(r[4 * q + 3]);
        const bx = this.map.pts[2 * best], by = this.map.pts[2 * best + 1];
        const near = Math.hypot(bx - this.map.pts[2 * second], by - this.map.pts[2 * second + 1]) < 4.0;
        if (db < ratio * ds || near) { qi.push(q); pts.push(bx, by); }
      }
      return [qi, pts];
    });
  }

  private fit(src: Float64Array, dst: Float64Array): [M3 | null, Uint8Array] {
    const n = src.length / 2;
    if (n < 3) return [null, new Uint8Array(n)];
    const r = estimateSimilarity(src, dst, this.cfg.register.ransac_px, this.random);
    return r ? [r.m, r.inliers] : [null, new Uint8Array(n)];
  }

  private inliersOf(c2s: M3, src: Float64Array, dst: Float64Array): Uint8Array {
    const n = src.length / 2, out = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const [x, y] = geo.apply(c2s, src[2 * i], src[2 * i + 1]);
      out[i] = Math.hypot(x - dst[2 * i], y - dst[2 * i + 1]) < this.cfg.register.ransac_px ? 1 : 0;
    }
    return out;
  }

  /** The earlier matched frame `ref` moved by the view change `m` implies, and where that is defined. */
  private predict(m: M3, ref: Frame): [Uint8Array, Uint8Array] {
    const rel = geo.mul(geo.inv(this.a2c(ref.m)), this.a2c(m));
    const moved = warpLinear(ref.gray, this.aw, this.ah, rel);
    const valid = warpNearest(this.mask, this.aw, this.ah, rel);
    for (let i = 0; i < valid.length; i++) valid[i] = valid[i] && this.mask[i] ? 1 : 0;
    return [moved, valid];
  }

  /** How many pixels of this frame views `a` and `b` each explain better (register.py's _compare). */
  private compare(gray: Uint8Array, a: M3, b: M3, ref: Frame): [number, number] | null {
    const [pa, va] = this.predict(a, ref);
    const [pb, vb] = this.predict(b, ref);
    const enough = 0.02 * this.maskCount;
    let sa = 0, sb = 0;
    for (let i = 0; i < va.length; i++) { sa += va[i]; sb += vb[i]; }
    if (sa < enough || sb < enough) return [sa >= enough ? 1 : 0, sb >= enough ? 1 : 0];
    const n = gray.length;
    const ea = new Float32Array(n), eb = new Float32Array(n);
    for (let i = 0; i < n; i++) { ea[i] = Math.abs(gray[i] - pa[i]); eb[i] = Math.abs(gray[i] - pb[i]); }
    const onlyA = new Uint8Array(n), onlyB = new Uint8Array(n);
    let anyA = false, anyB = false;
    for (let i = 0; i < n; i++) {
      onlyA[i] = va[i] && !vb[i] ? 1 : 0;
      onlyB[i] = vb[i] && !va[i] ? 1 : 0;
      anyA ||= !!onlyA[i];
      anyB ||= !!onlyB[i];
    }
    const patchA = anyA ? this.matchedPatches(gray, ea, va) : null;
    const patchB = anyB ? this.matchedPatches(gray, eb, vb) : null;
    let wa = 0, wb = 0;
    for (let i = 0; i < n; i++) {
      if (va[i] && vb[i] && Math.abs(pa[i] - pb[i]) > 16 && Math.min(ea[i], eb[i]) < 16) {
        if (ea[i] + 8 < eb[i]) wa++;
        if (eb[i] + 8 < ea[i]) wb++;
      }
      if (onlyA[i] && patchA![i]) wa++;
      if (onlyB[i] && patchB![i]) wb++;
    }
    return wa + wb < 50 ? null : [wa, wb];
  }

  /** Pixels whose PATCH x PATCH neighbourhood has texture and is matched by a prediction with errors
   * `err`, all of it inside where the prediction is defined (`valid`). */
  private matchedPatches(gray: Uint8Array, err: Float32Array, valid: Uint8Array): Uint8Array {
    const w = this.aw, h = this.ah, n = w * h;
    if (gray !== this.tex.gray) {
      const g = new Float32Array(n), g2 = new Float32Array(n);
      for (let i = 0; i < n; i++) { g[i] = gray[i]; g2[i] = gray[i] * gray[i]; }
      const mean = boxFilter(g, w, h, PATCH), sq = boxFilter(g2, w, h, PATCH);
      const t = new Uint8Array(n);
      for (let i = 0; i < n; i++) t[i] = sq[i] - mean[i] * mean[i] > PATCH_CONTRAST ** 2 ? 1 : 0;
      this.tex = { gray, t };
    }
    const inside = erode(valid, w, h, PATCH);
    const e = boxFilter(err, w, h, PATCH);
    const out = new Uint8Array(n);
    const t = this.tex.t!;
    for (let i = 0; i < n; i++) out[i] = inside[i] && t[i] && e[i] < PATCH_ERROR ? 1 : 0;
    return out;
  }

  /** Match against the map in one flip state: the convincing views, and the number of matches. */
  private async try(gray: Uint8Array, flipped: boolean, matched: [number[], number[]]): Promise<[[M3, number][], number]> {
    const rc = this.cfg.register;
    const { pts } = await this.features(gray, flipped);
    const [qi, canvasList] = matched;
    if (qi.length < rc.min_inliers) return [[], qi.length];
    const canvas = Float64Array.from(canvasList);
    const screen = new Float64Array(2 * qi.length);
    qi.forEach((q, i) => { screen[2 * i] = pts[2 * q]; screen[2 * i + 1] = pts[2 * q + 1]; });
    const [c2s, inl] = this.fit(canvas, screen);
    if (!c2s) return [[], qi.length];
    const candidates: M3[] = [c2s];
    const rest: number[] = [];
    let nInl = 0;
    inl.forEach((v, i) => { if (!v) rest.push(i); else nInl++; });
    if (rest.length >= rc.min_inliers) {
      const pick = (arr: Float64Array, idx: number[]) => Float64Array.from(idx.flatMap(i => [arr[2 * i], arr[2 * i + 1]]));
      const [second, inl2] = this.fit(pick(canvas, rest), pick(screen, rest));
      if (second && inl2.reduce((s, v) => s + v, 0) >= rc.min_inliers) candidates.push(second);
    }
    if (this.anchor && geo.similarity(this.anchor.m).flipped === flipped) {
      let prev = this.a2c(this.anchor.m);
      if (flipped) prev = geo.mul(prev, this.mirror);
      const prevInl = this.inliersOf(geo.inv(prev), canvas, screen);
      const idx: number[] = [];
      prevInl.forEach((v, i) => { if (v) idx.push(i); });
      const all = Array.from({ length: qi.length }, (_, i) => i);
      if (idx.length >= Math.max(rc.min_inliers, rc.keep_view_ratio * nInl)
          && hullArea(screen, idx) >= rc.keep_view_spread * hullArea(screen, all)) {
        const pick = (arr: Float64Array) => Float64Array.from(idx.flatMap(i => [arr[2 * i], arr[2 * i + 1]]));
        const [keep] = this.fit(pick(canvas), pick(screen));
        if (keep) candidates.unshift(keep);
      }
    }
    const views: [M3, number][] = [];
    for (const c of candidates) {
      const n = this.inliersOf(c, canvas, screen).reduce((s, v) => s + v, 0);
      if (n < rc.min_inliers || n < rc.min_inlier_ratio * qi.length) continue;
      if (!geo.isSimilarity(c, 1e-2)) continue;
      let ma = geo.inv(c);
      if (flipped) ma = geo.mul(ma, this.mirror);
      views.push([this.c2full(ma), n]);
    }
    return [views, qi.length];
  }

  async step(gray: Uint8Array): Promise<[Status, M3 | null, number]> {
    this.decision = null;
    const rc = this.cfg.register;
    if (this.last) {
      let s = 0;
      const lg = this.last.gray;
      for (let i = 0; i < gray.length; i++) if (this.mask[i]) s += Math.abs(gray[i] - lg[i]);
      if (s / this.maskCount < rc.static_diff) return [Status.STATIC, this.last.m, 0];
    }
    const order = [this.flipped, !this.flipped];
    await this.detect(gray, order);
    const matched = await this.match(order.map(f => this.feat.f.get(f)!.desc));
    const views: View[] = [];
    const matches: number[] = [];
    for (let i = 0; i < 2; i++) {
      const [found, n] = await this.try(gray, order[i], matched[i]);
      for (const [m, k] of found) views.push([m, k, order[i]]);
      matches.push(n);
    }
    const d: Decision = this.decision = {
      matches, map: this.map.size, views: views.map(v => this.describe(v[0], v[1])),
      ref: this.last ? 'last' : this.anchor ? 'anchor' : null, votes: [],
    };
    if (!views.length) {
      this.last = null;
      return [Status.LOST, null, 0];
    }
    const ref = this.last ?? this.anchor;
    if (!ref) views.sort((a, b) => b[1] - a[1]);
    const held = this.last !== null;
    let best: View = held ? [this.last!.m, 0, this.flipped] : views[0];
    let won: number | string = held ? 'held' : 0;
    for (let i = 0; i < views.length; i++) {
      const v = views[i];
      if (!ref || v === best) continue;
      const cmp = this.compare(gray, v[0], best[0], ref);
      const holding = held ? best[0] === this.last!.m : false;
      const replace = cmp === null ? holding : cmp[0] > cmp[1];
      d.votes.push([i, won, ...(cmp ?? [null, null]), replace]);
      if (replace) { best = v; won = i; }
    }
    d.won = won;
    const [m, nInl, flipped] = best;
    this.flipped = flipped;
    if (this.last && m === this.last.m) return [Status.STATIC, m, 0];
    const frame = { m, gray };
    d.keyframe = this.isKeyframe(frame);
    if (d.keyframe) await this.addKeyframe(frame);
    this.last = this.anchor = frame;
    return [Status.MATCHED, m, nInl];
  }

  private describe(m: M3, nInl: number): Record<string, unknown> {
    const s = geo.similarity(m);
    const [cx, cy] = geo.apply(m, this.aw / this.r / 2, this.ah / this.r / 2);
    return { inliers: nInl, flip: s.flipped, scale: +s.scale.toFixed(4), angle: +s.angle.toFixed(2), centre: [+cx.toFixed(1), +cy.toFixed(1)] };
  }

  private isKeyframe(frame: Frame): boolean {
    if (!this.lastKey) return true;
    const [ds, shift] = geo.viewChange(this.lastKey.m, frame.m);
    const s = geo.similarity(frame.m);
    if (ds > 0.05 || shift > 20 * s.scale) return true;
    if (s.flipped !== geo.similarity(this.lastKey.m).flipped) return true;
    let changed = 0;
    const a = frame.gray, b = this.lastKey.gray;
    for (let i = 0; i < a.length; i++) if (this.mask[i] && Math.abs(a[i] - b[i]) > 24) changed++;
    return changed / this.maskCount > 0.02;
  }
}

export interface RegisterProgress { n: number; done: number; total: number; status: Status; inliers: number }

/**
 * Register every sample: backward from the reference frame, then forward. `frames(indices)` yields the
 * analysis frames of sorted sample indices.
 */
export async function register(cfg: Config, count: number, fullWidth: number, fullHeight: number, sift: Sift, matcher: Matcher,
  frames: (indices: number[]) => AsyncGenerator<[number, Uint8Array]>,
  progress?: (p: RegisterProgress) => void, decisions?: (n: number, d: Decision & { status: string }) => void,
  stop?: () => boolean): Promise<Track> {
  const track = Track.empty(count, cfg.sample_fps);
  const [mRef] = referenceTransform(cfg);
  const spans = sampleSpans(cfg, count);
  const ref = pyRound(cfg.reference.time * cfg.sample_fps);
  if (!spans.some(([a, b]) => a <= ref && ref < b)) throw new Error('the target frame must lie between start and end, outside the cuts');
  const total = spans.reduce((s, [a, b]) => s + b - a, 0);
  let refGray: Uint8Array | null = null;
  for await (const [, g] of frames([ref])) refGray = g;
  if (!refGray) throw new Error('could not decode the target frame');
  const reg = new Registrar(cfg, fullWidth, fullHeight, sift, matcher);
  track.status[ref] = Status.MATCHED;
  track.set(ref, mRef);
  let done = 0;

  const run = async (chunks: AsyncGenerator<[number, Uint8Array]>) => {
    reg.random = rng(ref);
    await reg.seed(refGray!, mRef);
    for await (const [n, gray] of chunks) {
      if (stop?.()) throw new Error('stopped');
      reg.random = rng(n);
      const [status, m, nInl] = await reg.step(gray);
      if (decisions && reg.decision) decisions(n, { status: Status[status].toLowerCase(), ...reg.decision });
      track.status[n] = status;
      track.inliers[n] = nInl;
      if (m) track.set(n, m);
      progress?.({ n, done: ++done, total, status, inliers: nInl });
    }
  };

  // backward, decoding forward in chunks of READ_AHEAD and reversing them; a cut is passed over like a
  // stretch where the view was lost
  async function* backward(): AsyncGenerator<[number, Uint8Array]> {
    for (const [a, b] of [...spans].reverse()) {
      if (a >= ref) continue;
      let hi = Math.min(b, ref);
      while (hi > a) {
        const lo = Math.max(a, hi - READ_AHEAD);
        const chunk: [number, Uint8Array][] = [];
        for await (const f of frames(range(lo, hi))) chunk.push(f);
        yield* chunk.reverse();
        hi = lo;
      }
    }
  }
  async function* forward(): AsyncGenerator<[number, Uint8Array]> {
    for (const [a, b] of spans) if (b > ref + 1) yield* frames(range(Math.max(a, ref + 1), b));
  }
  await run(backward());
  await run(forward());
  return track;
}

export function range(a: number, b: number): number[] {
  return Array.from({ length: Math.max(0, b - a) }, (_, i) => a + i);
}
