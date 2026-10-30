/**
 * SIFT on the processor, for browsers without WebGPU: sift.ts's passes in the same order, each one shared out
 * by rows (or keypoints) across the team's workers (team.ts, cpu-kernels.ts). It finds the keypoints and
 * descriptors the GPU does, and OpenCV does, up to float rounding. Histograms are summed in floats, as
 * OpenCV's are; the GPU sums them in fixed point.
 */
import { Level } from './cpu-kernels';
import {
  BASE_SIGMA, CAP, Detector, Features, LAYERS, PER_OCTAVE, gaussKernel, levelSigmas, octaves, selectKeypoints,
} from './sift';
import { Team, theTeam } from './team';

export { reflect101, rne } from './cpu-kernels';

export class SiftCpu implements Detector {
  private readonly levels: Level[][]; // [octave][level] in pyr
  private readonly dogs: number[][]; // [octave][layer] offsets in dog
  private readonly pyr: Float32Array;
  private readonly dog: Float32Array;
  private readonly tmp: Float32Array;
  private readonly gray: Uint8Array;
  private readonly kernels: Float32Array[];

  constructor(readonly w: number, readonly h: number, readonly maxFeatures: number, private team: Team = theTeam()) {
    const [octs, size] = octaves(w, h);
    this.levels = octs.map(o => o.off.map(off => ({ off, w: o.w, h: o.h })));
    let d = 0;
    this.dogs = octs.map(o => Array.from({ length: PER_OCTAVE - 1 }, () => { const at = d; d += o.w * o.h; return at; }));
    this.pyr = team.alloc(Float32Array, size);
    this.dog = team.alloc(Float32Array, d);
    this.tmp = team.alloc(Float32Array, 4 * w * h);
    this.gray = team.alloc(Uint8Array, w * h);
    this.kernels = [BASE_SIGMA, ...levelSigmas().slice(1)].map(s => Float32Array.from(gaussKernel(s)));
  }

  private async blur(src: Float32Array, sOff: number, dOff: number, lv: Level, k: Float32Array) {
    const { w, h } = lv, t = this.team;
    await t.run('blurH', { src, sOff, dst: this.tmp, dOff: 0, w, k }, h, w * h);
    await t.run('blurV', { src: this.tmp, sOff: 0, dst: this.pyr, dOff, w, h, k }, h, w * h);
  }

  private async pyramid(flip: boolean) {
    const t = this.team, W = 2 * this.w, H = 2 * this.h;
    const base = this.dog.subarray(0, W * H); // the doubled image, before the DoGs are made
    await t.run('upsample', { gray: this.gray, w: this.w, h: this.h, flip, dst: base }, H, W * H);
    for (let o = 0; o < this.levels.length; o++) {
      const lv = this.levels[o];
      if (o === 0) await this.blur(this.dog, 0, lv[0].off, lv[0], this.kernels[0]);
      else await t.run('downsample', { pyr: this.pyr, src: this.levels[o - 1][LAYERS], dst: lv[0] }, lv[0].h, lv[0].w * lv[0].h);
      for (let l = 1; l < PER_OCTAVE; l++) await this.blur(this.pyr, lv[l - 1].off, lv[l].off, lv[l], this.kernels[l]);
    }
    await Promise.all(this.levels.map((lv, o) =>
      t.run('dog', { pyr: this.pyr, levels: lv, dog: this.dog, dogs: this.dogs[o] }, lv[0].h, lv[0].w * lv[0].h * (PER_OCTAVE - 1))));
  }

  private async detectOne(flip: boolean, mask: Uint8Array): Promise<Features> {
    const t = this.team;
    await this.pyramid(flip);
    const found = await Promise.all(this.levels.map((lv, o) =>
      t.run('extrema', { dog: this.dog, dogs: this.dogs[o], w: lv[0].w, h: lv[0].h, o }, lv[0].h, lv[0].w * lv[0].h * LAYERS)));
    const all = found.flat(2);
    const cands = Float64Array.from(all.slice(0, CAP * 8));
    const nc = cands.length / 8;
    const oriented = (await t.run('orient', { pyr: this.pyr, levels: this.levels, cands }, nc, nc * 1000)).flat();
    // as Kp records (sift.ts) for selectKeypoints
    const n = Math.min(oriented.length / 6, CAP);
    const kps = new ArrayBuffer(Math.max(n, 1) * 32), f = new Float32Array(kps), u = new Uint32Array(kps);
    for (let i = 0; i < n; i++) {
      f.set(oriented.slice(i * 6, i * 6 + 5), i * 8);
      u[i * 8 + 5] = oriented[i * 6 + 5];
    }
    const chosen = selectKeypoints(n, kps, mask, this.w, this.h, this.maxFeatures);
    const out = t.alloc(Uint8Array, chosen.n * 128);
    await t.run('describe', { pyr: this.pyr, levels: this.levels, input: chosen.input, out }, chosen.n, chosen.n * 1000);
    return { pts: chosen.pts, desc: out.slice() };
  }

  /** The flip states one after the other, each shared across the team. */
  async detect(gray: Uint8Array, flips: boolean[], masks: Uint8Array[]): Promise<Features[]> {
    this.gray.set(gray);
    const out: Features[] = [];
    for (let i = 0; i < flips.length; i++) out.push(await this.detectOne(flips[i], masks[i]));
    return out;
  }
}
