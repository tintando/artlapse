/**
 * Compositing on the processor, for browsers without WebGPU: composite.ts's passes (warp, blurs, ink, the
 * rules) in the same order over the canvas box a frame covers, each shared out by rows across the team's
 * workers (team.ts, cpu-kernels.ts). Pixels are RGBA bytes; results are rounded to 8 bits where the GPU's are.
 */
import { Canvas, FrameResult, INK_GROW, INK_SIGMA, MARGIN } from './composite';
import { Config } from './config';
import { FramePixels } from './frames';
import * as geo from './geometry';
import { M3 } from './geometry';
import { pyRound } from './imgops';
import { gaussKernel } from './sift';
import { Team, theTeam } from './team';

export class CpuCanvas extends Canvas {
  private color: Uint8ClampedArray; // RGBA
  private quality: Float32Array;
  private pending: Uint8Array; // RGB
  private count: Uint16Array;
  private valid: Uint8Array;
  private frame: Uint8ClampedArray;
  private frameBlur: Uint8ClampedArray | null = null;
  private fex: Uint8ClampedArray;
  private fb: Uint8ClampedArray;
  private cb: Uint8ClampedArray;
  private vex: Uint8Array;
  private ink0: Uint8Array;
  private ink1: Uint8Array;
  private tmp: Float32Array; // 3 floats a pixel for blurs, 4 for ink's first pass
  private tb: Float32Array;
  private px = new FramePixels();
  private out: OffscreenCanvasRenderingContext2D | null = null;

  constructor(size: [number, number], cfg: Config, sw: number, sh: number, valid: Uint8Array, private team: Team = theTeam()) {
    super(size, cfg, sw, sh, valid);
    const [w, h] = size, n = w * h, t = team;
    this.color = t.alloc(Uint8ClampedArray, n * 4);
    const [bb, bg, br] = cfg.canvas.background;
    for (let i = 0; i < n; i++) this.color.set([br, bg, bb, 255], i * 4);
    this.quality = t.alloc(Float32Array, n);
    this.pending = t.alloc(Uint8Array, n * 3);
    this.count = t.alloc(Uint16Array, n);
    this.valid = t.alloc(Uint8Array, sw * sh);
    this.valid.set(valid);
    this.frame = t.alloc(Uint8ClampedArray, sw * sh * 4);
    // the box a frame updates is at most the canvas
    this.fex = t.alloc(Uint8ClampedArray, n * 4);
    this.fb = t.alloc(Uint8ClampedArray, n * 4);
    this.cb = t.alloc(Uint8ClampedArray, n * 4);
    this.vex = t.alloc(Uint8Array, n);
    this.ink0 = t.alloc(Uint8Array, n);
    this.ink1 = t.alloc(Uint8Array, n);
    this.tmp = t.alloc(Float32Array, Math.max(n * 4, sw * sh * 3));
    this.tb = t.alloc(Float32Array, n * 3);
  }

  /** Blur an RGB region (w x h at x0, y0 of an RGBA image `stride` wide) into `dst` (at dx0, dy0 of an RGBA
   * image `dstride` wide): along y into floats, then along x, rounded. */
  private async blur(src: Uint8ClampedArray, stride: number, x0: number, y0: number, w: number, h: number,
    dst: Uint8ClampedArray, dstride: number, dx0: number, dy0: number, sigma: number) {
    const k = Float32Array.from(gaussKernel(sigma, true)), size = w * h * k.length;
    await this.team.run('blur8Y', { src, stride, x0, y0, w, h, tmp: this.tmp, k }, h, size);
    await this.team.run('blur8X', { tmp: this.tmp, dst, dstride, dx0, dy0, w, k }, h, size);
  }

  async add(frame: VideoFrame, m: M3): Promise<FrameResult> {
    const box = this.visibleBox(m);
    if (!box) return { changed: 0, processed: 0 };
    const cfg = this.cfg, t = this.team;
    const [w, h] = this.size;
    const { sw, sh } = this;
    const scale = geo.similarity(m).scale;
    this.frame.set(this.px.rgba(frame));
    let src = this.frame;
    if (scale < 0.6) { // zoomed in further than the canvas resolution: low-pass before shrinking
      this.frameBlur ??= t.alloc(Uint8ClampedArray, sw * sh * 4);
      await this.blur(this.frame, sw, 0, 0, sw, sh, this.frameBlur, sw, 0, 0, 0.4 / scale);
      src = this.frameBlur;
    }
    const [x0, y0, x1, y1] = box;
    const ex0 = Math.max(x0 - MARGIN, 0), ey0 = Math.max(y0 - MARGIN, 0), ex1 = Math.min(x1 + MARGIN, w), ey1 = Math.min(y1 + MARGIN, h);
    const ew = ex1 - ex0, eh = ey1 - ey0, en = ew * eh;
    const inv = geo.inv(geo.mul(geo.scaleTranslate(1, -ex0, -ey0), m)); // box pixel -> screen pixel
    const { fex, vex, fb, cb } = this;
    await t.run('warp', { src, sw, sh, valid: this.valid, m: Array.from(inv.slice(0, 6)), ew, fex, vex }, eh, en);
    const q = Math.min(1 / scale, 4.0);
    const sigma = Math.max(0.8, 0.5 * scale);
    await this.blur(fex, ew, 0, 0, ew, eh, fb, ew, 0, 0, sigma);
    await this.blur(this.color, w, ex0, ey0, ew, eh, cb, ew, 0, 0, sigma);
    // ink only matters where the canvas holds pixels sharper than this view; elsewhere the rules ignore it
    const hasInk = cfg.ink_threshold > 0;
    if (hasInk) {
      const k = Float32Array.from(gaussKernel(INK_SIGMA * scale, true)), size = en * k.length;
      await t.run('ink1', { fex, vex, color: this.color, cw: w, ex0, ey0, ew, eh, k, ta: this.tmp, tb: this.tb }, eh, size);
      await t.run('ink2', { ta: this.tmp, tb: this.tb, vex, ew, k, thr: cfg.ink_threshold, out: this.ink0 }, eh, size);
      // opening by a (2a+1) square, then growing back by g: max filters compose, so min a, max a + g
      const a = Math.max(1, pyRound(scale)), grow = Math.ceil(INK_GROW * scale);
      const morph = (src: Uint8Array, dst: Uint8Array, r: number, axis: number, opMax: boolean, outside: number) =>
        t.run('morph', { src, dst, ew, eh, r, axis, opMax, outside }, eh, en * (2 * r + 1));
      await morph(this.ink0, this.ink1, a, 0, false, 1);
      await morph(this.ink1, this.ink0, a, 1, false, 1);
      await morph(this.ink0, this.ink1, a + grow, 0, true, 0);
      await morph(this.ink1, this.ink0, a + grow, 1, true, 0);
    }
    const iw = x1 - x0, ih = y1 - y0;
    const counts = await t.run('rules', {
      fb, cb, fex, vex, ink: this.ink0, hasInk, color: this.color, quality: this.quality, pending: this.pending, count: this.count,
      ew, ix: x0 - ex0, iy: y0 - ey0, iw, cw: w, x0, y0,
      q, qg: q * cfg.refine_gain, faint: cfg.faint_threshold, diff: cfg.diff_threshold, gain: cfg.refine_gain, stable: this.stable,
    }, ih, iw * ih * 8);
    return { changed: counts.reduce((s, c) => s + c[0], 0), processed: counts.reduce((s, c) => s + c[1], 0) };
  }

  image(): ImageBitmap {
    const [w, h] = this.size;
    this.out ??= new OffscreenCanvas(w, h).getContext('2d')!;
    // ImageData takes memory of its own, not shared memory
    this.out.putImageData(new ImageData(this.color.slice() as Uint8ClampedArray<ArrayBuffer>, w, h), 0, 0);
    return this.out.canvas.transferToImageBitmap();
  }

  async pixels(): Promise<Uint8ClampedArray> {
    return this.color.slice();
  }
}
