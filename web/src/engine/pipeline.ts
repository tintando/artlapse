/**
 * The stages, as artlapse's pipeline.py runs them: register (the track), composite (canvas.mp4, its
 * snapshot index and the final canvas) and render (timelapse.mp4). Files are kept in the browser's
 * origin-private file system when it has one, else in memory.
 */
import { BlobSource, BufferTarget, Input, ALL_FORMATS, Mp4OutputFormat, Output, Quality, StreamTarget, VideoSample, VideoSampleSink, VideoSampleSource } from 'mediabunny';
import { Compositor, GpuCanvas, validMask } from './composite';
import { Config } from './config';
import { Grayer, Video, grayFrames } from './frames';
import { Gpu } from './gpu';
import { pyRound } from './imgops';
import { Matcher } from './match';
import { Decision, Status, Track, keptSamples, referenceTransform, register, sampleRange, screenMask } from './register';
import { Sift } from './sift';

export type Stage = 'register' | 'composite' | 'render';
export interface Progress { stage: Stage; done: number; total: number; detail: string }
export type Report = (p: Progress) => void;

export interface Snapshots { index: number[]; changed: number[] }

/** A file being written: to the origin-private file system, or to memory where there is none. */
class Sink {
  private constructor(readonly target: StreamTarget | BufferTarget, private handle: FileSystemFileHandle | null) {}

  /** A new file; the name gets a unique suffix, so an earlier run's file is not written over while in use. */
  static async create(name: string): Promise<Sink> {
    try {
      const dir = await navigator.storage.getDirectory();
      const handle = await dir.getFileHandle(`${Date.now()}-${Math.random().toString(36).slice(2, 8)}-${name}`, { create: true });
      const writable = await handle.createWritable();
      return new Sink(new StreamTarget(writable as unknown as WritableStream, { chunked: true }), handle);
    } catch {
      return new Sink(new BufferTarget(), null);
    }
  }

  async file(type: string): Promise<Blob> {
    if (this.handle) return this.handle.getFile();
    return new Blob([(this.target as BufferTarget).buffer!], { type });
  }
}

/** Remove the files earlier sessions left in the origin-private file system: those over a day old, so a run
 * in another tab keeps its own. */
export async function clearFiles(): Promise<void> {
  try {
    const dir = await navigator.storage.getDirectory();
    for await (const name of (dir as unknown as { keys(): AsyncIterable<string> }).keys()) {
      if (Date.now() - (parseInt(name, 10) || 0) > 86400e3) await dir.removeEntry(name).catch(() => {});
    }
  } catch { /* nothing kept there */ }
}

export async function runRegister(cfg: Config, video: Video, gpu: Gpu, report: Report, stop: () => boolean,
  decisions?: (n: number, d: Decision & { status: string }) => void): Promise<Track> {
  const { width, height } = video.info;
  const aw = cfg.register.analysis_width, ah = pyRound((height * aw) / width / 2) * 2;
  const sift = new Sift(gpu, aw, ah, cfg.register.max_features);
  const grayer = new Grayer(gpu, aw, ah);
  const counts = [0, 0, 0, 0];
  return register(cfg, video.count(cfg.sample_fps), width, height, sift, new Matcher(gpu),
    idx => grayFrames(video, grayer, idx, cfg.sample_fps),
    p => {
      counts[p.status]++;
      report({ stage: 'register', done: p.done, total: p.total, detail: `matched ${counts[Status.MATCHED]}, unmoved ${counts[Status.STATIC]}, lost ${counts[Status.LOST]}` });
    }, decisions, stop);
}

export interface Composite { canvas: Blob; snapshots: Snapshots; final: Blob; size: [number, number] }

export async function runComposite(cfg: Config, track: Track, video: Video, gpu: Gpu, report: Report, stop: () => boolean): Promise<Composite> {
  const { width, height } = video.info;
  const [, size] = referenceTransform(cfg);
  const valid = validMask(screenMask(cfg, width, height, 1.0), width, height, cfg.accumulate.erode_px);
  const canvas = new GpuCanvas(gpu, size, cfg, width, height, valid);
  const comp = new Compositor(canvas);
  const [lo, hi] = sampleRange(cfg, video.count(cfg.sample_fps));
  const placed: number[] = [];
  for (let n = lo; n < Math.min(hi, track.length); n++) if (track.ok(n)) placed.push(n);

  const sink = await Sink.create('canvas.mp4');
  const output = new Output({ format: new Mp4OutputFormat(), target: sink.target });
  // near-lossless, as artlapse's canvas.mp4 (crf 8)
  const source = new VideoSampleSource({ codec: 'avc', quality: new Quality({ quantizer: 8, bitrate: 60e6 }), keyFrameInterval: 10, latencyMode: 'quality' });
  output.addVideoTrack(source, { frameRate: 30 });
  await output.start();
  const snapshots: Snapshots = { index: [], changed: [] };
  const write = async (n: number, changed: number) => {
    const t = snapshots.index.length / 30;
    // closing the sample closes its frame, not the bitmap the frame was made from: that one is closed
    // here, or each snapshot keeps a canvas-sized image in GPU memory until garbage collection
    const bitmap = canvas.image();
    const sample = new VideoSample(bitmap, { timestamp: t, duration: 1 / 30 });
    await source.add(sample);
    sample.close();
    bitmap.close();
    snapshots.index.push(n);
    snapshots.changed.push(changed);
  };
  let pending = 0, done = 0;
  for await (const [n, frame] of video.frames(placed, cfg.sample_fps)) {
    if (stop()) { frame.close(); throw new Error('stopped'); }
    let res;
    try {
      res = await comp.add(frame, track.at(n));
    } finally {
      frame.close();
    }
    pending += res.changed;
    if (pending >= cfg.accumulate.min_changed_px) {
      await write(n, pending);
      pending = 0;
    }
    report({ stage: 'composite', done: ++done, total: placed.length, detail: `${snapshots.index.length} snapshots` });
  }
  await write(hi - 1, pending);
  source.close();
  await output.finalize();
  const [w, h] = size;
  const rgba = await canvas.pixels();
  const c2d = new OffscreenCanvas(w, h);
  c2d.getContext('2d')!.putImageData(new ImageData(rgba as unknown as Uint8ClampedArray<ArrayBuffer>, w, h), 0, 0);
  const final = await c2d.convertToBlob({ type: 'image/png' });
  return { canvas: await sink.file('video/mp4'), snapshots, final, size };
}

/** Which snapshot to show at each of n_out output frames (monotonic, repeats allowed): progress mixes
 * painting activity (changed pixels) with stream time. */
export function pickFrames(index: number[], changed: number[], nOut: number, activityWeight: number): number[] {
  const n = index.length;
  const act: number[] = [];
  let s = 0;
  for (const c of changed) act.push((s += c));
  const lin = (i: number) => (n > 1 ? i / (n - 1) : 0);
  const a = s > 0 ? act.map(v => v / s) : act.map((_, i) => lin(i));
  const t0 = index[0], t1 = index[n - 1];
  const t = t1 > t0 ? index.map(v => (v - t0) / (t1 - t0)) : index.map((_, i) => lin(i));
  const u = a.map((v, i) => activityWeight * v + (1 - activityWeight) * t[i]);
  const out: number[] = [];
  for (let k = 0; k < nOut; k++) {
    const target = nOut > 1 ? k / (nOut - 1) : 0;
    let lo = 0, hi = n; // searchsorted left
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (u[mid] < target) lo = mid + 1; else hi = mid;
    }
    out.push(Math.min(Math.max(lo, 0), n - 1));
  }
  return out;
}

/** The snapshot shown at each frame of the timelapse, the hold at the end included. */
export function renderPicks(cfg: Config, snaps: Snapshots): number[] {
  const rc = cfg.render;
  const picks = pickFrames(keptSamples(cfg, snaps.index), snaps.changed, Math.max(2, pyRound(rc.duration * rc.fps)), rc.activity_weight);
  return picks.concat(Array(pyRound(rc.hold_end * rc.fps)).fill(snaps.index.length - 1));
}

export async function runRender(cfg: Config, comp: Composite, report: Report, stop: () => boolean): Promise<Blob> {
  const rc = cfg.render;
  const picks = renderPicks(cfg, comp.snapshots);
  const input = new Input({ source: new BlobSource(comp.canvas), formats: ALL_FORMATS });
  const track = (await input.getPrimaryVideoTrack())!;
  const sink = await Sink.create('timelapse.mp4');
  const output = new Output({ format: new Mp4OutputFormat({ fastStart: 'in-memory' }), target: sink.target });
  const source = new VideoSampleSource({ codec: 'avc', quality: new Quality({ quantizer: rc.crf, bitrate: 20e6 }), keyFrameInterval: 2 });
  output.addVideoTrack(source, { frameRate: rc.fps });
  await output.start();
  const samples = new VideoSampleSink(track).samples();
  let i = -1;
  let cur: VideoSample | null = null;
  for (let k = 0; k < picks.length; k++) {
    if (stop()) throw new Error('stopped');
    while (i < picks[k]) {
      const next = await samples.next();
      if (next.done) break;
      cur?.close();
      cur = next.value;
      i++;
    }
    const frame = cur!.toVideoFrame();
    const out = new VideoSample(frame, { timestamp: k / rc.fps, duration: 1 / rc.fps });
    await source.add(out);
    out.close();
    frame.close();
    report({ stage: 'render', done: k + 1, total: picks.length, detail: '' });
  }
  cur?.close();
  await samples.return(undefined);
  source.close();
  await output.finalize();
  return sink.file('video/mp4');
}
