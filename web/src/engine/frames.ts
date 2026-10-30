/**
 * Frame sampling with WebCodecs (through mediabunny, which demuxes MP4, MOV, MKV, WebM and MPEG-TS).
 *
 * Sample n is the frame on screen at n / fps from the first frame, as in artlapse: the last frame whose
 * timestamp is at or before that time. Register's analysis frames are made on the GPU: the frame
 * shrunk to `analysis_width` by area averaging and turned to grey (full-range luma), as ffmpeg's
 * `scale=flags=area` and `gray` give them.
 */
import { ALL_FORMATS, BlobSource, EncodedPacketSink, Input, InputVideoTrack, Source, VideoSampleSink } from 'mediabunny';
import { Gpu } from './gpu';

export interface VideoInfo { width: number; height: number; duration: number; start: number }

export class Video {
  private constructor(readonly input: Input, readonly track: InputVideoTrack, readonly info: VideoInfo) {}

  static async open(file: Blob | Source): Promise<Video> {
    const input = new Input({ source: file instanceof Blob ? new BlobSource(file) : file, formats: ALL_FORMATS });
    const track = await input.getPrimaryVideoTrack();
    if (!track) throw new Error('the file has no video track');
    if (!(await track.canDecode())) throw new Error(`this browser can't decode the video's codec (${await track.getCodecParameterString() ?? track.codec})`);
    const start = await origin(track);
    const duration = (await input.computeDuration()) - start;
    return new Video(input, track, { width: track.displayWidth, height: track.displayHeight, duration, start });
  }

  /** How many samples the video has at `fps`. */
  count(fps: number): number {
    return Math.floor(this.info.duration * fps);
  }

  /** The frames of sorted sample indices, as VideoFrames the caller closes. A sample before the first
   * frame gets the first frame. */
  async *frames(indices: number[], fps: number): AsyncGenerator<[number, VideoFrame]> {
    const sink = new VideoSampleSink(this.track);
    // a frame shown at exactly the sample time may carry a timestamp a rounding error after it
    const times = indices.map(n => this.info.start + n / fps + 1e-3);
    let i = 0;
    for await (const s of sink.samplesAtTimestamps(times)) {
      const n = indices[i++];
      const sample = s ?? await sink.getSample(this.info.start);
      if (!sample) continue;
      const frame = sample.toVideoFrame();
      sample.close();
      yield [n, frame];
    }
  }
}

/** Where sample 0 is on the track's timeline. ffmpeg's: the first frame; when the first frames carry
 * negative timestamps (pre-roll before an edit list's start), the first frame at or after 0. */
async function origin(track: InputVideoTrack): Promise<number> {
  const first = await track.getFirstTimestamp();
  if (first >= 0) return first;
  const packets = new EncodedPacketSink(track);
  let best = Infinity;
  // decode order differs from presentation order by a few frames at most
  for await (const p of packets.packets(undefined, undefined, { metadataOnly: true })) {
    if (p.timestamp >= 0) best = Math.min(best, p.timestamp);
    if (p.timestamp > 1) break;
  }
  return Number.isFinite(best) ? best : 0;
}

const LUMA: Record<string, [number, number, number]> = {
  bt709: [0.2126, 0.7152, 0.0722],
  bt2020: [0.2627, 0.678, 0.0593],
  bt470bg: [0.299, 0.587, 0.114],
  smpte170m: [0.299, 0.587, 0.114],
};

export function luma(frame: VideoFrame): [number, number, number] {
  return LUMA[frame.colorSpace?.matrix ?? ''] ?? (frame.displayHeight > 576 ? LUMA.bt709 : LUMA.bt470bg);
}

const GRAY = /* wgsl */ `
struct P { w: u32, h: u32, sw: u32, sh: u32, k: vec4f }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var frame: texture_external;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
fn y(x: u32, yy: u32) -> f32 { return dot(textureLoad(frame, vec2u(x, yy)).rgb, p.k.xyz); }
fn area(i: u32) -> f32 {
  // the source area of output pixel (ox, oy), with the fractional coverage of its edge pixels
  let ox = i % p.w;
  let oy = i / p.w;
  let fx = f32(p.sw) / f32(p.w);
  let fy = f32(p.sh) / f32(p.h);
  let x0 = f32(ox) * fx;
  let x1 = x0 + fx;
  let y0 = f32(oy) * fy;
  let y1 = y0 + fy;
  var s = 0.0;
  for (var sy = u32(floor(y0)); f32(sy) < y1 && sy < p.sh; sy++) {
    let wy = min(y1, f32(sy + 1u)) - max(y0, f32(sy));
    for (var sx = u32(floor(x0)); f32(sx) < x1 && sx < p.sw; sx++) {
      let wx = min(x1, f32(sx + 1u)) - max(x0, f32(sx));
      s += wx * wy * y(sx, sy);
    }
  }
  return s / (fx * fy);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let word = g.x + g.y * 65535u * 64u;
  if (word * 4u >= p.w * p.h) { return; }
  var v = 0u;
  for (var k = 0u; k < 4u; k++) {
    let i = word * 4u + k;
    if (i < p.w * p.h) { v |= u32(clamp(round(area(i) * 255.0), 0.0, 255.0)) << (8u * k); }
  }
  out[word] = v;
}
`;

/** Grey analysis frames on the GPU. */
export class Grayer {
  private out: GPUBuffer;
  private params: GPUBuffer;

  constructor(private gpu: Gpu, readonly w: number, readonly h: number) {
    this.out = gpu.buffer(w * h);
    this.params = gpu.uniform(new ArrayBuffer(32));
  }

  async gray(frame: VideoFrame): Promise<Uint8Array> {
    const g = this.gpu;
    const b = new ArrayBuffer(32);
    new Uint32Array(b, 0, 4).set([this.w, this.h, frame.displayWidth, frame.displayHeight]);
    new Float32Array(b, 16, 3).set(luma(frame));
    g.device.queue.writeBuffer(this.params, 0, b);
    const tex = g.device.importExternalTexture({ source: frame });
    const enc = g.device.createCommandEncoder();
    const words = Math.ceil((this.w * this.h) / 4);
    const gx = Math.min(Math.ceil(words / 64), 65535), gy = Math.ceil(words / 64 / 65535);
    g.dispatch(enc, g.pipeline(GRAY, 'main'), [this.params, tex, this.out], gx, gy);
    const [raw] = await g.read(enc, [[this.out, 0, this.w * this.h]]);
    return new Uint8Array(raw);
  }
}

/** Analysis frames for sorted sample indices, decoding and converting a few ahead of the consumer. */
export async function* grayFrames(video: Video, grayer: Grayer, indices: number[], fps: number, depth = 4): AsyncGenerator<[number, Uint8Array]> {
  const pending: Promise<[number, Uint8Array]>[] = [];
  for await (const [n, frame] of video.frames(indices, fps)) {
    pending.push(grayer.gray(frame).then(g => { frame.close(); return [n, g] as [number, Uint8Array]; }));
    if (pending.length >= depth) yield await pending.shift()!;
  }
  while (pending.length) yield await pending.shift()!;
}
