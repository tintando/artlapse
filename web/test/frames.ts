// Save the analysis frames of some samples as raw bytes: test/frames.html?video=<path>&n=470,471&w=960&out=<name>
import { UrlSource } from 'mediabunny';
import { Grayer, Video, grayFrames } from '../src/engine/frames';
import { Gpu } from '../src/engine/gpu';

declare global { interface Window { __done: unknown } }

async function main() {
  const q = new URLSearchParams(location.search);
  const video = await Video.open(new UrlSource('/__file?path=' + encodeURIComponent(q.get('video')!)));
  if (q.has('start')) (video.info as { start: number }).start = +q.get('start')!;
  const aw = +(q.get('w') ?? 960), ah = Math.round(video.info.height * aw / video.info.width / 2) * 2;
  const gpu = await Gpu.create();
  const idx = q.get('n')!.split(',').map(Number);
  const parts: Uint8Array[] = [];
  for await (const [, g] of grayFrames(video, new Grayer(gpu, aw, ah), idx, +(q.get('fps') ?? 2))) parts.push(g);
  await fetch('/__save?name=' + q.get('out'), { method: 'POST', body: new Blob(parts as BlobPart[]) });
  window.__done = { info: video.info, w: aw, h: ah, n: parts.length };
}
main().catch(e => { console.error(e.stack); window.__done = { error: String(e) }; });
