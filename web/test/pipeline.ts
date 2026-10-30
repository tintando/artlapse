// Composite and render from a saved track, and save final.png, timelapse.mp4 and the snapshot index:
// test/pipeline.html?video=<path>&config=<config.yaml>&track=<web .track.json path>&out=<name>
import { UrlSource } from 'mediabunny';
import { parse } from '../src/engine/config';
import { Video } from '../src/engine/frames';
import { Gpu } from '../src/engine/gpu';
import { runComposite, runRender } from '../src/engine/pipeline';
import { Track } from '../src/engine/register';

declare global { interface Window { __done: unknown } }
const file = (p: string) => '/__file?path=' + encodeURIComponent(p);

async function main() {
  const q = new URLSearchParams(location.search);
  const cfg = parse(await (await fetch(file(q.get('config')!))).text());
  const t = await (await fetch(file(q.get('track')!))).json();
  const track = new Track(t.fps, Uint8Array.from(t.status), Float64Array.from(t.m), Int32Array.from(t.inliers));
  const video = await Video.open(new UrlSource(file(q.get('video')!)));
  const gpu = await Gpu.create();
  let last = 0;
  const report = (p: { stage: string; done: number; total: number; detail: string }) => {
    const now = performance.now();
    if (now - last > 10000) { last = now; console.log(`${p.stage} ${p.done}/${p.total} ${p.detail}`); }
  };
  const t0 = performance.now();
  const comp = await runComposite(cfg, track, video, gpu, report, () => false);
  const t1 = performance.now();
  const tl = await runRender(cfg, comp, report, () => false);
  const t2 = performance.now();
  const out = q.get('out')!;
  await fetch('/__save?name=' + out + '.final.png', { method: 'POST', body: comp.final });
  await fetch('/__save?name=' + out + '.timelapse.mp4', { method: 'POST', body: tl });
  await fetch('/__save?name=' + out + '.canvas.mp4', { method: 'POST', body: comp.canvas });
  await fetch('/__save?name=' + out + '.snapshots.json', { method: 'POST', body: JSON.stringify(comp.snapshots) });
  window.__done = { composite: +((t1 - t0) / 1000).toFixed(1), render: +((t2 - t1) / 1000).toFixed(1), snapshots: comp.snapshots.index.length, canvasMB: +(comp.canvas.size / 1e6).toFixed(1) };
}
main().catch(e => { console.error(e.stack); window.__done = { error: String(e) }; });
