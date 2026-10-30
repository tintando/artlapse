// SIFT on an image from disk, in both flip states, saved for test/compare_sift.py:
// test/sift.html?img=<png path>&out=<name>[&cpu=1] (cpu: the processor fallback's SIFT, sift-cpu.ts)
import { Gpu } from '../src/engine/gpu';
import { SiftCpu } from '../src/engine/sift-cpu';
import { Detector, Sift } from '../src/engine/sift';

declare global { interface Window { __done: unknown } }

async function main() {
  const q = new URLSearchParams(location.search);
  const blob = await (await fetch('/__file?path=' + encodeURIComponent(q.get('img')!))).blob();
  const bmp = await createImageBitmap(blob);
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext('2d')!;
  ctx.drawImage(bmp, 0, 0);
  const rgba = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
  const gray = new Uint8Array(bmp.width * bmp.height);
  for (let i = 0; i < gray.length; i++) gray[i] = rgba[4 * i];
  const n = +(q.get('n') ?? 1500);
  const sift: Detector = q.get('cpu') ? new SiftCpu(bmp.width, bmp.height, n) : new Sift(await Gpu.create(), bmp.width, bmp.height, n);
  const mask = new Uint8Array(gray.length).fill(255);
  let feats = await sift.detect(gray, [false, true], [mask, mask]);
  const times = [];
  for (let i = 0; i < 5; i++) {
    const t0 = performance.now();
    feats = await sift.detect(gray, [false, true], [mask, mask]);
    times.push(performance.now() - t0);
  }
  const out = feats.map(f => ({ pts: Array.from(f.pts), desc: Array.from(f.desc) }));
  await fetch('/__save?name=' + q.get('out'), { method: 'POST', body: JSON.stringify(out) });
  window.__done = { counts: feats.map(f => f.pts.length / 2), ms: times.map(t => +t.toFixed(1)) };
}
main().catch(e => { console.error(e.stack); window.__done = { error: String(e) }; });
