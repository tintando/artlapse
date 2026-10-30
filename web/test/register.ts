// Register a video with a config from disk and save the track and decision log:
// test/register.html?video=<path>&config=<config.yaml path>&out=<name>[&from=<sample>&to=<sample>]
// from/to limit the run like start/end (a reference frame outside is moved to `from`).
import { UrlSource } from 'mediabunny';
import { parse } from '../src/engine/config';
import { Grayer, Video, grayFrames } from '../src/engine/frames';
import { Gpu } from '../src/engine/gpu';
import { Matcher } from '../src/engine/match';
import { register, Status } from '../src/engine/register';
import { Sift } from '../src/engine/sift';

declare global { interface Window { __done: unknown } }
const file = (p: string) => '/__file?path=' + encodeURIComponent(p);

async function main() {
  const q = new URLSearchParams(location.search);
  const cfg = parse(await (await fetch(file(q.get('config')!))).text());
  if (q.get('from')) cfg.start = +q.get('from')! / cfg.sample_fps;
  if (q.get('to')) cfg.end = +q.get('to')! / cfg.sample_fps;
  const video = await Video.open(new UrlSource(file(q.get('video')!)));
  const { width, height } = video.info;
  const count = video.count(cfg.sample_fps);
  const gpu = await Gpu.create();
  const aw = cfg.register.analysis_width, ah = Math.round(height * aw / width / 2) * 2;
  const sift = new Sift(gpu, aw, ah, cfg.register.max_features);
  const grayer = new Grayer(gpu, aw, ah);
  const decisions: unknown[] = [];
  const counts = [0, 0, 0, 0];
  const t0 = performance.now();
  let last = t0;
  const track = await register(cfg, count, width, height, sift, new Matcher(gpu),
    idx => grayFrames(video, grayer, idx, cfg.sample_fps),
    p => {
      counts[p.status]++;
      const now = performance.now();
      if (now - last > 10000) {
        last = now;
        console.log(`register ${p.done}/${p.total} at sample ${p.n} (${(p.done / ((now - t0) / 1000)).toFixed(1)}/s) matched=${counts[1]} static=${counts[2]} lost=${counts[3]}`);
      }
    },
    (n, d) => decisions.push({ n, ...d }));
  const seconds = (performance.now() - t0) / 1000;
  const out = q.get('out')!;
  await fetch('/__save?name=' + out + '.track.json', { method: 'POST', body: JSON.stringify(track) });
  await fetch('/__save?name=' + out + '.decisions.json', { method: 'POST', body: JSON.stringify(decisions) });
  window.__done = { seconds: +seconds.toFixed(1), matched: counts[Status.MATCHED], static: counts[Status.STATIC], lost: counts[Status.LOST], count };
}
main().catch(e => { console.error(e.stack); window.__done = { error: String(e) }; });
