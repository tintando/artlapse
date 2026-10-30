// The page end to end in headless Chrome: load a recording, import a setup, make the timelapse, save it.
// node test/page.mjs <video> <setup.yaml> <out dir> [timeout s]
// QUERY=?cpu makes it on the processor (the fallback for browsers without WebGPU).
// The out dir gets screenshots and the saved files; it must not exist yet.
import { mkdirSync, existsSync, readdirSync, statSync } from 'node:fs';
import puppeteer from 'puppeteer-core';

const [videoPath, setupPath, out, timeout = '1800'] = process.argv.slice(2);
if (existsSync(out)) throw new Error(`${out} exists`);
mkdirSync(out, { recursive: true });
const args = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=native', '--use-angle=vulkan',
  '--disable-vulkan-surface', '--ignore-gpu-blocklist', '--enable-gpu', '--window-size=1600,1000'];
const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args, protocolTimeout: 0,
  defaultViewport: { width: 1600, height: 1000 } });
const page = await browser.newPage();
const cdp = await page.createCDPSession();
await cdp.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: out });
page.on('console', m => console.log('console:', m.text()));
page.on('pageerror', e => console.log('PAGE ERROR', e.message));
page.on('dialog', d => { console.log('dialog:', d.message()); d.accept(); });
// the <input type=file> fallback, which puppeteer can answer
await page.evaluateOnNewDocument(() => { delete window.showOpenFilePicker; });

const shot = name => page.screenshot({ path: `${out}/${name}.png` });
const text = sel => page.$eval(sel, el => el.hidden ? null : el.textContent.trim());
const t0 = Date.now();
const log = (...a) => console.log(`${((Date.now() - t0) / 1000).toFixed(1)}s`, ...a);

await page.goto(`http://localhost:${process.env.PORT ?? 5199}/${process.env.QUERY ?? ''}`);
await page.waitForSelector('#empty:not([hidden])');
log('empty screen:', await text('#empty h2'));
await shot('1-empty');

// importing first keeps the setup for the recording loaded next
await (await page.$('#importFile')).uploadFile(setupPath);
await page.waitForFunction(() => !document.getElementById('fileNote').hidden);
log('import:', await text('#fileNote span'));

const [chooser] = await Promise.all([page.waitForFileChooser(), page.click('#emptyLoad')]);
await chooser.accept([videoPath]);
await page.waitForFunction(() => document.getElementById('empty').hidden && document.getElementById('video').readyState >= 2, { timeout: 60e3 });
// the tour starts on the first visit, once there is a recording to show
await page.waitForSelector('#tour:not([hidden])');
await shot('2-tour');
await page.click('#tourSkip');
log('loaded:', await text('#videoName'), '· start', await text('#start'), '· end', await text('#end'), '· status', await text('#status'));
log('regions:', await page.$$eval('#list li', ls => ls.map(l => l.textContent.trim()).join(' | ')));
log('checks:', await page.$eval('#checks', el => el.hidden ? 'none' : el.textContent.trim()));
await new Promise(r => setTimeout(r, 1500));
await shot('3-loaded');

log('processor note:', await text('#cpuNote'));
await page.click('#run');
let last = {};
const deadline = Date.now() + +timeout * 1000;
while (Date.now() < deadline) {
  const s = await page.evaluate(() => ({
    running: !document.getElementById('runStatus').hidden,
    stage: document.getElementById('runStage').textContent, text: document.getElementById('runText').textContent,
    result: document.getElementById('runResult').hidden ? null : document.getElementById('resultText').textContent,
    error: document.getElementById('runError').textContent,
  }));
  const note = await text('#cpuNote');
  const line = (s.running ? `${s.stage} · ${s.text}` : `result: ${s.result} ${s.error}`) + (note ? ` [${note}]` : '');
  // one line per step, and its last progress line before the next
  if (s.stage !== last.stage || !s.running) { if (last.line) log(last.line); log(line); }
  last = { stage: s.stage, line };
  if (!s.running) break;
  await new Promise(r => setTimeout(r, 2000));
}
await shot('4-done');
const ok = await page.$eval('#saveVideo', el => !el.hidden);
if (ok) {
  await page.click('#saveVideo');
  await page.click('#saveCanvas');
  await page.click('#playResult');
  await new Promise(r => setTimeout(r, 3000));
  await shot('5-player');
  for (let i = 0; i < 60 && readdirSync(out).filter(f => !f.endsWith('.png') || f.includes('canvas')).filter(f => !f.endsWith('.crdownload')).length < 2; i++)
    await new Promise(r => setTimeout(r, 500));
}
for (const f of readdirSync(out)) log('file', f, statSync(`${out}/${f}`).size);
await browser.close();
process.exit(ok ? 0 : 1);
