// The tour in headless Chrome, on a desktop's and a phone's window: with no video it shows the page on the
// example, lights something at every step, leaves the page empty and keeps nothing; the tour on a recording
// still comes after it. node test/tour.mjs <out dir> [video]
// BASE is the page's address (default the test server). Without a video only the empty page is tested.
// The out dir gets a screenshot of each step; it must not exist yet.
import { mkdirSync, existsSync } from 'node:fs';
import puppeteer from 'puppeteer-core';

const [out, videoPath] = process.argv.slice(2);
if (existsSync(out)) throw new Error(`${out} exists`);
mkdirSync(out, { recursive: true });
const base = process.env.BASE ?? 'http://localhost:5199/';
const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: true });
let failed = 0;
const check = (ok, what) => { console.log(ok ? 'ok  ' : 'FAIL', what); if (!ok) failed++; };

// goes through the tour from its first step to its end
async function walk(page, name) {
  const total = +(await page.$eval('#tour .count', el => el.textContent)).split(' of ')[1];
  for (let i = 0; i < total; i++) {
    await new Promise(r => setTimeout(r, 300));  // the spot's move
    const s = await page.evaluate(() => {
      const q = sel => document.querySelector(sel).getBoundingClientRect();
      const spot = q('#tour .spot'), card = q('#tour .card');
      return { title: document.querySelector('#tour h3').textContent, w: spot.width, h: spot.height,
        cardIn: card.left >= 0 && card.top >= 0 && card.right <= innerWidth && card.bottom <= innerHeight };
    });
    await page.screenshot({ path: `${out}/${name}-${i + 1}.png` });
    if (i > 0) check(s.w > 20 && s.h > 20, `${name}: "${s.title}" lights something (${Math.round(s.w)}x${Math.round(s.h)})`);
    check(s.cardIn, `${name}: "${s.title}" has its card on screen`);
    await page.click('#tourNext');
  }
  check(await page.$eval('#tour', el => el.hidden), `${name}: the tour ended`);
}

for (const [name, viewport] of [['desktop', { width: 1600, height: 1000 }], ['phone', { width: 390, height: 844, hasTouch: true }]]) {
  const context = await browser.createBrowserContext();
  const page = await context.newPage();
  await page.setViewport(viewport);
  page.on('pageerror', e => check(false, `${name}: page error: ${e.message}`));
  await page.evaluateOnNewDocument(() => { delete window.showOpenFilePicker; });
  await page.goto(base);
  await page.waitForSelector('#empty:not([hidden])');
  const stored = () => page.evaluate(() => JSON.stringify(Object.entries(localStorage).sort()));
  const before = await stored();
  check(await page.$eval('#tour', el => el.hidden), `${name}: no tour by itself on the empty page`);

  await page.evaluate(() => { const slow = document.getElementById('slow'); if (slow) slow.hidden = true; });  // the website's warning on a touch screen
  await page.click('#helpOpen');
  await page.click('#tourAgain');
  await page.waitForSelector('#tour:not([hidden])');
  await walk(page, `${name}-empty`);

  const after = await page.evaluate(() => ({
    empty: document.body.classList.contains('empty'), example: document.body.classList.contains('example'),
    prompt: !document.getElementById('empty').hidden, poster: document.getElementById('video').hasAttribute('poster'),
    inert: document.querySelector('main').inert || document.querySelector('aside').inert,
    draw: getComputedStyle(document.getElementById('draw')).display }));
  check(after.empty && !after.example && after.prompt && !after.poster && !after.inert && after.draw === 'none',
    `${name}: the page is empty again ${JSON.stringify(after)}`);
  check(await stored() === before, `${name}: nothing kept in localStorage`);

  // Escape ends it the same way
  await page.click('#helpOpen');
  await page.click('#tourAgain');
  await page.waitForSelector('#tour:not([hidden])');
  await page.keyboard.press('Escape');
  check(await page.$eval('body', el => el.classList.contains('empty')) && await stored() === before, `${name}: Escape leaves it empty, nothing kept`);

  if (videoPath) {
    const [chooser] = await Promise.all([page.waitForFileChooser(), page.click('#emptyLoad')]);
    await chooser.accept([videoPath]);
    await page.waitForFunction(() => document.getElementById('empty').hidden && document.getElementById('video').readyState >= 2, { timeout: 60e3 });
    await page.waitForSelector('#tour:not([hidden])');
    check(true, `${name}: the tour on a recording still comes`);
    check(await page.$$eval('#list li', ls => !ls.some(l => l.querySelector('input')?.value === 'Avatar')), `${name}: the example's boxes are not in the recording's setup`);
    await walk(page, `${name}-video`);
    check((await page.evaluate(() => localStorage.getItem('artlapse.toured'))) === '1', `${name}: now it has been seen`);
  }
  await context.close();
}
await browser.close();
console.log(failed ? `${failed} failed` : 'all passed');
process.exit(failed ? 1 : 0);
