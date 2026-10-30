// Open a test page in headless Chrome with WebGPU and wait for it to finish: node test/run.mjs <path?query> [timeout s]
import puppeteer from 'puppeteer-core';
const [url, timeout = '600'] = process.argv.slice(2);
const args = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=native', '--use-angle=vulkan',
  '--disable-vulkan-surface', '--ignore-gpu-blocklist', '--enable-gpu'];
const browser = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args, protocolTimeout: 0 });
const page = await browser.newPage();
page.on('console', m => console.log(m.text()));
page.on('pageerror', e => console.log('PAGE ERROR', e.message));
await page.goto(`http://localhost:${process.env.PORT ?? 5199}/${url}`);
const result = await page.waitForFunction('window.__done', { timeout: +timeout * 1000, polling: 500 }).then(h => h.jsonValue());
console.log('RESULT', JSON.stringify(result));
await browser.close();
process.exit(result && result.error ? 1 : 0);
