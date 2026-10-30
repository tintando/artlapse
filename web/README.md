# Artlapse in the browser

[artlapse](../README.md) as a web page: the setup page and the whole engine (register, composite, render)
run in the visitor's browser, on their graphics card through WebGPU, with WebCodecs decoding the
recording and encoding the timelapse. The recording is read where it is and never uploaded; the site is
static files.

It needs a browser with WebCodecs, and runs best with WebGPU: Chrome or Edge 113 or later. Safari 26 and
Firefox 141 on Windows have both too, but are untested. Where WebGPU gives no device (no WebGPU, or no
graphics card it can use, as in Chrome on Linux without its flags), register and composite run on the
processor instead: the same passes as plain loops, each shared out by rows across a team of workers, one
per core. They share their images through `SharedArrayBuffer`, which needs the page cross-origin isolated:
`public/_headers` sends the two headers on the site, `vite.config.ts` on the dev server; without them the
passes run on one thread. The page says it uses the processor before, during and after a run. Adding `?cpu`
to the page's address makes it use the processor anyway.

```bash
npm install
npm run dev          # http://localhost:5173
npm run build        # dist/, the site
npm run deploy       # build, then upload dist/ to Cloudflare Pages by hand (wrangler login first)
```

A push to main that changes `web/` builds it and deploys it to the Cloudflare Pages project `artlapse`
(`.github/workflows/pages.yaml`).

## What is where

| | |
|---|---|
| `index.html` | artlapse's setup page (`src/artlapse/setup.html`), its server calls replaced by `src/ui/local.ts` |
| `src/ui/local.ts` | what the setup server did: the setup of each recording kept in `localStorage`, the recording's file handle in IndexedDB (to open it again after a reload), import and export, the run |
| `src/engine/worker.ts` | the run, off the page's thread; a stage whose inputs did not change is skipped, as in `pipeline.run_all` |
| `src/engine/frames.ts` | sampling (mediabunny over WebCodecs) and the grey analysis frames, made on the GPU |
| `src/engine/sift.ts` | SIFT in WGSL, following OpenCV's |
| `src/engine/match.ts`, `ransac.ts` | kNN matching on the GPU, 2-point similarity RANSAC |
| `src/engine/register.ts` | `register.py`: the feature map, the views, the decisions, the backward and forward passes |
| `src/engine/composite.ts` | `accumulate_gpu.py`'s canvas in WGSL |
| `src/engine/pipeline.ts` | the stages; canvas.mp4 and the timelapse are written to the origin-private file system |
| `src/engine/config.ts`, `geometry.ts`, `imgops.ts` | `config.py`, `geometry.py` and the OpenCV operations register uses |
| `src/engine/team.ts`, `team-worker.ts` | without a GPU: the team of workers a pass is shared across |
| `src/engine/cpu-kernels.ts` | the passes on the processor, each over a range of rows |
| `src/engine/sift-cpu.ts`, `cpu.ts`, `composite-cpu.ts` | SIFT, matching and the canvas on the processor, in the GPU's order |

Not ported: continuing a canvas (`canvas.continues`), `check`, `overlay`, `sidebyside`, the sample
cache, the decision log as a file, and the learned features (XFeat, ALIKED).

## Tests

The test pages in `test/` run in headless Chrome against the dev server, with the flags that give it
WebGPU on Linux. Started with `ARTLAPSE_TEST_OUT=<dir>`, the dev server serves local files to them
(`/__file?path=`), takes their results (`/__save?name=`, into that directory) and never reloads a page
mid-run.

```bash
ARTLAPSE_TEST_OUT=<fresh dir> npx vite --port 5199 --strictPort &
node test/page.mjs <video> <setup.yaml> <fresh dir>        # the page end to end: load, import, make, save
QUERY='?cpu' node test/page.mjs <video> <setup.yaml> <fresh dir>   # the same on the processor
node test/run.mjs "test/sift.html?img=<grey png>&out=<name>"   # SIFT on one image (&cpu=1: on the processor); compare with test/compare_sift.py
node test/run.mjs "test/register.html?video=<path>&config=<path>&out=<name>"   # a track, as JSON
node test/run.mjs "test/pipeline.html?video=<path>&config=<path>&track=<json>&out=<name>"
uv run --project .. python test/track_npz.py <track.json> <fresh dir>   # for artlapse's own tools
```

## On the processor

The first 5 minutes of the same test stream (600 samples), through the page by `test/page.mjs` in Chrome
on an i5-13400F (16 threads, so 15 workers), with `?cpu`, against the page on the RTX 3080:

| | processor | GPU |
|---|---|---|
| register | 135 s | 40 s |
| composite | 108 s | 14 s |
| render | 8 s | 6 s |
| all | 4 min 13 s | 58 s |

The final canvases differ by more than 30 on 0.17% of the pixels. On one frame the processor's SIFT finds
the GPU's keypoints (all within 0.01 px) with descriptors a median distance of 0 apart, in 300 ms for both
flip states against 18 ms.

## Against native artlapse

A test stream (36 min, 1920×1080, 4320 samples at 2 fps), made
through the page by `test/page.mjs` in Chrome 155 on an RTX 3080, against artlapse 501dbcf490 on the
same machine (i5-13400F):

| | browser | native | |
|---|---|---|---|
| register | 301 s | 218 s (CPU, 16 threads) | 1.4× |
| composite | 125 s | 47 s (CUDA) | 2.7× |
| render | 8 s | 19 s | 0.4× |
| all | 7 min 14 s | 4 min 44 s | 1.5× |

How close the results are:

- SIFT: on a frame of one test stream, every keypoint and descriptor is OpenCV's; on a frame of another, 94% of the
  keypoints are, with descriptors a median distance of 5 apart. The rest are ties in flat areas.
- Register: 98.0% of the samples get the same status (browser: 2289 matched, 1567 unmoved, 464
  lost; native: 2280, 1568, 472). The tracks drift apart slowly: the canvas corners are a median
  10.7 canvas px apart. Each is as consistent with the stream as the other: `scripts/residual.py --lag 60`
  gives a median of 1.5 px for the browser's track and 1.6 px for native's.
- Composite, from native's track: 69 snapshots on 200–300 s, as natively; 0.18% of the final canvas's
  pixels differ by more than 30.
- The whole run: the same 1890-frame timelapse; 8.5% of the final canvas's pixels differ by more than
  30, all along the edges of the lines, as the browser's canvas sits about 4 px to the side of native's.
  Which of the two is placed better is not settled.
