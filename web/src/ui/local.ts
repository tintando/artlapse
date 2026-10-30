/**
 * What artlapse's setup server does for its page, done in the browser: the recording is a file the
 * visitor picks (read where it is, never uploaded), the setup of each recording is kept in this
 * browser's storage, and the timelapse is made by a worker on this computer's GPU.
 */
import { load as loadYaml, dump } from 'js-yaml';
import { Config, defaults, fromData, toYaml } from '../engine/config';
import { Video } from '../engine/frames';
import type { FromWorker, ToWorker } from '../engine/worker';
import type { Stage } from '../engine/pipeline';

const SETUP = 'artlapse.setup.'; // + the recording's key: { config, names }
const LAST = 'artlapse.last'; // the key of the recording loaded last
const STAGES: Stage[] = ['register', 'composite', 'render'];

interface Stored { config: Config; names: string[] }
interface Info { width: number; height: number; duration: number }

let file: File | null = null;
let key: string | null = null;
let info: Info | null = null;
let url: string | null = null;
// a setup imported before any recording was loaded: used for the next one, with the name of the one it was made on
let pending: { stored: Stored; video: string } | null = null;

const keyOf = (f: File) => `${f.name}|${f.size}|${f.lastModified}`;

function read(k: string): Stored | null {
  try {
    const s = localStorage.getItem(SETUP + k);
    if (!s) return null;
    const d = JSON.parse(s);
    return { config: fromData(d.config), names: d.names ?? [] };
  } catch { return null; }
}
function write(k: string, s: Stored): boolean {
  try { localStorage.setItem(SETUP + k, JSON.stringify(s)); localStorage.setItem(LAST, k); return true; } catch { return false; }
}

// The file handle of the last recording, where the browser can keep one (Chrome, Edge): the page offers
// to open it again after a reload, with one click to grant access.
const DB = 'artlapse', STORE = 'handles';
function db(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}
async function keepHandle(h: FileSystemFileHandle | null) {
  try {
    const d = await db();
    const tx = d.transaction(STORE, 'readwrite');
    if (h) tx.objectStore(STORE).put(h, 'video'); else tx.objectStore(STORE).delete('video');
  } catch { /* not kept: the page asks for the file again */ }
}
async function lastHandle(): Promise<FileSystemFileHandle | null> {
  try {
    const d = await db();
    return await new Promise(resolve => {
      const r = d.transaction(STORE).objectStore(STORE).get('video');
      r.onsuccess = () => resolve(r.result ?? null);
      r.onerror = () => resolve(null);
    });
  } catch { return null; }
}

/** The recording loaded last, if the browser kept it: its name, and a function that opens it again
 * (to be called from a click, which lets the browser ask for access). */
export async function reopenable(): Promise<{ name: string; open: () => Promise<void> } | null> {
  const h = await lastHandle();
  if (!h) return null;
  return {
    name: h.name,
    open: async () => {
      const opts = { mode: 'read' } as const;
      const ok = (await (h as any).queryPermission?.(opts)) === 'granted' || (await (h as any).requestPermission?.(opts)) === 'granted';
      if (!ok) throw new Error('the browser did not allow reading it');
      await setVideo(await h.getFile(), h);
    },
  };
}

/** Pick the recording; resolves false when the visitor cancels. */
export async function pickVideo(): Promise<boolean> {
  const w = window as unknown as { showOpenFilePicker?: (o: unknown) => Promise<FileSystemFileHandle[]> };
  if (w.showOpenFilePicker) {
    try {
      const [h] = await w.showOpenFilePicker({ types: [{ description: 'Videos', accept: { 'video/*': ['.mp4', '.mkv', '.mov', '.webm', '.ts', '.m4v'] } }] });
      await setVideo(await h.getFile(), h);
      return true;
    } catch (e) {
      if ((e as Error).name === 'AbortError') return false;
      throw e;
    }
  }
  const input = Object.assign(document.createElement('input'), { type: 'file', accept: 'video/*,.mkv,.ts' });
  const f = await new Promise<File | null>(resolve => {
    input.onchange = () => resolve(input.files?.[0] ?? null);
    input.addEventListener('cancel', () => resolve(null));
    input.click();
  });
  if (!f) return false;
  await setVideo(f, null);
  return true;
}

/** Make `f` the recording (setup_ui.load_video): its own setup when it was set up before in this browser;
 * otherwise the current boxes when it has the same size, with start, end and cuts reset. */
async function setVideo(f: File, handle: FileSystemFileHandle | null) {
  let v: Video;
  try {
    v = await Video.open(f);
  } catch (e) {
    throw new Error(`can't read ${f.name} as a video (${(e as Error).message})`);
  }
  const k = keyOf(f);
  const nextInfo = { width: v.info.width, height: v.info.height, duration: v.info.duration };
  let stored = read(k);
  if (!stored && pending) {
    stored = pending.stored;
    if (pending.video !== f.name) { stored.config.start = 0; stored.config.end = null; stored.config.cuts = []; }
    pending = null;
  }
  if (!stored) {
    const prev = key ? read(key) : null;
    const cfg = prev ? prev.config : defaults();
    let names = prev ? prev.names : [];
    if (!prev || !info || info.width !== nextInfo.width || info.height !== nextInfo.height) {
      cfg.workspace = [0, 0, nextInfo.width, nextInfo.height];
      cfg.exclude = [];
      cfg.reference.target = [0, 0, 0, 0];
      names = [];
    }
    cfg.start = 0; cfg.end = null; cfg.cuts = [];
    cfg.reference.time = Math.min(cfg.reference.time, nextInfo.duration);
    stored = { config: cfg, names };
  }
  stored.config.video = f.name;
  file = f; key = k; info = nextInfo;
  write(k, stored);
  if (url) URL.revokeObjectURL(url);
  url = URL.createObjectURL(f);
  await keepHandle(handle);
}

/** The page's /config: the setup and the video's size and length, or { video: null } before one is loaded. */
export function config(): Record<string, unknown> {
  if (!file || !key || !info) return { video: null };
  const { config: c, names } = read(key)!;
  return {
    workspace: c.workspace, exclude: c.exclude, start: c.start, end: c.end, cuts: c.cuts,
    reference: { time: c.reference.time, target: c.reference.target }, duration: c.render.duration,
    video: { path: file.name, ...info }, names,
  };
}

export const videoUrl = () => url;

/** The page's POST /config: the edited settings and names. */
export function save(body: string): boolean {
  if (!key) return false;
  const data = JSON.parse(body);
  const s = read(key)!;
  const c = s.config;
  for (const k of ['workspace', 'exclude', 'start', 'end', 'cuts'] as const) if (k in data) (c as any)[k] = data[k];
  if (data.reference) c.reference = { time: +data.reference.time, target: data.reference.target.map(Number) };
  if ('duration' in data) c.render.duration = +data.duration;
  if ('names' in data) s.names = data.names;
  return write(key, s);
}

/** The setup as a file: config.yaml with the exclusions' names, as artlapse's setup page exports it. */
export function exportSetup(): void {
  if (!key) return;
  const s = read(key)!;
  const text = toYaml(s.config) + dump({ names: s.names });
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(new Blob([text], { type: 'application/yaml' })),
    download: `${file!.name.replace(/\.[^.]*$/, '')}-setup.yaml`,
  });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/** Take an exported setup (setup_ui.import_setup): its boxes and settings; made on another recording, not its
 * start, end and cuts. Returns whether it was made on another recording, or is waiting for one to be loaded. */
export function importSetup(text: string): { other_video: boolean; waiting?: boolean } {
  const data = loadYaml(text) as Record<string, unknown>;
  if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('this is not an artlapse setup');
  const names = ((data.names as string[]) ?? []).map(String);
  delete data.names;
  const theirs = String(data.video ?? '');
  data.video = file?.name ?? theirs;
  const cfg = fromData(data); // unknown settings are refused before anything is kept
  const other = !!file && theirs.split(/[\\/]/).pop() !== file.name;
  if (other) { cfg.start = 0; cfg.end = null; cfg.cuts = []; }
  if (!key) {
    pending = { stored: { config: cfg, names }, video: theirs.split(/[\\/]/).pop()! };
    return { other_video: false, waiting: true };
  }
  write(key, { config: cfg, names });
  return { other_video: other };
}

// The run: the same status as the setup server's /run, for the page's renderRun.
interface RunStatus {
  state: 'idle' | 'running' | 'done' | 'failed' | 'stopped';
  stages: Partial<Record<Stage, string>>;
  progress: { stage?: Stage; done?: number; total?: number; eta?: number };
  started: number | null; ended: number | null; lines: string[]; changed: boolean; timelapse: number | null;
}
const st: RunStatus = { state: 'idle', stages: {}, progress: {}, started: null, ended: null, lines: [], changed: false, timelapse: null };
let worker: Worker | null = null;
let runConfig = '';
let stageStart = 0;
let result: { timelapse: Blob; final: Blob; url: string; name: string } | null = null;
let wake: { release: () => Promise<void> } | null = null;

function ensureWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL('../engine/worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (ev: MessageEvent<FromWorker>) => {
    const m = ev.data;
    if (m.type === 'stage') {
      st.stages[m.stage] = m.how;
      if (m.how === 'running') { st.progress = {}; stageStart = performance.now(); }
    } else if (m.type === 'progress') {
      const p = m.progress;
      const rate = p.done / ((performance.now() - stageStart) / 1000);
      st.progress = { stage: p.stage, done: p.done, total: p.total, eta: rate > 0 ? (p.total - p.done) / rate : 0 };
      st.lines = [`${p.stage} ${p.done}/${p.total} ${p.detail}`];
    } else {
      st.state = m.type === 'done' ? 'done' : m.type;
      st.ended = Date.now() / 1000;
      if (m.type === 'failed') st.lines = [m.error];
      if (m.type === 'done') {
        if (result) URL.revokeObjectURL(result.url);
        result = { timelapse: m.timelapse, final: m.final, url: URL.createObjectURL(m.timelapse), name: file!.name.replace(/\.[^.]*$/, '') };
        st.timelapse = st.ended;
      }
      wake?.release().catch(() => {});
      wake = null;
    }
  };
  worker.onerror = ev => {
    st.state = 'failed';
    st.lines = [ev.message || 'the worker stopped'];
    st.ended = Date.now() / 1000;
  };
  return worker;
}

export function runStatus(): RunStatus {
  st.changed = st.started !== null && !!key && JSON.stringify(read(key)?.config) !== runConfig;
  return { ...st, stages: { ...st.stages }, progress: { ...st.progress } };
}

export function run(): RunStatus {
  if (!file || !key) return runStatus();
  if (st.state === 'running') stop();
  const cfg = read(key)!.config;
  runConfig = JSON.stringify(cfg);
  Object.assign(st, { state: 'running', stages: { register: 'running' }, progress: {}, started: Date.now() / 1000, ended: null, lines: [] });
  stageStart = performance.now();
  if (!navigator.gpu) {
    Object.assign(st, { state: 'failed', lines: ['This browser has no WebGPU, which making the timelapse needs: use a recent Chrome, Edge or Safari.'], ended: Date.now() / 1000 });
    return runStatus();
  }
  (navigator as any).wakeLock?.request('screen').then((w: typeof wake) => { wake = w; }).catch(() => {});
  ensureWorker().postMessage({ type: 'run', file, video: key, config: cfg } satisfies ToWorker);
  return runStatus();
}

export function stop(): RunStatus {
  if (st.state === 'running') {
    worker?.postMessage({ type: 'stop' } satisfies ToWorker);
    st.state = 'stopped';
    st.ended = Date.now() / 1000;
  }
  return runStatus();
}

export const timelapseUrl = () => result?.url ?? null;

export function download(what: 'timelapse' | 'final'): void {
  if (!result) return;
  const blob = what === 'timelapse' ? result.timelapse : result.final;
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(blob), download: `${result.name}-${what === 'timelapse' ? 'timelapse.mp4' : 'canvas.png'}`,
  });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

export const stages = STAGES;
