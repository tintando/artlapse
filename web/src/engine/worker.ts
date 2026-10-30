/**
 * Makes the timelapse off the page's thread: register, composite, render. A stage whose inputs did not
 * change since its last run in this worker is skipped (pipeline.py's run_all), so a new length only
 * renders again.
 */
import { Config } from './config';
import { Video } from './frames';
import { Gpu } from './gpu';
import { Composite, Progress, Stage, clearFiles, runComposite, runRegister, runRender } from './pipeline';
import { Track } from './register';

export type ToWorker = { type: 'run'; file: File; video: string; config: Config } | { type: 'stop' };
export type FromWorker =
  | { type: 'stage'; stage: Stage; how: 'running' | 'done' | 'skipped' }
  | { type: 'progress'; progress: Progress }
  | { type: 'done'; timelapse: Blob; final: Blob }
  | { type: 'failed'; error: string }
  | { type: 'stopped' };

const SETTINGS: Record<Stage, (keyof Config)[]> = {
  register: ['sample_fps', 'start', 'end', 'cuts', 'workspace', 'exclude', 'reference', 'register'],
  composite: ['canvas', 'accumulate'],
  render: ['render'],
};

/** Everything a stage's output depends on: the video, and its own and earlier stages' settings. */
function inputs(cfg: Config, video: string, stage: Stage): string {
  const stages: Stage[] = ['register', 'composite', 'render'];
  const keys = stages.slice(0, stages.indexOf(stage) + 1).flatMap(s => SETTINGS[s]);
  return JSON.stringify([video, ...keys.map(k => cfg[k])]);
}

let gpu: Gpu | null = null;
let stopped = false;
let current = 0; // the run going on; an earlier one still unwinding stops at its next sample and says nothing
let last: { register?: [string, Track]; composite?: [string, Composite]; render?: [string, Blob] } = {};

async function run(id: number, file: File, videoKey: string, cfg: Config) {
  stopped = false;
  const stop = () => stopped || id !== current;
  const post = (m: FromWorker) => { if (id === current) postMessage(m); };
  if (!gpu) {
    await clearFiles();
    gpu = await Gpu.create();
  }
  const video = await Video.open(file);
  const report = (p: Progress) => post({ type: 'progress', progress: p });
  const step = async <T>(stage: Stage, make: () => Promise<T>, keep: (k: string, v: T) => void, had?: [string, T]): Promise<T> => {
    const key = inputs(cfg, videoKey, stage);
    if (had && had[0] === key) {
      post({ type: 'stage', stage, how: 'skipped' });
      return had[1];
    }
    post({ type: 'stage', stage, how: 'running' });
    const out = await make();
    keep(key, out);
    post({ type: 'stage', stage, how: 'done' });
    return out;
  };
  // once a stage runs, the ones after it run too
  let force = false;
  const track = await step('register', () => runRegister(cfg, video, gpu!, report, stop), (k, v) => { last = { register: [k, v] }; force = true; }, last.register);
  const comp = await step('composite', () => runComposite(cfg, track, video, gpu!, report, stop),
    (k, v) => { last.composite = [k, v]; last.render = undefined; force = true; }, force ? undefined : last.composite);
  const timelapse = await step('render', () => runRender(cfg, comp, report, stop), (k, v) => { last.render = [k, v]; }, force ? undefined : last.render);
  post({ type: 'done', timelapse, final: comp.final });
}

onmessage = (ev: MessageEvent<ToWorker>) => {
  const m = ev.data;
  if (m.type === 'stop') { stopped = true; return; }
  const id = ++current;
  run(id, m.file, m.video, m.config).catch(e => {
    if (id !== current) return;
    if (stopped) postMessage({ type: 'stopped' } satisfies FromWorker);
    else postMessage({ type: 'failed', error: e instanceof Error ? e.message : String(e) } satisfies FromWorker);
    console.error(e);
  });
};
