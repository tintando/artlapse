/**
 * Project configuration, the same settings and YAML layout as artlapse's config.yaml (see its config.py
 * for what each does). Screen coordinates are full-resolution video pixels, rectangles [x, y, w, h],
 * times seconds into the video ("h:mm:ss" accepted). Colours are BGR, as in artlapse.
 */
import { dump, load } from 'js-yaml';

export type Rect = [number, number, number, number];

export interface Config {
  video: string;
  device: string;
  sample_fps: number;
  start: number;
  end: number | null;
  cuts: [number, number][];
  workspace: Rect;
  exclude: Rect[];
  reference: { time: number; target: Rect };
  canvas: { long_side: number; background: [number, number, number]; continues: string };
  register: {
    analysis_width: number; max_features: number; ratio: number; ransac_px: number; min_inliers: number;
    min_inlier_ratio: number; keep_view_ratio: number; keep_view_spread: number; static_diff: number;
    map_cell_px: number; gpu_features: boolean;
  };
  accumulate: {
    diff_threshold: number; faint_threshold: number; ink_threshold: number; stable_seconds: number;
    erode_px: number; refine_gain: number; min_changed_px: number;
  };
  render: { duration: number; fps: number; hold_end: number; activity_weight: number; crf: number };
}

export function defaults(): Config {
  return {
    video: '', device: 'auto', sample_fps: 2.0, start: 0.0, end: null, cuts: [],
    workspace: [0, 0, 0, 0], exclude: [],
    reference: { time: 0.0, target: [0, 0, 0, 0] },
    canvas: { long_side: 2048, background: [255, 255, 255], continues: '' },
    register: {
      analysis_width: 960, max_features: 1500, ratio: 0.75, ransac_px: 2.0, min_inliers: 25, min_inlier_ratio: 0.15,
      keep_view_ratio: 0.3, keep_view_spread: 0.2, static_diff: 1.5, map_cell_px: 3.0, gpu_features: false,
    },
    accumulate: {
      diff_threshold: 28, faint_threshold: 10, ink_threshold: 8, stable_seconds: 8.0, erode_px: 3, refine_gain: 1.15,
      min_changed_px: 40,
    },
    render: { duration: 60.0, fps: 30, hold_end: 3.0, activity_weight: 0.8, crf: 18 },
  };
}

/** A time as seconds: a number, or "h:mm:ss" / "m:ss" with optional decimals. */
export function seconds(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value === 'number') return value;
  let total = 0;
  for (const part of String(value).trim().split(':')) total = total * 60 + parseFloat(part);
  if (Number.isNaN(total)) throw new Error(`not a time: ${value}`);
  return total;
}

/** Seconds as h:mm:ss.s. */
export function clock(t: number): string {
  t = Math.round(t * 10) / 10;
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  return `${h}:${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`;
}

function merge(into: Record<string, unknown>, data: Record<string, unknown>, path: string): void {
  for (const [k, v] of Object.entries(data ?? {})) {
    if (!(k in into)) throw new Error(`unknown setting ${path}${k}`);
    const cur = into[k];
    if (cur && typeof cur === 'object' && !Array.isArray(cur)) merge(cur as Record<string, unknown>, (v ?? {}) as Record<string, unknown>, `${path}${k}.`);
    else into[k] = v;
  }
}

/** A config from parsed YAML (or JSON) data: defaults for what it leaves out, times as seconds. */
export function fromData(data: Record<string, unknown>): Config {
  const cfg = defaults();
  merge(cfg as unknown as Record<string, unknown>, data, '');
  cfg.start = seconds(cfg.start) ?? 0;
  cfg.end = seconds(cfg.end);
  cfg.reference.time = seconds(cfg.reference.time) ?? 0;
  cfg.cuts = cfg.cuts.map(([a, b]) => [seconds(a)!, seconds(b)!]);
  return cfg;
}

export function parse(text: string): Config {
  return fromData((load(text) ?? {}) as Record<string, unknown>);
}

export function toYaml(cfg: Config): string {
  return dump(cfg, { sortKeys: false } as never);
}
