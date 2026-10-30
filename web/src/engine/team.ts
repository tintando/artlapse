/**
 * The processor fallback's workers: a pass (one of cpu-kernels.ts's kernels over n rows) is split into
 * ranges that the team's workers run at once, on images in memory they all share. Shared memory needs a
 * cross-origin isolated page (the site's _headers); without it, or for a small pass, the kernel runs here,
 * over all n rows. Either way the results are the same: ranges are gathered in order.
 */
import { Kernel, kernels } from './cpu-kernels';

const isolated = typeof SharedArrayBuffer !== 'undefined' && (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
const SMALL = 1 << 15; // a pass over fewer elements than this isn't worth the messages

type Typed = Float32Array | Uint8Array | Uint8ClampedArray | Uint16Array | Uint32Array;
type Ctor<T> = { new (b: ArrayBufferLike, off?: number, n?: number): T; BYTES_PER_ELEMENT: number };

export class Team {
  private workers: Worker[] = [];
  private calls = new Map<number, { resolve: (r: unknown) => void; reject: (e: Error) => void }>();
  private id = 0;

  constructor() {
    if (!isolated || typeof Worker === 'undefined') return;
    const n = Math.max(1, Math.min((navigator.hardwareConcurrency || 2) - 1, 16));
    try {
      for (let i = 0; i < n; i++) {
        const w = new Worker(new URL('./team-worker.ts', import.meta.url), { type: 'module' });
        w.onmessage = (ev: MessageEvent<{ id: number; result?: unknown; error?: string }>) => {
          const c = this.calls.get(ev.data.id)!;
          this.calls.delete(ev.data.id);
          if (ev.data.error !== undefined) c.reject(new Error(ev.data.error)); else c.resolve(ev.data.result);
        };
        w.onerror = ev => {
          for (const c of this.calls.values()) c.reject(new Error(ev.message || 'a worker of the team stopped'));
          this.calls.clear();
        };
        this.workers.push(w);
      }
    } catch {
      this.workers.forEach(w => w.terminate());
      this.workers = [];
    }
  }

  /** How many threads share a pass. */
  get threads(): number { return Math.max(this.workers.length, 1); }

  /** A typed array of n elements, on shared memory when the team has workers. */
  alloc<T extends Typed>(ctor: Ctor<T>, n: number): T {
    const bytes = Math.max(n * ctor.BYTES_PER_ELEMENT, 4);
    return new ctor(this.workers.length ? new SharedArrayBuffer(bytes) : new ArrayBuffer(bytes), 0, n);
  }

  /** Run `kernel` over [0, n), in ranges across the team; `size` (elements touched, n by default) decides
   * whether it is worth sharing. Returns each range's result, in order. */
  async run<K extends Kernel>(kernel: K, args: Parameters<(typeof kernels)[K]>[0], n: number, size = n): Promise<ReturnType<(typeof kernels)[K]>[]> {
    const fn = kernels[kernel] as (a: unknown, i0: number, i1: number) => ReturnType<(typeof kernels)[K]>;
    if (!this.workers.length || n < 2 || size < SMALL) return [fn(args, 0, n)];
    const parts = Math.min(n, this.workers.length * 3);
    const calls = [];
    for (let p = 0; p < parts; p++) {
      const i0 = Math.floor((p * n) / parts), i1 = Math.floor(((p + 1) * n) / parts);
      const id = ++this.id;
      calls.push(new Promise<ReturnType<(typeof kernels)[K]>>((resolve, reject) => {
        this.calls.set(id, { resolve: resolve as (r: unknown) => void, reject });
        this.workers[p % this.workers.length].postMessage({ id, kernel, args, i0, i1 });
      }));
    }
    return Promise.all(calls);
  }
}

let team: Team | null = null;
/** The team, started at its first use. */
export const theTeam = () => (team ??= new Team());
