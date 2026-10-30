/** A worker of the processor fallback's team (team.ts): runs a kernel over a range. */
import { Kernel, kernels } from './cpu-kernels';

onmessage = (ev: MessageEvent<{ id: number; kernel: Kernel; args: unknown; i0: number; i1: number }>) => {
  const { id, kernel, args, i0, i1 } = ev.data;
  try {
    const result = (kernels[kernel] as (a: unknown, i0: number, i1: number) => unknown)(args, i0, i1);
    postMessage({ id, result });
  } catch (e) {
    postMessage({ id, error: e instanceof Error ? e.message : String(e) });
  }
};
