/** WebGPU helpers: the device, compute pipelines, buffers and reading them back. */

export class Gpu {
  private pipelines = new Map<string, GPUComputePipeline>();

  private constructor(readonly device: GPUDevice, readonly adapter: GPUAdapter) {}

  static async create(): Promise<Gpu> {
    if (!navigator.gpu) throw new Error('this browser has no WebGPU');
    // headless Chrome on Linux sometimes returns no adapter to the first request of a page
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
      ?? await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('WebGPU found no graphics card it can use');
    const want = ['shader-f16'].filter(f => adapter.features.has(f)) as GPUFeatureName[];
    const device = await adapter.requestDevice({
      requiredFeatures: want,
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage),
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      },
    });
    device.lost.then(info => console.error('WebGPU device lost:', info.message));
    return new Gpu(device, adapter);
  }

  /** A compute pipeline for `entry` in `code`, compiled once. */
  pipeline(code: string, entry: string, constants?: Record<string, number>): GPUComputePipeline {
    const key = `${entry}|${JSON.stringify(constants ?? {})}|${code}`;
    let p = this.pipelines.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      p = this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: entry, constants } });
      this.pipelines.set(key, p);
    }
    return p;
  }

  buffer(size: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST): GPUBuffer {
    return this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
  }

  uniform(data: ArrayBuffer | ArrayBufferView): GPUBuffer {
    const bytes = ArrayBuffer.isView(data) ? data.byteLength : data.byteLength;
    const buf = this.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buf, 0, data as BufferSource);
    return buf;
  }

  bind(p: GPUComputePipeline, buffers: (GPUBuffer | GPUExternalTexture)[], group = 0): GPUBindGroup {
    return this.device.createBindGroup({
      layout: p.getBindGroupLayout(group),
      entries: buffers.map((b, i) => ({ binding: i, resource: b instanceof GPUBuffer ? { buffer: b } : b })),
    });
  }

  /** Record a dispatch of `p` with its bindings on `enc` (one compute pass per dispatch keeps the order). */
  dispatch(enc: GPUCommandEncoder, p: GPUComputePipeline, bindings: (GPUBuffer | GPUExternalTexture)[], x: number, y = 1, z = 1): void {
    const pass = enc.beginComputePass();
    pass.setPipeline(p);
    pass.setBindGroup(0, this.bind(p, bindings));
    pass.dispatchWorkgroups(x, y, z);
    pass.end();
  }

  dispatchIndirect(enc: GPUCommandEncoder, p: GPUComputePipeline, bindings: GPUBuffer[], args: GPUBuffer, offset = 0): void {
    const pass = enc.beginComputePass();
    pass.setPipeline(p);
    pass.setBindGroup(0, this.bind(p, bindings));
    pass.dispatchWorkgroupsIndirect(args, offset);
    pass.end();
  }

  /** Copy `size` bytes of each (buffer, offset, size) into one staging buffer, submit, and return them. */
  async read(enc: GPUCommandEncoder, parts: [GPUBuffer, number, number][]): Promise<ArrayBuffer[]> {
    const sizes = parts.map(([, , s]) => Math.ceil(s / 4) * 4);
    const total = sizes.reduce((a, b) => a + b, 0);
    const staging = this.device.createBuffer({ size: Math.max(16, total), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    let at = 0;
    parts.forEach(([b, off], i) => {
      if (sizes[i]) enc.copyBufferToBuffer(b, off, staging, at, sizes[i]);
      at += sizes[i];
    });
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const all = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    const out: ArrayBuffer[] = [];
    at = 0;
    for (let i = 0; i < parts.length; i++) {
      out.push(all.slice(at, at + parts[i][2]));
      at += sizes[i];
    }
    return out;
  }
}

/** Workgroup counts for `n` threads in groups of `size`, split over x and y when x would exceed the limit. */
export function groups(n: number, size = 256): [number, number] {
  const g = Math.ceil(n / size);
  if (g <= 65535) return [Math.max(1, g), 1];
  const y = Math.ceil(g / 65535);
  return [Math.ceil(g / y), y];
}
