/**
 * Matching on the processor, for browsers without WebGPU: match.ts's brute force, the queries shared out
 * across the team's workers (team.ts).
 */
import { Knn } from './match';
import { Team, theTeam } from './team';

export class CpuMatcher implements Knn {
  private map: Uint8Array;
  private nm = 0;

  constructor(private team: Team = theTeam()) {
    this.map = team.alloc(Uint8Array, 0);
  }

  setMap(desc: Uint8Array, n: number): void {
    if (this.map.length < n * 128) this.map = this.team.alloc(Uint8Array, Math.max(n, 2 * this.map.length / 128, 1024) * 128);
    this.map.set(desc.subarray(0, n * 128));
    this.nm = n;
  }

  async knn2(queries: Uint8Array[]): Promise<Uint32Array[]> {
    const counts = queries.map(q => q.length / 128);
    const total = counts.reduce((a, b) => a + b, 0);
    if (!total || this.nm < 2) return counts.map(() => new Uint32Array(0));
    const query = this.team.alloc(Uint8Array, total * 128), out = this.team.alloc(Uint32Array, total * 4);
    let at = 0;
    for (const q of queries) { query.set(q, at * 128); at += q.length / 128; }
    await this.team.run('knn', { query, map: this.map, nm: this.nm, out }, total, total * this.nm * 16);
    at = 0;
    return counts.map(c => { const r = out.slice(at * 4, (at + c) * 4); at += c; return r; });
  }
}
