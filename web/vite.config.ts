import { defineConfig, type Plugin } from 'vite';
import { createReadStream, statSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, basename } from 'node:path';

/** Dev-server routes for the test pages only: serve a local file (with ranges, for videos) and save results. */
function testRoutes(): Plugin {
  return {
    name: 'artlapse-test-routes',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/__file', (req, res) => {
        const path = new URL(req.url!, 'http://x').searchParams.get('path')!;
        const size = statSync(path).size;
        const m = /bytes=(\d+)-(\d*)/.exec(req.headers.range ?? '');
        const start = m ? +m[1] : 0, end = m && m[2] ? +m[2] : size - 1;
        res.writeHead(m ? 206 : 200, {
          'Content-Length': end - start + 1, 'Accept-Ranges': 'bytes',
          ...(m ? { 'Content-Range': `bytes ${start}-${end}/${size}` } : {}),
        });
        createReadStream(path, { start, end }).pipe(res);
      });
      server.middlewares.use('/__save', (req, res) => {
        const name = basename(new URL(req.url!, 'http://x').searchParams.get('name')!);
        const dir = process.env.ARTLAPSE_TEST_OUT ?? 'test-out';
        mkdirSync(dir, { recursive: true });
        const chunks: Buffer[] = [];
        req.on('data', c => chunks.push(c));
        req.on('end', () => { writeFileSync(join(dir, name), Buffer.concat(chunks)); res.end('ok'); });
      });
    },
  };
}

export default defineConfig({
  plugins: [testRoutes()],
  worker: { format: 'es' },
  // the test server (ARTLAPSE_TEST_OUT set) never reloads a page mid-run when a file changes
  server: { hmr: !process.env.ARTLAPSE_TEST_OUT },
  build: { target: 'es2022' },
});
