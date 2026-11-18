import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

test('DOM Contract: Every element queried by el() in app.mjs exists in index.html', () => {
  const htmlPath = path.resolve('apps/kinelab-motion/index.html');
  const appPath = path.resolve('apps/kinelab-motion/src/app.mjs');

  const html = fs.readFileSync(htmlPath, 'utf8');
  const app = fs.readFileSync(appPath, 'utf8');

  const matches = [...app.matchAll(/el\('([^']+)'\)/g)].map(m => m[1]);
  const uniqueIds = [...new Set(matches)];

  const missing = [];
  for (const id of uniqueIds) {
    const hasId = html.includes(`id="${id}"`) || html.includes(`id='${id}'`);
    if (!hasId) {
      missing.push(id);
    }
  }

  assert.deepEqual(missing, [], `Missing DOM IDs found in app.mjs: ${missing.join(', ')}`);
});
