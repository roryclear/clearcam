import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Resolve relative to this file so the suite passes from any working directory.
const APP_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

test('DOM Contract: Every element queried by el() in app.mjs exists in index.html', () => {
  const htmlPath = path.join(APP_DIR, 'index.html');
  const appPath = path.join(APP_DIR, 'src/app.mjs');

  const html = fs.readFileSync(htmlPath, 'utf8');
  const app = fs.readFileSync(appPath, 'utf8');

  const matches = [...app.matchAll(/\bel\('([^']+)'\)/g)].map(m => m[1]);
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
