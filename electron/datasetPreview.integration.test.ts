// End-to-end check of the dataset preview against a real Python with
// VapourSynth, BestSource and OpenCV.
//
// Opt-in: it spawns python and ffmpeg, so it stays out of the default run.
// Point VK_DATASET_PYTHON at a python that has them, with ffmpeg on PATH:
//
//   VK_DATASET_PYTHON=/path/python npx vitest run -c vitest.config.electron.ts electron/datasetPreview.integration.test.ts
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import * as path from 'path';
import * as fs from 'fs-extra';
import { spawnSync } from 'child_process';

const { root } = vi.hoisted(() => {
  const nodeOs = require('os') as typeof import('os');
  const nodePath = require('path') as typeof import('path');
  return { root: nodePath.join(nodeOs.tmpdir(), `vk-dataset-preview-it-${process.pid}`) };
});

vi.mock('electron', () => ({ ipcMain: { handle: vi.fn() }, dialog: {}, BrowserWindow: class {} }));
vi.mock('./logger', () => ({
  logger: { info: console.log, warn: console.warn, error: console.error, debug: () => {}, dependency: console.log },
}));
vi.mock('./constants', () => ({
  PATHS: {
    CONFIG: require('path').join(root, 'config'),
    PYTHON: process.env.VK_DATASET_PYTHON ?? 'python',
    FFMPEG: require('path').join(root, 'ffmpeg', 'bin', 'ffmpeg'),
  },
}));
vi.mock('./utils', () => ({ setupVSEnvironment: () => ({ ...process.env }) }));

import { closeDatasetPreview, loadPreview, previewChunks, previewFrame } from './datasetPreviewHandlers';
import { syncDatasetEngine } from './datasetEngine';

const configured = Boolean(process.env.VK_DATASET_PYTHON);

describe.skipIf(!configured)('dataset preview, end to end', () => {
  const clip = path.join(root, 'clip.mkv');
  const out = path.join(root, 'out');

  beforeAll(async () => {
    await fs.ensureDir(root);
    await syncDatasetEngine(path.resolve(__dirname, '..'));
    const made = spawnSync('ffmpeg', ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=320x240:rate=24:duration=1',
      '-pix_fmt', 'yuv420p', '-c:v', 'ffv1', clip]);
    expect(made.status).toBe(0);
  });
  afterAll(async () => {
    await closeDatasetPreview();
    await fs.remove(root);
  });

  const job = (scale = 0.5) => ({
    version: 1,
    seed: 3,
    nodes: [
      { id: 'src', type: 'load_video', params: { path: 'clip.mkv' }, ui: { x: 0, y: 0 } },
      { id: 'chunk', type: 'chunk', params: { strategy: 'frames', frames_per_chunk: 8 } },
      { id: 'down', type: 'resample', params: { size: 'scale', scale } },
      { id: 'enc', type: 'codec', params: {} },
      { id: 'seq', type: 'make_sequences', params: { length: 4 } },
      { id: 'out', type: 'write_dataset', params: { directory: out } },
    ],
    edges: [
      { from: 'src.clip', to: 'chunk.clip' },
      { from: 'chunk.clip', to: 'down.clip' },
      { from: 'down.clip', to: 'enc.clip' },
      { from: 'chunk.clip', to: 'seq.hr' },
      { from: 'enc.clip', to: 'seq.lr' },
      { from: 'seq.sequences', to: 'out.sequences' },
    ],
  });
  const frame = (overrides: Record<string, unknown> = {}) => ({
    node: 'enc', port: 'clip', video: 0, chunk: 1, frame: 2,
    compare: null, view: null, compareView: null, seed: null, ...overrides,
  });

  it('loads a job with relative paths read against its file, and lists its chunks', async () => {
    const loaded = await loadPreview(job(), path.join(root, 'job.vkdataset'));
    expect(loaded).toEqual({ success: true, value: { seed: 3, videos: [{ index: 0, name: 'clip' }], reused: 0 } });
    const chunks = await previewChunks(0, null);
    expect(chunks.success && chunks.value.chunks).toEqual([[0, 8], [8, 16], [16, 24]]);
  }, 120000);

  it('serves a frame with a compare picture, its draws and nothing written', async () => {
    const outcome = await previewFrame(frame({ compare: { node: 'src', port: 'clip' } }));
    expect(outcome.success).toBe(true);
    if (!outcome.success) return;
    const value = outcome.value;
    expect([value.width, value.height, value.data.length]).toEqual([160, 120, 160 * 120 * 3]);
    expect(value.compare).toMatchObject({ node: 'src', scope: 'video', offset: 8, n: 2, width: 320 });
    expect(value.compareData!.length).toBe(320 * 240 * 3);
    expect(Object.keys(value.settings).sort()).toEqual(['down', 'enc', 'src']);
    expect(await fs.pathExists(out)).toBe(false);

    const sequences = await previewFrame(frame({ node: 'seq', port: 'sequences', view: 'lr' }));
    expect(sequences.success && sequences.value.sequences!.items.map(s => s.frames.lr)).toEqual([[0, 1, 2, 3], [4, 5, 6, 7]]);
  }, 120000);

  it('answers a bad request with its error and keeps serving', async () => {
    const bad = await previewFrame(frame({ chunk: 7 }));
    expect(bad).toMatchObject({ success: false, error: expect.stringContaining('no chunk 7') });
    const good = await previewFrame(frame());
    expect(good.success).toBe(true);
  }, 120000);

  it('reloads an edit, keeping the unchanged nodes', async () => {
    const loaded = await loadPreview(job(0.25), path.join(root, 'job.vkdataset'));
    expect(loaded.success && loaded.value.reused).toBe(2);
    const outcome = await previewFrame(frame());
    expect(outcome.success && [outcome.value.width, outcome.value.height]).toEqual([80, 60]);
  }, 120000);

  it('rerolls the draws with a seed override', async () => {
    const draws = new Set<string>();
    for (const seed of [null, 1, 2, 3, 4]) {
      const outcome = await previewFrame(frame({ seed }));
      if (outcome.success) draws.add(JSON.stringify(outcome.value.settings));
    }
    expect(draws.size).toBeGreaterThan(1);
  }, 120000);

  it('closes, and starts again on the next load', async () => {
    await closeDatasetPreview();
    expect(await previewChunks(0, null)).toMatchObject({ success: false, error: expect.stringContaining('not running') });
    const loaded = await loadPreview(job(), path.join(root, 'job.vkdataset'));
    expect(loaded.success).toBe(true);
  }, 120000);
});
