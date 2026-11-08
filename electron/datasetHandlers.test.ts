import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as path from 'path';
import * as fs from 'fs-extra';

const { root } = vi.hoisted(() => {
  const nodeOs = require('os') as typeof import('os');
  const nodePath = require('path') as typeof import('path');
  return { root: nodePath.join(nodeOs.tmpdir(), `vk-dataset-test-${process.pid}`) };
});

vi.mock('electron', () => ({ ipcMain: { handle: vi.fn() }, dialog: {}, BrowserWindow: class {} }));
vi.mock('./logger', () => ({ logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn(), dependency: vi.fn() } }));
vi.mock('./constants', () => ({ PATHS: { CONFIG: require('path').join(root, 'config'), PYTHON: 'python', FFMPEG: 'ffmpeg' } }));
vi.mock('./utils', () => ({ setupVSEnvironment: () => ({}) }));
vi.mock('./inferenceBackend', () => ({
  RUNTIME_SETTINGS_ENV: 'VK_DATASET_RUNTIME',
  engineRuntimeEnvValue: () => '{"backend": "directml"}',
}));

import { datasetEngineEnv, listPresets, readJobFile, writeJob } from './datasetHandlers';
import { datasetEnginePath, datasetPresetsPath, syncDatasetEngine } from './datasetEngine';

const job = { version: 1, seed: 1, nodes: [{ id: 'a', type: 'load_video', ui: { x: 1, y: 2 } }], edges: [] };

beforeEach(async () => { await fs.ensureDir(root); });
afterEach(async () => { await fs.remove(root); });

describe('installing the dataset engine', () => {
  it('replaces the installed package as a whole and leaves bytecode behind', async () => {
    const bundled = path.join(root, 'app');
    await fs.outputFile(path.join(bundled, 'include', 'vk_dataset', 'nodes', 'source.py'), 'new');
    await fs.outputFile(path.join(bundled, 'include', 'vk_dataset', '__pycache__', 'x.pyc'), '');
    await fs.outputFile(path.join(datasetEnginePath(), 'nodes', 'dropped.py'), 'old');

    await syncDatasetEngine(bundled);

    expect(await fs.readFile(path.join(datasetEnginePath(), 'nodes', 'source.py'), 'utf8')).toBe('new');
    expect(await fs.pathExists(path.join(datasetEnginePath(), 'nodes', 'dropped.py'))).toBe(false);
    expect(await fs.pathExists(path.join(datasetEnginePath(), '__pycache__'))).toBe(false);
  });

  it('leaves the install alone when the app has no bundled engine', async () => {
    await fs.outputFile(path.join(datasetEnginePath(), '__main__.py'), 'kept');
    await syncDatasetEngine(path.join(root, 'nowhere'));
    expect(await fs.readFile(path.join(datasetEnginePath(), '__main__.py'), 'utf8')).toBe('kept');
  });
});

describe('dataset job files', () => {
  it('saves and opens a job with its node positions', async () => {
    const file = path.join(root, 'a.vkdataset');
    await writeJob(file, job);
    expect(await readJobFile(file)).toEqual({ success: true, value: job });
  });

  it('says why a file is not a job', async () => {
    const file = path.join(root, 'b.vkdataset');
    await fs.writeFile(file, JSON.stringify({ nodes: [] }));
    const read = await readJobFile(file);
    expect(read.success).toBe(false);
    if (!read.success) expect(read.error).toMatch(/^This is not a dataset job \(edges: /);
    await fs.writeFile(file, '{');
    expect((await readJobFile(file)).success).toBe(false);
  });

  it('lists the presets in name order and skips files that are not jobs', async () => {
    expect(await listPresets()).toEqual([]);
    await fs.outputJson(path.join(datasetPresetsPath(), 'video_pairs.json'), job);
    await fs.outputJson(path.join(datasetPresetsPath(), 'a_first.json'), job);
    await fs.outputFile(path.join(datasetPresetsPath(), 'broken.json'), '{');
    await fs.outputFile(path.join(datasetPresetsPath(), 'notes.txt'), '');
    expect((await listPresets()).map(p => p.name)).toEqual(['A first', 'Video pairs']);
  });
});

describe('the engine environment', () => {
  it('carries the inference settings for run and preview alike', () => {
    expect(JSON.parse(datasetEngineEnv().VK_DATASET_RUNTIME!)).toEqual({ backend: 'directml' });
  });
});
