// End-to-end check of the dataset runner against a real Python with
// VapourSynth, BestSource and OpenCV.
//
// Opt-in: it spawns python and ffmpeg, so it stays out of the default run.
// Point VK_DATASET_PYTHON at a python that has them, with ffmpeg on PATH:
//
//   VK_DATASET_PYTHON=/path/python npx vitest run -c vitest.config.electron.ts electron/datasetHandlers.integration.test.ts
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import * as path from 'path';
import * as fs from 'fs-extra';
import { spawnSync } from 'child_process';

const { root } = vi.hoisted(() => {
  const nodeOs = require('os') as typeof import('os');
  const nodePath = require('path') as typeof import('path');
  return { root: nodePath.join(nodeOs.tmpdir(), `vk-dataset-it-${process.pid}`) };
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

import { checkJob, getNodeSpecs, getStages, runJob, stopDatasetJob } from './datasetHandlers';
import { syncDatasetEngine } from './datasetEngine';

const configured = Boolean(process.env.VK_DATASET_PYTHON);

describe.skipIf(!configured)('dataset runner, end to end', () => {
  const clip = path.join(root, 'clip.mkv');
  const out = path.join(root, 'out');
  const longClip = path.join(root, 'long.mkv');

  beforeAll(async () => {
    await fs.ensureDir(root);
    await syncDatasetEngine(path.resolve(__dirname, '..'));
    const made = spawnSync('ffmpeg', ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=320x240:rate=24:duration=1',
      '-pix_fmt', 'yuv420p', '-c:v', 'ffv1', clip]);
    expect(made.status).toBe(0);
    const madeLong = spawnSync('ffmpeg', ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=320x240:rate=24:duration=60',
      '-pix_fmt', 'yuv420p', '-c:v', 'ffv1', longClip]);
    expect(madeLong.status).toBe(0);
  });
  afterAll(async () => { await fs.remove(root); });

  it('loads the node declarations', async () => {
    const specs = await getNodeSpecs();
    expect(specs.success).toBe(true);
    if (specs.success) expect(specs.value.map(s => (s as { type: string }).type)).toContain('write_dataset');
  });

  it('loads the stages of a job, in order', async () => {
    const stages = await getStages();
    expect(stages.success).toBe(true);
    if (stages.success) expect(stages.value.map(s => (s as { stage: number }).stage)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });

  const job = () => ({
    version: 1,
    seed: 3,
    nodes: [
      { id: 'src', type: 'load_video', params: { path: clip }, ui: { x: 0, y: 0 } },
      { id: 'chunk', type: 'chunk', params: { strategy: 'frames', frames_per_chunk: 8 } },
      { id: 'down', type: 'resample', params: { size: 'scale', scale: { uniform: [0.45, 0.55] } } },
      { id: 'seq', type: 'make_sequences', params: { length: 4 } },
      { id: 'out', type: 'write_dataset', params: { directory: out } },
    ],
    edges: [
      { from: 'src.clip', to: 'chunk.clip' },
      { from: 'chunk.clip', to: 'down.clip' },
      { from: 'chunk.clip', to: 'seq.hr' },
      { from: 'down.clip', to: 'seq.lr' },
      { from: 'seq.sequences', to: 'out.sequences' },
    ],
  });

  it('checks a job, and passes on the engine’s reason for refusing one', async () => {
    expect(await checkJob(job(), null)).toEqual({ success: true, value: null });
    const noWriter = job();
    noWriter.nodes.pop();
    noWriter.edges.pop();
    const refused = await checkJob(noWriter, null);
    expect(refused).toEqual({ success: false, error: 'a job needs at least one writer node' });
  });

  it('runs a job with progress, and writes the pairs', async () => {
    const progress: number[] = [];
    const sender = { isDestroyed: () => false, send: (_: string, e: { kind: string; progress?: number }) => {
      if (e.kind === 'progress') progress.push(e.progress!);
    } };
    const outcome = await runJob(job(), path.join(root, 'job.vkdataset'), sender as never);
    expect(outcome).toMatchObject({ success: true, value: { videos: 1, sequences: 6 } });
    expect(progress.at(-1)).toBe(1);
    expect((await fs.readdir(path.join(out, 'HR'))).length).toBe(24);
    expect((await fs.readdir(path.join(out, 'LR'))).length).toBe(24);
    expect(await fs.pathExists(path.join(out, 'manifest.json'))).toBe(true);
  }, 120000);

  it('stops a run when asked, and says it was stopped', async () => {
    const long = job();
    long.nodes[0].params = { path: longClip };
    long.nodes[4].params = { directory: path.join(root, 'stopped') };
    let stopped = false;
    const sender = { isDestroyed: () => false, send: (_: string, e: { kind: string }) => {
      if (e.kind === 'progress' && !stopped) stopped = stopDatasetJob();
    } };
    const outcome = await runJob(long, null, sender as never);
    expect(stopped).toBe(true);
    expect(outcome).toEqual({ success: false, error: 'The dataset job was stopped.', cancelled: true });
  }, 120000);
});
