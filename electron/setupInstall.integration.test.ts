// The whole first-run setup against the real package indexes: the embedded
// Python, VapourSynth, FFmpeg and the plugin phase, as the Setup screen runs
// them, followed by the dataset engine's node list from the installed Python.
//
// Opt-in: it downloads several GB and takes a long while. Point
// VK_SETUP_INSTALL_ROOT at an empty folder; the install lands in its data\
// folder, as it would beside the app. VK_SETUP_GPU_VENDOR picks the vendor
// (default: nvidia).
//
//   VK_SETUP_INSTALL_ROOT=C:\path\to\root npx vitest run -c vitest.config.electron.ts electron/setupInstall.integration.test.ts
import { describe, expect, it, vi } from 'vitest';
import * as path from 'path';
import * as fs from 'fs-extra';
import { spawnSync } from 'child_process';

const { installRoot, repoRoot } = vi.hoisted(() => {
  const nodePath = require('path') as typeof import('path');
  return {
    installRoot: process.env.VK_SETUP_INSTALL_ROOT ?? '',
    repoRoot: nodePath.resolve(__dirname, '..'),
  };
});

vi.mock('electron', () => ({
  app: {
    isPackaged: false,
    getAppPath: () => installRoot,
    getPath: () => installRoot,
    getVersion: () => '2.0.0',
  },
  BrowserWindow: class {},
  ipcMain: { handle: vi.fn() },
}));
vi.mock('./logger', () => {
  const line = (level: string) => (...args: unknown[]) => console.log(`[${level}]`, ...args);
  return {
    logger: {
      info: line('info'), warn: line('warn'), error: line('error'), debug: () => {}, dependency: line('dep'),
      separator: () => {}, getLogPath: () => path.join(installRoot, 'setup.log'),
    },
  };
});
vi.mock('./utils', async (importOriginal) => ({
  ...await importOriginal<typeof import('./utils')>(),
  getBundledBasePath: () => repoRoot,
}));
vi.mock('./gpuDetection', () => ({
  detectGpuVendor: vi.fn(async () => process.env.VK_SETUP_GPU_VENDOR ?? 'nvidia'),
}));

import { DependencyManager } from './dependencyManager';
import { PluginInstaller } from './pluginInstaller';
import { configManager } from './configManager';
import { PATHS } from './constants';
import { datasetEngineArgs } from './datasetProtocol';
import { datasetEngineEnv } from './datasetHandlers';

describe.skipIf(!installRoot)('first-run setup, end to end', () => {
  it('installs everything the dataset engine needs', async () => {
    await fs.ensureDir(installRoot);
    const started = Date.now();

    await new DependencyManager(null).setupDependencies();
    await configManager.load();
    const plugins = await new PluginInstaller(null).installDependenciesForSetup();
    console.log(`Setup took ${((Date.now() - started) / 60000).toFixed(1)} min`);
    expect(plugins.success, plugins.summary ?? plugins.error).toBe(true);

    expect((await new PluginInstaller(null).checkInstalled()).installed).toBe(true);
    expect(await fs.pathExists(path.join(PATHS.CONFIG, 'vsmlrt.py'))).toBe(true);
    expect(await fs.pathExists(path.join(PATHS.SCRIPTS, 'havsfunc_org.py'))).toBe(true);
    if (process.platform === 'win32') {
      expect(await fs.pathExists(path.join(PATHS.PLUGINS, 'TCanny.dll'))).toBe(true);
    }

    const nodes = spawnSync(PATHS.PYTHON, datasetEngineArgs(PATHS.CONFIG, 'nodes'), {
      cwd: PATHS.CONFIG, env: datasetEngineEnv(), encoding: 'utf8',
    });
    expect(nodes.status, nodes.stderr).toBe(0);
    expect(nodes.stdout).toContain('fine_dehalo');
  }, 3 * 60 * 60 * 1000);
});
