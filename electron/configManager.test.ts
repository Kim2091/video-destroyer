import * as fs from 'fs-extra';
import * as path from 'path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { dir } = vi.hoisted(() => {
  const nodePath = require('path') as typeof import('path');
  const nodeOs = require('os') as typeof import('os');
  return { dir: nodePath.join(nodeOs.tmpdir(), `vk-config-test-${process.pid}`) };
});

vi.mock('./constants', () => ({ PATHS: { CONFIG: dir } }));
vi.mock('./logger', () => ({ logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() } }));
vi.mock('./rocmEnvironment', () => ({ DEFAULT_ROCM_SETTING: { mode: 'auto' }, setRocmSetting: vi.fn() }));

import { ConfigManager } from './configManager';

const configFile = path.join(dir, 'app-config.json');

describe('config file', () => {
  beforeEach(async () => { await fs.remove(dir); await fs.ensureDir(dir); });
  afterEach(async () => { await fs.remove(dir); });

  it('stays valid JSON with the last values when setters save at once', async () => {
    // Each setter saves the whole file; overlapping plain writes used to
    // leave invalid JSON.
    await fs.writeJson(configFile, { keptFromAnOlderVersion: { a: 1 } });
    const manager = new ConfigManager();
    await manager.load();
    for (let round = 0; round < 10; round++) {
      await Promise.all([
        manager.setGpuVendor(round % 2 ? 'nvidia' : 'amd'),
        manager.setPluginsGpuVendor(round % 2 ? 'amd' : 'nvidia'),
        manager.setAppVersion(`2.0.${round}`),
        manager.setRocmSetting(round % 2 ? { mode: 'environment' } : { mode: 'custom', customRoot: '/opt/rocm' }),
      ]);
      const saved = await fs.readJson(configFile);
      expect(saved.gpuVendor).toBe(round % 2 ? 'nvidia' : 'amd');
      expect(saved.appVersion).toBe(`2.0.${round}`);
      expect(saved.keptFromAnOlderVersion).toEqual({ a: 1 });
    }
  });

  it('moves an unreadable config aside before starting from defaults', async () => {
    await fs.writeFile(configFile, '{"appVersion": "1.0.0"}}');
    const manager = new ConfigManager();
    await manager.load();
    expect(manager.getAppVersion()).toBeUndefined();
    await manager.setAppVersion('2.0.0');

    const kept = (await fs.readdir(dir)).filter(name => name.startsWith('app-config.unreadable-'));
    expect(kept).toHaveLength(1);
    expect(await fs.readFile(path.join(dir, kept[0]), 'utf8')).toContain('1.0.0');
    expect((await fs.readJson(configFile)).appVersion).toBe('2.0.0');
  });

  it('does not let one manager mutate the defaults another starts from', async () => {
    const first = new ConfigManager();
    await first.load();
    await first.setGpuVendor('nvidia');
    const second = new ConfigManager();
    expect(second.getGpuVendor()).toBeUndefined();
  });
});
