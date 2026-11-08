import { describe, expect, it, vi } from 'vitest';

// inferenceBackend imports constants.ts, which reads app paths at module scope.
vi.mock('electron', async () => {
  const p = await import('path');
  const o = await import('os');
  const root = p.join(o.tmpdir(), `vk-backend-test-${process.pid}`);
  return { app: { isPackaged: false, getAppPath: () => root, getPath: () => root, getVersion: () => '0.0.0' } };
});
vi.mock('./logger', () => ({ logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() } }));

const config = vi.hoisted(() => ({ backend: 'auto' as string, vendor: 'nvidia' as string | undefined }));
vi.mock('./configManager', () => ({
  configManager: {
    getInferenceBackend: () => config.backend,
    getPluginsGpuVendor: () => config.vendor,
    getGpuVendor: () => undefined,
  },
}));

import { PATHS } from './constants';
import {
  availableBackends,
  currentInferenceBackends,
  describeInferenceBackends,
  engineRuntimeEnvValue,
  engineRuntimeSettings,
  parseInferenceBackendSetting,
  resolveInferenceBackend,
} from './inferenceBackend';

const nothingInstalled = () => false;
const everythingInstalled = () => true;

describe('the backends this machine can use', () => {
  it("is the vendor's list, best first, before any plugin is installed", () => {
    expect(availableBackends('nvidia', 'win32', nothingInstalled)).toEqual(['tensorrt', 'directml', 'ncnn']);
    expect(availableBackends('amd', 'linux', nothingInstalled)).toEqual(['ncnn', 'migraphx']);
  });

  it('leaves out a backend whose plugin is missing', () => {
    const exists = (file: string) => !file.includes(`${require('path').sep}trt${require('path').sep}`);
    expect(availableBackends('nvidia', 'win32', exists)).toEqual(['directml', 'ncnn']);
  });

  it('never offers a backend the OS does not support', () => {
    expect(availableBackends('intel', 'linux', everythingInstalled)).toEqual(['ncnn']);
    expect(availableBackends('nvidia', 'darwin', everythingInstalled)).toEqual([]);
  });
});

describe('the Default AI backend setting', () => {
  it('reads anything that is not a backend as auto', () => {
    expect(parseInferenceBackendSetting('directml')).toBe('directml');
    expect(parseInferenceBackendSetting('cuda')).toBe('auto');
    expect(parseInferenceBackendSetting(undefined)).toBe('auto');
  });

  it('resolves auto to the best available backend', () => {
    expect(resolveInferenceBackend('auto', ['tensorrt', 'directml'], 'win32')).toBe('tensorrt');
  });

  it('keeps a chosen backend while it is available, and falls back to the best one when not', () => {
    expect(resolveInferenceBackend('directml', ['tensorrt', 'directml'], 'win32')).toBe('directml');
    expect(resolveInferenceBackend('migraphx', ['tensorrt', 'directml'], 'win32')).toBe('tensorrt');
  });

  it('describes the choices with their labels', () => {
    const info = describeInferenceBackends('ncnn', ['tensorrt', 'ncnn'], 'win32');
    expect(info).toMatchObject({ setting: 'ncnn', recommended: 'tensorrt', effective: 'ncnn' });
    expect(info.available.map(b => b.label)).toEqual(['TensorRT', 'NCNN Vulkan']);
  });
});

describe('the settings the dataset engine runs with', () => {
  it('hands vk_runtime.configure the backend, the registry map and the app paths', () => {
    const settings = engineRuntimeSettings('tensorrt', 'win32');
    expect(settings).toMatchObject({
      backend: 'tensorrt',
      backend_attrs: { tensorrt: 'TRT', directml: 'ORT_DML', ncnn: 'NCNN_VK', migraphx: 'MIGX' },
      trtexec_path: PATHS.TRTEXEC_SHIM,
      models_path: PATHS.MODELS,
      migraphx_driver_path: PATHS.MIGX_DRIVER_WIN,
    });
    // What cmd.exe needs to start the trtexec shim from vsmlrt's bare environment
    expect(Object.keys(settings.build_env).sort()).toEqual(['COMSPEC', 'SystemRoot']);
  });

  it('leaves the migraphx-driver to the chosen ROCm on Linux', () => {
    const settings = engineRuntimeSettings('ncnn', 'linux');
    expect(settings.migraphx_driver_path).toBeNull();
    expect(settings.build_env).toEqual({});
  });

  it('follows the saved setting, as JSON', () => {
    config.backend = process.platform === 'win32' ? 'directml' : 'ncnn';
    const value = JSON.parse(engineRuntimeEnvValue());
    expect(value.backend).toBe(currentInferenceBackends().effective);
    expect(value.backend).toBe(config.backend);
    config.backend = 'auto';
    expect(JSON.parse(engineRuntimeEnvValue()).backend).toBe(currentInferenceBackends().recommended);
  });
});
