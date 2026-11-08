// electron/inferenceBackend.ts
//
// The Default AI backend setting and the inference settings the dataset
// engine runs with. The engine reads them from VK_DATASET_RUNTIME once at
// start and hands them to vk_runtime.configure() (include/vk_dataset/
// runtime_settings.py), so a model node whose backend is "auto" follows the
// setting, and TensorRT engine builds find the app's trtexec shim.

import * as fs from 'fs';
import * as path from 'path';
import { configManager } from './configManager';
import { PATHS } from './constants';
import type { GpuVendor } from './gpuDetection';
import { isBackendId } from './providers/descriptors';
import { BACKENDS, getDefaultBackendForPlatform, getProvider, type BackendId } from './providers/registry';
import { getSystemRoot } from './trtexecShim';
import { getBackendsForVendor } from './vendorPackages';

/** 'auto' is the best backend available on this machine. */
export type InferenceBackendSetting = BackendId | 'auto';

/** The environment variable the dataset engine reads its inference settings from. */
export const RUNTIME_SETTINGS_ENV = 'VK_DATASET_RUNTIME';

export function parseInferenceBackendSetting(value: unknown): InferenceBackendSetting {
  return isBackendId(value) ? value : 'auto';
}

/** Whether a backend's plugin is in place: each provider names the files that prove it. */
export function isBackendInstalled(id: BackendId, exists: (file: string) => boolean = fs.existsSync): boolean {
  return getProvider(id).pluginHealthPaths().every(alternatives => alternatives.some(exists));
}

/**
 * The backends this machine can use, best first: the ones the plugin install
 * gives this GPU vendor on this OS (getBackendsForVendor), narrowed to those
 * whose plugin is actually installed. Before the plugins are installed none
 * is, and the vendor's whole list is returned so the setting can still be made.
 */
export function availableBackends(
  vendor: GpuVendor,
  platform: NodeJS.Platform = process.platform,
  exists: (file: string) => boolean = fs.existsSync,
): BackendId[] {
  const forVendor = getBackendsForVendor(vendor, platform);
  const installed = forVendor.filter(id => isBackendInstalled(id, exists));
  return installed.length > 0 ? installed : forVendor;
}

/** The backend a setting means here: the one chosen while it is available, else the best available. */
export function resolveInferenceBackend(
  setting: InferenceBackendSetting,
  available: readonly BackendId[],
  platform: NodeJS.Platform = process.platform,
): BackendId {
  if (setting !== 'auto' && available.includes(setting)) return setting;
  return available[0] ?? getDefaultBackendForPlatform(platform);
}

/** What vk_runtime.configure() takes, as the engine reads it from RUNTIME_SETTINGS_ENV. */
export interface EngineRuntimeSettings {
  backend: BackendId;
  backend_attrs: Record<string, string>;
  build_env: Record<string, string>;
  models_path: string;
  trtexec_path: string;
  migraphx_driver_path: string | null;
}

/**
 * Environment seeded into vsmlrt's TensorRT backend (`custom_env`). vsmlrt
 * spawns the engine builder with `{"CUDA_MODULE_LOADING": "LAZY"}` and nothing
 * else, which on Windows is not enough for cmd.exe to launch the .cmd shim.
 */
export function engineBuildEnv(platform: NodeJS.Platform = process.platform): Record<string, string> {
  if (platform !== 'win32') return {};
  const systemRoot = getSystemRoot();
  return { SystemRoot: systemRoot, COMSPEC: path.join(systemRoot, 'System32', 'cmd.exe') };
}

export function engineRuntimeSettings(backend: BackendId, platform: NodeJS.Platform = process.platform): EngineRuntimeSettings {
  return {
    backend,
    backend_attrs: Object.fromEntries(BACKENDS.map(descriptor => [descriptor.id, descriptor.vsmlrtBackendAttr])),
    build_env: engineBuildEnv(platform),
    models_path: PATHS.MODELS,
    trtexec_path: PATHS.TRTEXEC_SHIM,
    // On Linux vk_runtime finds the driver of the ROCm chosen in Settings.
    migraphx_driver_path: platform === 'win32' ? PATHS.MIGX_DRIVER_WIN : null,
  };
}

export interface InferenceBackendChoice {
  id: BackendId;
  label: string;
  description: string;
}

export interface InferenceBackendInfo {
  setting: InferenceBackendSetting;
  /** What 'auto', and the setting itself, resolve to on this machine */
  recommended: BackendId;
  effective: BackendId;
  available: InferenceBackendChoice[];
}

export function describeInferenceBackends(
  setting: InferenceBackendSetting,
  available: readonly BackendId[],
  platform: NodeJS.Platform = process.platform,
): InferenceBackendInfo {
  return {
    setting,
    recommended: resolveInferenceBackend('auto', available, platform),
    effective: resolveInferenceBackend(setting, available, platform),
    available: available.map(id => {
      const descriptor = getProvider(id).descriptor;
      return { id, label: descriptor.label, description: descriptor.description };
    }),
  };
}

/** The GPU vendor the installed plugins target, else the one detected. */
function machineVendor(): GpuVendor {
  return configManager.getPluginsGpuVendor() ?? configManager.getGpuVendor() ?? 'unknown';
}

/** The Default AI backend setting as it stands on this machine. */
export function currentInferenceBackends(): InferenceBackendInfo {
  return describeInferenceBackends(configManager.getInferenceBackend(), availableBackends(machineVendor()));
}

/** RUNTIME_SETTINGS_ENV's value for a dataset engine started now. */
export function engineRuntimeEnvValue(): string {
  return JSON.stringify(engineRuntimeSettings(currentInferenceBackends().effective));
}
