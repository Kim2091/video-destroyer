// electron/configManager.ts
import * as fs from 'fs-extra';
import * as path from 'path';
import { PATHS } from './constants';
import { logger } from './logger';
import { JsonFileWriter, readJsonOrSetAside } from './jsonFile';
import type { GpuVendor } from './gpuDetection';
import { DEFAULT_ROCM_SETTING, setRocmSetting, type RocmSetting } from './rocmEnvironment';
import type { BackendId } from './providers/descriptors';

interface AppConfig {
  /** GPU vendor detected on this machine, refreshed at every app mount. */
  gpuVendor?: GpuVendor;
  /**
   * GPU vendor the currently installed Python package set targets, written only
   * after a fully successful install. A difference from gpuVendor is exactly
   * the "needs reinstall" signal (GPU swap, driver install, ...).
   */
  pluginsGpuVendor?: GpuVendor;
  appVersion?: string;
  /** Which ROCm the MIGraphX backend runs against on Linux (rocmEnvironment.ts). */
  rocm?: RocmSetting;
  /** The Default AI backend model nodes set to "auto" run on (inferenceBackend.ts). */
  inferenceBackend?: BackendId | 'auto';
}

const CONFIG_FILE = path.join(PATHS.CONFIG, 'app-config.json');

const DEFAULT_CONFIG: AppConfig = {
  gpuVendor: undefined,
  pluginsGpuVendor: undefined,
};

/** A fresh copy: the defaults object must never become the live config and be mutated. */
const defaultConfig = (): AppConfig => structuredClone(DEFAULT_CONFIG);

export class ConfigManager {
  private config: AppConfig = defaultConfig();
  private readonly writer = new JsonFileWriter(() => CONFIG_FILE);
  /** Saves still in flight, so a reload reads what they wrote rather than racing them. */
  private pending = new Set<Promise<void>>();

  async load(): Promise<void> {
    await Promise.allSettled([...this.pending]);
    try {
      await fs.ensureDir(PATHS.CONFIG);
      const read = await readJsonOrSetAside<Record<string, unknown>>(CONFIG_FILE);
      switch (read.kind) {
        case 'ok':
          this.writer.unblock();
          this.config = this.deepMerge(defaultConfig() as unknown as Record<string, unknown>, read.value) as unknown as AppConfig;
          logger.info('Config loaded successfully');
          break;
        case 'missing':
          this.writer.unblock();
          logger.info('No config file found, using defaults (it is written by the first setting saved)');
          // Don't save here; the first setter writes the file
          break;
        case 'corrupt':
          // Starting from defaults is only safe once the old file is kept
          // somewhere: the first save would otherwise replace every setting the
          // user had with the defaults.
          this.config = defaultConfig();
          if (read.setAsideTo) {
            this.writer.unblock();
            logger.error(`Config file was not valid JSON; moved it to ${read.setAsideTo} and started from defaults:`, read.error);
          } else {
            this.writer.block('the config file could not be read or moved aside');
            logger.error('Config file was not valid JSON and could not be moved aside; not saving over it this session:', read.error);
          }
          break;
        case 'unreadable':
          this.config = defaultConfig();
          this.writer.block('the config file could not be read');
          logger.error('Config file could not be read; using defaults without saving over it this session:', read.error);
          break;
      }
    } catch (error) {
      logger.error('Error loading config:', error);
      this.config = defaultConfig();
      this.writer.block('loading the config failed');
    }
    setRocmSetting(this.config.rocm);
  }

  private deepMerge(
    base: Record<string, unknown>,
    override: Record<string, unknown>
  ): Record<string, unknown> {
    const result: Record<string, unknown> = { ...base };

    for (const [key, value] of Object.entries(override)) {
      const existing = result[key];
      if (this.isObject(existing) && this.isObject(value)) {
        result[key] = this.deepMerge(existing, value);
      } else {
        result[key] = value;
      }
    }

    return result;
  }

  private isObject(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
  }

  /**
   * Every setter saves the whole config, often several in a row, so the
   * writer takes this call's snapshot now and lets only the newest one stay.
   */
  async save(): Promise<void> {
    const write = this.writer.write(this.config)
      .then(() => logger.debug('Config saved successfully'))
      .catch(error => logger.error('Error saving config:', error));
    this.pending.add(write);
    try {
      await write;
    } finally {
      this.pending.delete(write);
    }
  }

  getRocmSetting(): RocmSetting {
    return this.config.rocm ?? DEFAULT_ROCM_SETTING;
  }

  async setRocmSetting(setting: RocmSetting): Promise<void> {
    this.config.rocm = setting;
    setRocmSetting(setting);
    await this.save();
  }

  getInferenceBackend(): BackendId | 'auto' {
    return this.config.inferenceBackend ?? 'auto';
  }

  async setInferenceBackend(backend: BackendId | 'auto'): Promise<void> {
    this.config.inferenceBackend = backend;
    await this.save();
  }

  getGpuVendor(): GpuVendor | undefined {
    return this.config.gpuVendor;
  }

  async setGpuVendor(vendor: GpuVendor): Promise<void> {
    this.config.gpuVendor = vendor;
    await this.save();
  }

  getPluginsGpuVendor(): GpuVendor | undefined {
    return this.config.pluginsGpuVendor;
  }

  async setPluginsGpuVendor(vendor: GpuVendor): Promise<void> {
    this.config.pluginsGpuVendor = vendor;
    await this.save();
  }

  getAppVersion(): string | undefined {
    return this.config.appVersion;
  }

  async setAppVersion(version: string): Promise<void> {
    this.config.appVersion = version;
    await this.save();
  }
}

export const configManager = new ConfigManager();
