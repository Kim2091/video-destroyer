import { ipcMain, app } from 'electron';
import * as fs from 'fs-extra';
import { logger } from './logger';
import { configManager } from './configManager';
import { describeRocm, type RocmMode, type RocmSetting } from './rocmEnvironment';
import { currentInferenceBackends, parseInferenceBackendSetting } from './inferenceBackend';
import { closeDatasetPreview } from './datasetPreviewHandlers';

// Cache for log file reading - stores last file size for change detection
let lastLogSize = 0;

/**
 * Registers all configuration-related IPC handlers
 */
export function registerConfigHandlers() {
  // Which ROCm the MIGraphX backend runs against (Linux; rocmEnvironment.ts).
  // Both return the setting with what it resolves to, for the status line.
  ipcMain.handle('get-rocm-setting', async () => {
    const setting = configManager.getRocmSetting();
    return { setting, status: describeRocm(setting) };
  });

  ipcMain.handle('set-rocm-setting', async (event, setting: RocmSetting) => {
    const mode: RocmMode = ['auto', 'environment', 'custom'].includes(setting?.mode) ? setting.mode : 'auto';
    const customRoot = typeof setting?.customRoot === 'string' ? setting.customRoot : undefined;
    const clean: RocmSetting = { mode, ...(customRoot ? { customRoot } : {}) };
    logger.info(`Setting ROCm installation: ${mode}${customRoot ? ` (${customRoot})` : ''}`);
    await configManager.setRocmSetting(clean);
    return { setting: clean, status: describeRocm(clean) };
  });

  // The Default AI backend: the setting, what it and 'auto' resolve to here,
  // and the backends this machine can choose from.
  ipcMain.handle('get-inference-backend', async () => currentInferenceBackends());

  ipcMain.handle('set-inference-backend', async (_event, setting: unknown) => {
    const clean = parseInferenceBackendSetting(setting);
    logger.info(`Setting the default AI backend: ${clean}`);
    await configManager.setInferenceBackend(clean);
    // A running preview read the old setting when it started; the next
    // preview load starts a new one with this setting.
    await closeDatasetPreview();
    return currentInferenceBackends();
  });

  ipcMain.handle('get-version', async () => {
    return { version: app.getVersion() };
  });

  // Read the last N lines from the log file (efficient tail reading)
  ipcMain.handle('read-log-tail', async (event, maxLines: number = 300) => {
    try {
      const logPath = logger.getLogPath();
      
      // Check if file exists
      if (!await fs.pathExists(logPath)) {
        return { lines: [], hasNewContent: false };
      }
      
      const stats = await fs.stat(logPath);
      const currentSize = stats.size;
      
      // Check if file has changed since last read
      const hasNewContent = currentSize !== lastLogSize;
      
      if (!hasNewContent) {
        // File hasn't changed, return empty to signal no update needed
        return { lines: [], hasNewContent: false };
      }
      
      lastLogSize = currentSize;
      
      // Read the file efficiently - only read last portion if file is large
      const MAX_BYTES_TO_READ = 512 * 1024; // 512KB max
      let content: string;
      
      if (currentSize > MAX_BYTES_TO_READ) {
        // Read only the last portion of the file
        const fd = await fs.open(logPath, 'r');
        const buffer = Buffer.alloc(MAX_BYTES_TO_READ);
        const startPosition = currentSize - MAX_BYTES_TO_READ;
        await fs.read(fd, buffer, 0, MAX_BYTES_TO_READ, startPosition);
        await fs.close(fd);
        content = buffer.toString('utf-8');
        // Remove potentially incomplete first line
        const firstNewline = content.indexOf('\n');
        if (firstNewline > 0) {
          content = content.substring(firstNewline + 1);
        }
      } else {
        content = await fs.readFile(logPath, 'utf-8');
      }
      
      // Split into lines and take last N
      const allLines = content.split('\n').filter(line => line.trim().length > 0);
      const lines = allLines.slice(-maxLines);
      
      return { lines, hasNewContent: true };
    } catch (error) {
      // Don't log errors here to avoid infinite loops
      return { lines: [], hasNewContent: false, error: String(error) };
    }
  });

  // Reset log read cache (call when user wants to force refresh)
  ipcMain.handle('reset-log-cache', async () => {
    lastLogSize = 0;
    return { success: true };
  });
}
