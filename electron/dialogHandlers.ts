import { ipcMain, dialog, shell } from 'electron';
import * as fs from 'fs-extra';
import { logger } from './logger';
import { PATHS } from './constants';

/**
 * Registers all dialog-related IPC handlers
 */
export function registerDialogHandlers() {
  ipcMain.handle('select-folder', async () => {
    logger.info('Opening folder selection dialog');
    const result = await dialog.showOpenDialog({
      properties: ['openDirectory']
    });

    if (result.canceled || result.filePaths.length === 0) {
      logger.info('Folder selection canceled');
      return null;
    }

    logger.info(`Selected folder: ${result.filePaths[0]}`);
    return result.filePaths[0];
  });

  // Folder opening handlers
  ipcMain.handle('open-logs-folder', async () => {
    logger.info('Opening logs folder');
    try {
      const logsDir = logger.getLogsDir();
      await fs.ensureDir(logsDir);
      shell.openPath(logsDir);
      return { success: true };
    } catch (error) {
      logger.error('Error opening logs folder:', error);
      throw error;
    }
  });

  ipcMain.handle('open-config-folder', async () => {
    logger.info('Opening config folder');
    try {
      await fs.ensureDir(PATHS.CONFIG);
      shell.openPath(PATHS.CONFIG);
      return { success: true };
    } catch (error) {
      logger.error('Error opening config folder:', error);
      throw error;
    }
  });

  ipcMain.handle('open-vs-plugins-folder', async () => {
    logger.info('Opening VapourSynth plugins folder');
    try {
      await fs.ensureDir(PATHS.PLUGINS);
      shell.openPath(PATHS.PLUGINS);
      return { success: true };
    } catch (error) {
      logger.error('Error opening VS plugins folder:', error);
      throw error;
    }
  });

  ipcMain.handle('open-vs-scripts-folder', async () => {
    logger.info('Opening VapourSynth scripts folder');
    try {
      await fs.ensureDir(PATHS.SCRIPTS);
      shell.openPath(PATHS.SCRIPTS);
      return { success: true };
    } catch (error) {
      logger.error('Error opening VS scripts folder:', error);
      throw error;
    }
  });

  ipcMain.handle('open-external', async (event, url: string) => {
    logger.info(`Opening external URL: ${url}`);
    await shell.openExternal(url);
    return { success: true };
  });
}
