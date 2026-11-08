import { ipcMain, shell } from 'electron';
import { checkForUpdates, getReleasesPageUrl } from './updateChecker';
import { logger } from './logger';

/**
 * Registers IPC handlers for update operations
 */
export function registerUpdateHandlers(): void {
  // Check for updates
  ipcMain.handle('check-for-updates', async () => {
    try {
      logger.info('IPC: check-for-updates called');
      const updateInfo = await checkForUpdates();
      return { success: true, data: updateInfo };
    } catch (error) {
      logger.error('Error checking for updates:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  });

  // Open releases page in browser
  ipcMain.handle('open-releases-page', async () => {
    try {
      logger.info('IPC: open-releases-page called');
      const url = getReleasesPageUrl();
      if (!url) return { success: false, error: 'No update repository is configured' };
      await shell.openExternal(url);
      return { success: true };
    } catch (error) {
      logger.error('Error opening releases page:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  });

  // Open specific release URL
  ipcMain.handle('open-release-url', async (_event, url: string) => {
    try {
      logger.info(`IPC: open-release-url called with URL: ${url}`);
      await shell.openExternal(url);
      return { success: true };
    } catch (error) {
      logger.error('Error opening release URL:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      };
    }
  });
  
  logger.info('Update handlers registered');
}
