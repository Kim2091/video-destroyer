import './asarFix'; // Must be first — fixes 7zip-bin path before any module imports 7zip-min
import { app } from 'electron';
import * as path from 'path';
import { logger } from './logger';
import { configManager } from './configManager';
import { WindowManager } from './windowManager';
import { registerAllIpcHandlers } from './ipcRegistry';
import { stopDatasetJob } from './datasetHandlers';
import { closeDatasetPreview } from './datasetPreviewHandlers';

// ============================================================================
// INITIALIZATION
// ============================================================================

const windowManager = new WindowManager();

// Keep Windows' portable install behaviour. AppImages are mounted read-only,
// so Linux must retain Electron's XDG-backed userData location.
if (app.isPackaged && process.platform === 'win32') {
  const portableUserDataPath = path.join(path.dirname(app.getPath('exe')), 'data', 'user-data');
  app.setPath('userData', portableUserDataPath);
  logger.info(`Using portable userData path: ${portableUserDataPath}`);
}

// Load config
configManager.load().then(() => {
  logger.setMainWindow(windowManager.getWindow());
  logger.info('Config loaded');
});

// ============================================================================
// APP LIFECYCLE
// ============================================================================

app.whenReady().then(() => {
  // Create the main window
  const mainWindow = windowManager.createWindow();
  const managers = windowManager.getManagers();

  // Register all IPC handlers
  if (managers) {
    registerAllIpcHandlers(
      mainWindow,
      managers.dependencyManager,
      managers.pluginInstaller
    );
    logger.info('All IPC handlers registered');
  }
});

// ============================================================================
// ERROR HANDLING
// ============================================================================

// Global error handler for main process
process.on('uncaughtException', (error) => {
  logger.error('Uncaught exception:', error);
  // Removed dialog.showErrorBox to prevent focus stealing issues
  // Critical errors are logged and can be reviewed in log files
});

process.on('unhandledRejection', (reason) => {
  logger.error('Unhandled rejection:', reason);
  // Removed dialog.showErrorBox to prevent focus stealing issues
  // Critical errors are logged and can be reviewed in log files
});

// ============================================================================
// APP LIFECYCLE EVENTS
// ============================================================================

app.on('window-all-closed', () => {
  logger.info('All windows closed');
  if (process.platform !== 'darwin') {
    logger.info('Quitting application');
    app.quit();
  }
});

app.on('before-quit', () => {
  logger.info('App quitting, cleaning up child processes');
  // Stop a running dataset job (python process)
  stopDatasetJob();
  // Stop the dataset preview server (python process)
  void closeDatasetPreview();
});

app.on('activate', () => {
  logger.info('App activated');
  if (windowManager.getWindow() === null) {
    windowManager.createWindow();
  }
});
