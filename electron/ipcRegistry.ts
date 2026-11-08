import { registerDatasetHandlers } from './datasetHandlers';
import { registerDatasetPreviewHandlers } from './datasetPreviewHandlers';
import { BrowserWindow } from 'electron';
import { registerDialogHandlers } from './dialogHandlers';
import { registerConfigHandlers } from './configHandlers';
import { registerDependencyHandlers } from './dependencyHandlers';
import { registerUpdateHandlers } from './updateHandlers';
import { DependencyManager } from './dependencyManager';
import { PluginInstaller } from './pluginInstaller';

/**
 * Central registry for all IPC handlers
 * Organizes handlers into logical groups
 */
export function registerAllIpcHandlers(
  mainWindow: BrowserWindow | null,
  dependencyManager: DependencyManager,
  pluginInstaller: PluginInstaller
) {
  // Register all handler groups
  registerDialogHandlers();
  registerConfigHandlers();
  registerDependencyHandlers(dependencyManager, pluginInstaller);
  registerUpdateHandlers();
  registerDatasetHandlers(mainWindow);
  registerDatasetPreviewHandlers();
}
