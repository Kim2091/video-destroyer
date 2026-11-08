// electron/preload.ts
import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
  // Dependency management
  platform: process.platform,
  checkDependencies: () => ipcRenderer.invoke('check-dependencies'),
  detectCudaSupport: () => ipcRenderer.invoke('detect-cuda-support'),
  getInferenceBackendInfo: () => ipcRenderer.invoke('get-inference-backend-info'),
  setupDependencies: () => ipcRenderer.invoke('setup-dependencies'),
  onSetupProgress: (callback: (progress: any) => void) => {
    const listener = (event: any, progress: any) => callback(progress);
    ipcRenderer.on('setup-progress', listener);
    return () => ipcRenderer.removeListener('setup-progress', listener);
  },

  // Dataset workspace (include/vk_dataset)
  datasetGetNodes: () => ipcRenderer.invoke('dataset-get-nodes'),
  datasetGetStages: () => ipcRenderer.invoke('dataset-get-stages'),
  datasetGetPresets: () => ipcRenderer.invoke('dataset-get-presets'),
  datasetCheck: (request: { job: unknown; jobPath: string | null }) => ipcRenderer.invoke('dataset-check', request),
  datasetRun: (request: { job: unknown; jobPath: string | null }) => ipcRenderer.invoke('dataset-run', request),
  datasetCancel: () => ipcRenderer.invoke('dataset-cancel'),
  datasetOpenJob: () => ipcRenderer.invoke('dataset-open-job'),
  datasetSaveJob: (request: { job: unknown; filePath: string | null }) => ipcRenderer.invoke('dataset-save-job', request),
  datasetSelectPath: (mode: 'file' | 'folder') => ipcRenderer.invoke('dataset-select-path', mode),
  onDatasetProgress: (callback: (event: unknown) => void) => {
    const listener = (_event: unknown, event: unknown) => callback(event);
    ipcRenderer.on('dataset-progress', listener);
    return () => { ipcRenderer.removeListener('dataset-progress', listener); };
  },
  datasetPreviewLoad: (request: { job: unknown; jobPath: string | null }) => ipcRenderer.invoke('dataset-preview-load', request),
  datasetPreviewChunks: (request: { video: number; seed: number | null }) => ipcRenderer.invoke('dataset-preview-chunks', request),
  datasetPreviewFrame: (request: unknown) => ipcRenderer.invoke('dataset-preview-frame', request),
  datasetPreviewClose: () => ipcRenderer.invoke('dataset-preview-close'),

  selectFolder: () => ipcRenderer.invoke('select-folder'),

  // Shell operations
  openExternal: (url: string) => ipcRenderer.invoke('open-external', url),

  // App information
  getVersion: () => ipcRenderer.invoke('get-version'),

  // Log file reading (efficient polling-based console)
  readLogTail: (maxLines?: number) => ipcRenderer.invoke('read-log-tail', maxLines),
  resetLogCache: () => ipcRenderer.invoke('reset-log-cache'),

  // Folder access
  openLogsFolder: () => ipcRenderer.invoke('open-logs-folder'),
  openConfigFolder: () => ipcRenderer.invoke('open-config-folder'),
  openVSPluginsFolder: () => ipcRenderer.invoke('open-vs-plugins-folder'),
  openVSScriptsFolder: () => ipcRenderer.invoke('open-vs-scripts-folder'),

  // ROCm installation (Linux, MIGraphX backend)
  getRocmSetting: () => ipcRenderer.invoke('get-rocm-setting'),
  setRocmSetting: (setting: { mode: string; customRoot?: string }) => ipcRenderer.invoke('set-rocm-setting', setting),
  getInferenceBackend: () => ipcRenderer.invoke('get-inference-backend'),
  setInferenceBackend: (setting: string) => ipcRenderer.invoke('set-inference-backend', setting),

  // Plugin dependency operations
  installPluginDependencies: (mode?: 'partial' | 'complete') => ipcRenderer.invoke('install-plugin-dependencies', mode),
  retrySetupPlugins: () => ipcRenderer.invoke('retry-setup-plugins'),
  uninstallPluginDependencies: () => ipcRenderer.invoke('uninstall-plugin-dependencies'),
  checkPluginDependencies: () => ipcRenderer.invoke('check-plugin-dependencies'),
  cancelPluginDependencyInstall: () => ipcRenderer.invoke('cancel-plugin-dependency-install'),
  onPluginDependencyProgress: (callback: (progress: any) => void) => {
    const listener = (event: any, progress: any) => callback(progress);
    ipcRenderer.on('plugin-dependency-progress', listener);
    return () => ipcRenderer.removeListener('plugin-dependency-progress', listener);
  },

  // Update operations
  checkForUpdates: () => ipcRenderer.invoke('check-for-updates'),
  openReleasesPage: () => ipcRenderer.invoke('open-releases-page'),
  openReleaseUrl: (url: string) => ipcRenderer.invoke('open-release-url', url),
});

