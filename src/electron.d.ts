// src/electron.d.ts
import type { BackendId } from '../electron/providers/descriptors';
import type { DatasetEvent } from '../electron/datasetProtocol';
import type {
  DatasetPreviewChunks, DatasetPreviewFrame, DatasetPreviewFrameRequest, DatasetPreviewLoaded,
} from '../electron/datasetPreviewProtocol';
import type { DatasetJob, DatasetNodeSpec, DatasetStage } from './utils/datasetGraph';

export type { BackendId };

export type DatasetOutcome<T> =
  | { success: true; value: T }
  | { success: false; error: string; cancelled?: boolean };

export type RocmMode = 'auto' | 'environment' | 'custom';

export interface RocmSetting {
  mode: RocmMode;
  customRoot?: string;
}

export interface RocmStatus {
  supported: boolean;
  root: string | null;
  migraphxVersion: string | null;
  hasMigraphxLibrary: boolean;
  hasMigraphxDriver: boolean;
  summary: string;
}

export interface RocmSettingResult {
  setting: RocmSetting;
  status: RocmStatus;
}

/** The Default AI backend setting (electron/inferenceBackend.ts). */
export type InferenceBackendSetting = BackendId | 'auto';

export interface InferenceBackendInfo {
  setting: InferenceBackendSetting;
  /** What 'auto' resolves to on this machine */
  recommended: BackendId;
  /** What the setting resolves to on this machine */
  effective: BackendId;
  available: { id: BackendId; label: string; description: string }[];
}

export interface ElectronAPI {
  // Dependency management
  platform: NodeJS.Platform;
  checkDependencies: () => Promise<boolean>;
  setupDependencies: () => Promise<InstallResult & { phase?: 'core' | 'plugins' }>;
  onSetupProgress: (callback: (progress: SetupProgress) => void) => () => void;
  detectCudaSupport: () => Promise<boolean>;
  getInferenceBackendInfo: () => Promise<{ hasCudaSupport: boolean; backend: BackendId }>;


  /** Dataset workspace: the vk_dataset engine's node types, cached for the session. */
  datasetGetNodes: () => Promise<DatasetOutcome<DatasetNodeSpec[]>>;
  /** The stages of a job, in order, with a sentence each; cached like the node types. */
  datasetGetStages: () => Promise<DatasetOutcome<DatasetStage[]>>;
  /** The example jobs in include/vk_dataset/presets, by their own name, else their file's. */
  datasetGetPresets: () => Promise<{ name: string; job: DatasetJob }[]>;
  /** The engine's own check of a job, without running it. */
  datasetCheck: (request: { job: DatasetJob; jobPath: string | null }) => Promise<DatasetOutcome<null>>;
  /** Runs a job; onDatasetProgress reports on it until this settles. One runs at a time. */
  datasetRun: (request: { job: DatasetJob; jobPath: string | null }) => Promise<DatasetOutcome<Record<string, unknown>>>;
  datasetCancel: () => Promise<{ success: boolean }>;
  /** Null when the dialog is cancelled. */
  datasetOpenJob: () => Promise<{ success: true; filePath: string; job: DatasetJob } | { success: false; error: string } | null>;
  /** Saves to filePath, or asks where when it is null. Null when the dialog is cancelled. */
  datasetSaveJob: (request: { job: DatasetJob; filePath: string | null }) =>
    Promise<{ success: true; filePath: string } | { success: false; error: string } | null>;
  datasetSelectPath: (mode: 'file' | 'folder') => Promise<string | null>;
  onDatasetProgress: (callback: (event: DatasetEvent) => void) => () => void;
  /** Starts the dataset preview if needed and loads the job into it, keeping what an edit left unchanged. */
  datasetPreviewLoad: (request: { job: DatasetJob; jobPath: string | null }) => Promise<DatasetOutcome<DatasetPreviewLoaded>>;
  datasetPreviewChunks: (request: { video: number; seed: number | null }) => Promise<DatasetOutcome<DatasetPreviewChunks>>;
  /** `superseded` marks a request a newer one replaced before it was rendered. */
  datasetPreviewFrame: (request: DatasetPreviewFrameRequest) => Promise<
    | { success: true; value: DatasetPreviewFrame }
    | { success: false; error: string; superseded?: boolean }
  >;
  datasetPreviewClose: () => Promise<{ success: boolean }>;

  selectFolder: () => Promise<string | null>;

  // Shell operations
  openExternal: (url: string) => Promise<void>;

  // App information
  getVersion: () => Promise<{ version: string }>;

  // Log file reading
  readLogTail: (maxLines?: number) => Promise<{ lines: string[]; hasNewContent: boolean; error?: string }>;
  resetLogCache: () => Promise<{ success: boolean }>;

  // Folder access
  openLogsFolder: () => Promise<{ success: boolean }>;
  openConfigFolder: () => Promise<{ success: boolean }>;
  openVSPluginsFolder: () => Promise<{ success: boolean }>;
  openVSScriptsFolder: () => Promise<{ success: boolean }>;

  // ROCm installation (Linux, MIGraphX backend)
  getRocmSetting: () => Promise<RocmSettingResult>;
  setRocmSetting: (setting: RocmSetting) => Promise<RocmSettingResult>;

  // The backend model nodes set to "auto" run on
  getInferenceBackend: () => Promise<InferenceBackendInfo>;
  setInferenceBackend: (setting: InferenceBackendSetting) => Promise<InferenceBackendInfo>;

  // Plugin dependency operations
  installPluginDependencies: (mode?: 'partial' | 'complete') => Promise<InstallResult>;
  retrySetupPlugins: () => Promise<InstallResult>;
  uninstallPluginDependencies: () => Promise<InstallResult>;
  checkPluginDependencies: () => Promise<{ installed: boolean; packages: string[] }>;
  cancelPluginDependencyInstall: () => Promise<{ success: boolean }>;
  onPluginDependencyProgress: (callback: (progress: PluginDependencyProgress) => void) => () => void;

  // Update operations
  checkForUpdates: () => Promise<{ success: boolean; data?: UpdateInfo; error?: string }>;
  openReleasesPage: () => Promise<{ success: boolean; error?: string }>;
  openReleaseUrl: (url: string) => Promise<{ success: boolean; error?: string }>;
}

/** How an install, uninstall or setup phase ended (electron/installFlow.ts). */
export interface InstallResult {
  success: boolean;
  /** summary plus where the full log is */
  error?: string;
  /** One sentence saying what went wrong and what to do */
  summary?: string;
  /** The output lines the summary was drawn from */
  evidence?: string;
  logPath?: string;
  cancelled?: boolean;
  alreadyRunning?: boolean;
  blocked?: boolean;
  warnings?: string[];
}

/** An install failure as the UI shows it: the sentence up front, the evidence behind "Details". */
export interface InstallFailureInfo {
  summary: string;
  evidence?: string;
  logPath?: string;
}

export interface SetupProgress {
  // 'retrying': an attempt failed and another is starting; 'warning': worth
  // showing, stops nothing. Only 'complete' and 'error' end a phase.
  type: 'download' | 'extract' | 'installing' | 'python-setup' | 'retrying' | 'warning' | 'complete' | 'error';
  component: string;
  progress: number;
  message: string;
  summary?: string;
  evidence?: string;
  logPath?: string;
  warnings?: string[];
}

export interface PluginDependencyProgress {
  type: 'installing' | 'retrying' | 'warning' | 'complete' | 'error';
  progress: number;
  message: string;
  summary?: string;
  evidence?: string;
  logPath?: string;
  warnings?: string[];
}

export interface UpdateInfo {
  /** Whether a release was looked up at all; false when no repository is configured */
  checked: boolean;
  available: boolean;
  currentVersion: string;
  latestVersion: string;
  releaseUrl: string;
  changelog: string;
  publishedAt: string;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}
