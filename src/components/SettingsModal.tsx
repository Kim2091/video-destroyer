import { memo } from 'react';
import { Settings, Terminal, FolderOpen, Package, FileCode, Palette } from 'lucide-react';
import { RocmSettingsSection } from './RocmSettingsSection';
import { InferenceBackendSettingsSection } from './InferenceBackendSettingsSection';
import { DEFAULT_ACCENT_COLOR } from '../hooks/useAccentColor';
import { DEFAULT_MAIN_COLOR } from '../hooks/useMainColor';
import { ModalSectionHeader } from './ModalSectionHeader';
import { ModalShell } from './ModalShell';

interface SettingsModalProps {
  show: boolean;
  onClose: () => void;
  mainColor: string;
  onChangeMainColor: (color: string) => void;
  onResetMainColor: () => void;
  accentColor: string;
  onChangeAccentColor: (color: string) => void;
  onResetAccentColor: () => void;
}

export const SettingsModal = memo<SettingsModalProps>(({
  show,
  onClose,
  mainColor,
  onChangeMainColor,
  onResetMainColor,
  accentColor,
  onChangeAccentColor,
  onResetAccentColor,
}) => {
  if (!show) return null;

  const handleOpenLogsFolder = async (): Promise<void> => {
    try {
      await window.electronAPI.openLogsFolder();
    } catch (error) {
      console.error('Error opening logs folder:', error);
    }
  };

  const handleOpenConfigFolder = async (): Promise<void> => {
    try {
      await window.electronAPI.openConfigFolder();
    } catch (error) {
      console.error('Error opening config folder:', error);
    }
  };

  const handleOpenVSPluginsFolder = async (): Promise<void> => {
    try {
      await window.electronAPI.openVSPluginsFolder();
    } catch (error) {
      console.error('Error opening VS plugins folder:', error);
    }
  };

  const handleOpenVSScriptsFolder = async (): Promise<void> => {
    try {
      await window.electronAPI.openVSScriptsFolder();
    } catch (error) {
      console.error('Error opening VS scripts folder:', error);
    }
  };

  return (
    <ModalShell
      title="Settings"
      icon={Settings}
      onClose={onClose}
      closeLabel="Close settings"
      className="max-w-2xl w-full max-h-[90vh] overflow-hidden flex flex-col"
    >
      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {/* The backend model nodes set to "auto" run on */}
        <InferenceBackendSettingsSection />

        {/* ROCm install for the MIGraphX backend (Linux only; renders nothing elsewhere) */}
        <RocmSettingsSection />

        {/* Accent Color Section */}
        <section className="mt-2 border-t border-ink-700">
          <ModalSectionHeader icon={Palette} title="Appearance" />

          <div className="flex items-center gap-3 px-4 py-2.5 border-b border-ink-900">
            <label htmlFor="main-color" className="flex-1 min-w-0 cursor-pointer">
              <p className="text-[12.5px] text-ink-200">Main Color</p>
              <p className="text-[11px] text-ink-500 mt-0.5">Tint the interface surfaces while keeping the application dark.</p>
            </label>
            <input
              id="main-color"
              type="color"
              value={mainColor}
              onChange={(event) => onChangeMainColor(event.target.value)}
              aria-label="Choose main color"
              className="w-9 h-7 p-0.5 rounded bg-ink-850 border border-ink-750 cursor-pointer"
            />
            <code className="w-[68px] text-[11px] font-mono text-ink-400 uppercase">{mainColor}</code>
            <button
              type="button"
              onClick={onResetMainColor}
              disabled={mainColor === DEFAULT_MAIN_COLOR}
              className="text-[11px] text-accent-400 hover:text-accent-300 transition-colors disabled:text-ink-600 disabled:cursor-not-allowed"
            >
              Reset
            </button>
          </div>

          <div className="flex items-center gap-3 px-4 py-2.5 border-b border-ink-900">
            <label htmlFor="accent-color" className="flex-1 min-w-0 cursor-pointer">
              <p className="text-[12.5px] text-ink-200">Accent Color</p>
              <p className="text-[11px] text-ink-500 mt-0.5">Customize highlights, controls, and active states throughout Video Destroyer.</p>
            </label>
            <input
              id="accent-color"
              type="color"
              value={accentColor}
              onChange={(event) => onChangeAccentColor(event.target.value)}
              aria-label="Choose accent color"
              className="w-9 h-7 p-0.5 rounded bg-ink-850 border border-ink-750 cursor-pointer"
            />
            <code className="w-[68px] text-[11px] font-mono text-ink-400 uppercase">{accentColor}</code>
            <button
              type="button"
              onClick={onResetAccentColor}
              disabled={accentColor === DEFAULT_ACCENT_COLOR}
              className="text-[11px] text-accent-400 hover:text-accent-300 transition-colors disabled:text-ink-600 disabled:cursor-not-allowed"
            >
              Reset
            </button>
          </div>
        </section>

        {/* VapourSynth Folders Section */}
        <section className="mt-2 border-t border-ink-700">
          <ModalSectionHeader icon={Package} title="VapourSynth" />

          {/* Open VS Plugins Folder */}
          <button
            onClick={handleOpenVSPluginsFolder}
            className="w-full h-8 flex items-center gap-2.5 px-4 border-b border-ink-900 text-left text-[12.5px] text-ink-300 hover:bg-ink-850 hover:text-ink-200 transition-colors group"
          >
            <Package className="w-4 h-4 text-ink-500 group-hover:text-ink-400 transition-colors" />
            <span className="flex-1 truncate">VS Plugins</span>
            <FolderOpen className="w-3.5 h-3.5 text-ink-600 group-hover:text-ink-400 transition-colors" />
          </button>

          {/* Open VS Scripts Folder */}
          <button
            onClick={handleOpenVSScriptsFolder}
            className="w-full h-8 flex items-center gap-2.5 px-4 border-b border-ink-900 text-left text-[12.5px] text-ink-300 hover:bg-ink-850 hover:text-ink-200 transition-colors group"
          >
            <FileCode className="w-4 h-4 text-ink-500 group-hover:text-ink-400 transition-colors" />
            <span className="flex-1 truncate">VS Scripts</span>
            <FolderOpen className="w-3.5 h-3.5 text-ink-600 group-hover:text-ink-400 transition-colors" />
          </button>
        </section>

        {/* Application Folders Section */}
        <section className="mt-2 border-t border-ink-700">
          <ModalSectionHeader icon={FolderOpen} title="Application Folders" />

          {/* Open Config Folder */}
          <button
            onClick={handleOpenConfigFolder}
            className="w-full h-8 flex items-center gap-2.5 px-4 border-b border-ink-900 text-left text-[12.5px] text-ink-300 hover:bg-ink-850 hover:text-ink-200 transition-colors group"
          >
            <Settings className="w-4 h-4 text-ink-500 group-hover:text-ink-400 transition-colors" />
            <span className="flex-1 truncate">Config</span>
            <FolderOpen className="w-3.5 h-3.5 text-ink-600 group-hover:text-ink-400 transition-colors" />
          </button>

          {/* Open Logs Folder */}
          <button
            onClick={handleOpenLogsFolder}
            className="w-full h-8 flex items-center gap-2.5 px-4 border-b border-ink-900 text-left text-[12.5px] text-ink-300 hover:bg-ink-850 hover:text-ink-200 transition-colors group"
          >
            <Terminal className="w-4 h-4 text-ink-500 group-hover:text-ink-400 transition-colors" />
            <span className="flex-1 truncate">Logs</span>
            <FolderOpen className="w-3.5 h-3.5 text-ink-600 group-hover:text-ink-400 transition-colors" />
          </button>
        </section>
      </div>

      {/* Footer */}
      <div className="h-9 flex-shrink-0 flex items-center justify-end gap-2 px-4 border-t border-ink-800 bg-ink-900 text-[11px] text-ink-500">
        <kbd className="px-1.5 py-0.5 bg-ink-850 border border-ink-750 rounded text-[10px] font-mono text-ink-300">Esc</kbd>
        <span>to close</span>
      </div>
    </ModalShell>
  );
});
