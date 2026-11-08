// src/App.tsx - first-run setup, then the Dataset workspace with the rail,
// the console drawer and the app's modals around it.

import { useState, useCallback } from 'react';
import { NotificationContainer } from './components/NotificationContainer';
import { ConfirmDialogHost } from './components/ConfirmDialogHost';
import { AppModals } from './components/AppModals';
import { ConsoleDrawer } from './components/ConsoleDrawer';
import type { UpdateInfo } from './electron';
import { AppRail } from './components/AppRail';
import { DatasetWorkspace } from './components/dataset/DatasetWorkspace';
import { usePrivacyMode } from './hooks/usePrivacyMode';
import { useConsoleLog } from './hooks/useConsoleLog';
import { useSetup } from './hooks/useSetup';
import { useUIState } from './hooks/useUIState';
import { useAppEffects } from './hooks/useAppEffects';
import { SetupScreen } from './components/SetupScreen';
import { useAccentColor } from './hooks/useAccentColor';
import { useMainColor } from './hooks/useMainColor';

function App() {
  const { consoleOutput, consoleEndRef, addConsoleLog } = useConsoleLog();
  const { isSetupComplete, isCheckingDeps, hasCudaSupport, setupProgress, isSettingUp, handleSetup, pluginInstallError, setupError, setupWarnings, handleRetryPlugins, handleContinueWithoutPlugins } = useSetup(addConsoleLog);
  const { privacyMode, togglePrivacyMode } = usePrivacyMode();
  const { accentColor, setAccentColor, resetAccentColor } = useAccentColor();
  const { mainColor, setMainColor, resetMainColor } = useMainColor();
  const {
    showConsole,
    setShowConsole,
    showAbout,
    setShowAbout,
    showSettings,
    setShowSettings,
    showPlugins,
    setShowPlugins,
  } = useUIState();

  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
  const [showUpdateModal, setShowUpdateModal] = useState(false);

  // Update check, error handlers, focus recovery
  const { closeModalWithFocusRestore } = useAppEffects({
    isSetupComplete,
    addConsoleLog,
    setUpdateInfo,
    setShowUpdateModal,
  });

  // Stable identities so the memoised rail and drawer don't re-render on
  // every parent tick.
  const handleToggleConsole = useCallback(() => setShowConsole(!showConsole), [showConsole, setShowConsole]);
  const handleCloseConsole = useCallback(() => setShowConsole(false), [setShowConsole]);

  if (isCheckingDeps || !isSetupComplete) {
    return (
      <SetupScreen
        isCheckingDeps={isCheckingDeps}
        isSetupComplete={isSetupComplete}
        hasCudaSupport={hasCudaSupport}
        setupProgress={setupProgress}
        isSettingUp={isSettingUp}
        onSetup={handleSetup}
        pluginInstallError={pluginInstallError}
        setupError={setupError}
        setupWarnings={setupWarnings}
        onRetryPlugins={handleRetryPlugins}
        onContinueWithoutPlugins={handleContinueWithoutPlugins}
      />
    );
  }

  return (
    <div className="h-screen flex bg-ink-950 overflow-hidden">
      <NotificationContainer />
      <ConfirmDialogHost privacyMode={privacyMode} />

      <AppRail
        privacyMode={privacyMode}
        showConsole={showConsole}
        onToggleConsole={handleToggleConsole}
        onSettingsClick={() => setShowSettings(true)}
        onPluginsClick={() => setShowPlugins(true)}
        onTogglePrivacyMode={togglePrivacyMode}
        onAboutClick={() => setShowAbout(true)}
      />

      {/* The console opens over the workspace */}
      <div className="relative flex-1 min-w-0 flex">
        <DatasetWorkspace privacyMode={privacyMode} />
        <ConsoleDrawer
          open={showConsole}
          onClose={handleCloseConsole}
          consoleOutput={consoleOutput}
          consoleEndRef={consoleEndRef}
          privacyMode={privacyMode}
        />
      </div>

      <AppModals
        showSettings={showSettings}
        onCloseSettings={() => closeModalWithFocusRestore(() => setShowSettings(false))}
        mainColor={mainColor}
        onChangeMainColor={setMainColor}
        onResetMainColor={resetMainColor}
        accentColor={accentColor}
        onChangeAccentColor={setAccentColor}
        onResetAccentColor={resetAccentColor}
        showAbout={showAbout}
        onCloseAbout={() => closeModalWithFocusRestore(() => setShowAbout(false))}
        showPlugins={showPlugins}
        onClosePlugins={() => closeModalWithFocusRestore(() => setShowPlugins(false))}
        showUpdateModal={showUpdateModal}
        updateInfo={updateInfo}
        onCloseUpdateModal={() => closeModalWithFocusRestore(() => setShowUpdateModal(false))}
      />
    </div>
  );
}

export default App;
