// src/components/AppModals.tsx - the app's modals, rendered from App.tsx

import { memo } from 'react';
import { SettingsModal } from './SettingsModal';
import { AboutModal } from './AboutModal';
import { PluginsModal } from './PluginsModal';
import { UpdateNotificationModal } from './UpdateNotificationModal';
import type { UpdateInfo } from '../electron';

interface AppModalsProps {
  // Settings
  showSettings: boolean;
  onCloseSettings: () => void;
  mainColor: string;
  onChangeMainColor: (color: string) => void;
  onResetMainColor: () => void;
  accentColor: string;
  onChangeAccentColor: (color: string) => void;
  onResetAccentColor: () => void;

  // About
  showAbout: boolean;
  onCloseAbout: () => void;

  // Plugins
  showPlugins: boolean;
  onClosePlugins: () => void;

  // Update
  showUpdateModal: boolean;
  updateInfo: UpdateInfo | null;
  onCloseUpdateModal: () => void;
}

export const AppModals = memo(function AppModals(props: AppModalsProps) {
  return (
    <>
      <SettingsModal
        show={props.showSettings}
        onClose={props.onCloseSettings}
        mainColor={props.mainColor}
        onChangeMainColor={props.onChangeMainColor}
        onResetMainColor={props.onResetMainColor}
        accentColor={props.accentColor}
        onChangeAccentColor={props.onChangeAccentColor}
        onResetAccentColor={props.onResetAccentColor}
      />

      <AboutModal
        show={props.showAbout}
        onClose={props.onCloseAbout}
      />

      <PluginsModal
        show={props.showPlugins}
        onClose={props.onClosePlugins}
      />

      {props.showUpdateModal && props.updateInfo && (
        <UpdateNotificationModal
          updateInfo={props.updateInfo}
          onClose={props.onCloseUpdateModal}
        />
      )}
    </>
  );
});
