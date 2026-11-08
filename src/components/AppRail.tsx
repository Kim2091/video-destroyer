// src/components/AppRail.tsx — the vertical tool rail down the left edge.
//
// 62px wide, one state: every entry is an icon over its own 10px label, so the
// rail is always readable without an expand mode to discover.

import { memo } from 'react';
import { Settings, Plug, Lock, LockOpen, Info, Terminal } from 'lucide-react';
import { Logo } from './Logo';

interface AppRailProps {
  privacyMode: boolean;
  showConsole: boolean;
  onToggleConsole: () => void;
  onSettingsClick: () => void;
  onPluginsClick: () => void;
  onTogglePrivacyMode: () => void;
  onAboutClick: () => void;
}

interface RailButtonProps {
  label: string;
  title: string;
  disabled?: boolean;
  active?: boolean;
  tone?: 'default' | 'warn';
  badge?: number;
  onClick: () => void;
  children: React.ReactNode;
}

const RailButton = ({
  label, title, disabled, active, tone = 'default', badge, onClick, children,
}: RailButtonProps) => (
  <button
    onClick={onClick}
    disabled={disabled}
    title={title}
    aria-pressed={active}
    className={`w-full h-[44px] rounded-lg flex flex-col items-center justify-center gap-[3px] transition-colors flex-shrink-0 relative
      focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500
      disabled:opacity-35 disabled:cursor-not-allowed ${
      active
        ? tone === 'warn'
          ? 'bg-warn-500/20 text-warn-400'
          : 'bg-accent-500/18 text-accent-400'
        : 'text-ink-400 hover:text-ink-200 hover:bg-ink-850'
    }`}
  >
    {children}
    <span className="text-[10px] font-medium leading-none whitespace-nowrap">
      {label}
    </span>
    {badge !== undefined && badge > 0 && (
      <span className="absolute top-1 right-1.5 min-w-[14px] h-[14px] px-[3px] rounded-full bg-accent-500 text-accent-ink text-[9px] font-bold font-display grid place-items-center leading-none">
        {badge}
      </span>
    )}
  </button>
);

export const AppRail = memo<AppRailProps>(({
  privacyMode,
  showConsole,
  onToggleConsole,
  onSettingsClick,
  onPluginsClick,
  onTogglePrivacyMode,
  onAboutClick,
}: AppRailProps) => {
  const ICON = 'w-[17px] h-[17px] flex-shrink-0';

  return (
    <div className="w-[62px] flex-shrink-0 bg-ink-900 border-r border-ink-800 flex flex-col items-stretch px-1 py-2.5 gap-1">
      <div className="h-7 mb-2 grid place-items-center flex-shrink-0 text-accent-500">
        <Logo className="w-[26px] h-[26px]" monochrome />
      </div>

      <RailButton label="Settings" title="Settings" onClick={onSettingsClick}>
        <Settings className={ICON} />
      </RailButton>

      <RailButton label="Plugins" title="Plugin dependencies" onClick={onPluginsClick}>
        <Plug className={ICON} />
      </RailButton>

      <RailButton
        label="Console"
        title={showConsole ? 'Hide the app log' : 'Show the app log'}
        active={showConsole}
        onClick={onToggleConsole}
      >
        <Terminal className={ICON} />
      </RailButton>

      <RailButton
        label="Privacy"
        title={privacyMode
          ? 'Privacy mode is on — click to show file names, logs and notifications'
          : 'Privacy mode is off — click to hide file names, logs and notifications'}
        active={privacyMode}
        tone="warn"
        onClick={onTogglePrivacyMode}
      >
        {privacyMode ? <Lock className={ICON} /> : <LockOpen className={ICON} />}
      </RailButton>

      <div className="flex-1" />

      <RailButton label="About" title="About Video Destroyer" onClick={onAboutClick}>
        <Info className={ICON} />
      </RailButton>
    </div>
  );
});
