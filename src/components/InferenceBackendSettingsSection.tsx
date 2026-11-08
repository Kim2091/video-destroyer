import { memo, useCallback, useEffect, useState } from 'react';
import { Zap } from 'lucide-react';
import type { InferenceBackendInfo, InferenceBackendSetting } from '../electron.d';
import { ModalSectionHeader } from './ModalSectionHeader';
import { backendChoiceLabel } from '../utils/inferenceBackend';

/**
 * The Default AI backend: what model nodes set to "auto" run on
 * (electron/inferenceBackend.ts). Lists only the backends this machine can
 * use, and loads and saves itself.
 */
export const InferenceBackendSettingsSection = memo(() => {
  const [info, setInfo] = useState<InferenceBackendInfo | null>(null);

  useEffect(() => {
    let cancelled = false;
    window.electronAPI.getInferenceBackend()
      .then(loaded => { if (!cancelled) setInfo(loaded); })
      .catch(() => { /* the section just stays hidden */ });
    return () => { cancelled = true; };
  }, []);

  const save = useCallback(async (setting: InferenceBackendSetting) => {
    setInfo(await window.electronAPI.setInferenceBackend(setting));
  }, []);

  if (!info || info.available.length === 0) return null;
  const chosen = info.available.find(backend => backend.id === info.effective);

  return (
    <section className="mt-2 border-t border-ink-700">
      <ModalSectionHeader icon={Zap} title="AI Models" />

      <div className="flex items-center gap-3 px-4 py-2.5 border-b border-ink-900">
        <label htmlFor="inference-backend" className="flex-1 min-w-0 cursor-pointer">
          <p className="text-[12.5px] text-ink-200">Default AI backend</p>
          <p className="text-[11px] text-ink-500 mt-0.5">
            Model nodes set to &quot;auto&quot; run on this; a backend chosen on a node wins. A run or preview starts on the backend chosen here.
          </p>
        </label>
        <select
          id="inference-backend"
          value={info.setting}
          onChange={(event) => void save(event.target.value as InferenceBackendSetting)}
          className="h-7 px-2 rounded bg-ink-850 border border-ink-750 text-[12px] text-ink-200"
        >
          <option value="auto">{backendChoiceLabel('auto', info)}</option>
          {info.available.map(backend => (
            <option key={backend.id} value={backend.id}>{backendChoiceLabel(backend.id, info)}</option>
          ))}
        </select>
      </div>
      {chosen && (
        <p className="px-4 py-2 border-b border-ink-900 text-[11px] text-ink-500">{chosen.description}</p>
      )}
    </section>
  );
});
