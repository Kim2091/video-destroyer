// src/utils/inferenceBackend.ts — how the Default AI backend setting names its choices.

import type { InferenceBackendInfo, InferenceBackendSetting } from '../electron.d';

/** "Auto (recommended: TensorRT)" for auto, else the backend's own name. */
export function backendChoiceLabel(setting: InferenceBackendSetting, info: InferenceBackendInfo): string {
  const label = (id: string) => info.available.find(backend => backend.id === id)?.label ?? id;
  return setting === 'auto' ? `Auto (recommended: ${label(info.recommended)})` : label(setting);
}
