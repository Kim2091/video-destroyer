import { describe, expect, it } from 'vitest';
import { backendChoiceLabel } from './inferenceBackend';
import type { InferenceBackendInfo } from '../electron.d';

const info: InferenceBackendInfo = {
  setting: 'auto',
  recommended: 'tensorrt',
  effective: 'tensorrt',
  available: [
    { id: 'tensorrt', label: 'TensorRT', description: '' },
    { id: 'directml', label: 'DirectML', description: '' },
  ],
};

describe('backendChoiceLabel', () => {
  it('names the backend auto resolves to', () => {
    expect(backendChoiceLabel('auto', info)).toBe('Auto (recommended: TensorRT)');
  });

  it('names a chosen backend by its label', () => {
    expect(backendChoiceLabel('directml', info)).toBe('DirectML');
  });
});
