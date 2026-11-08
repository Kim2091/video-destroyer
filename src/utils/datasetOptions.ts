// src/utils/datasetOptions.ts — the options a Random Pick style setting
// (kind "options") picks from: a choice sampler {"choice": [...], "weights":
// [...]} as params.py reads it, typed as a list when the node feeds no
// dropdown.

import { isSampler } from './datasetSampler';

export interface PickOptions {
  choice: unknown[];
  /** One weight per option, or null for equal odds. */
  weights: number[] | null;
}

/** The options and weights a setting holds; anything else reads as no options. */
export function readOptions(value: unknown): PickOptions {
  if (!isSampler(value) || !Array.isArray(value.choice)) return { choice: [], weights: null };
  const weights = Array.isArray(value.weights) && value.weights.length === value.choice.length
    && value.weights.every(w => typeof w === 'number') ? (value.weights as number[]) : null;
  return { choice: [...value.choice], weights };
}

/** The sampler to save: weights are kept only while they differ, since equal weights are the default. */
export function writeOptions({ choice, weights }: PickOptions): Record<string, unknown> {
  const differ = !!weights && weights.length === choice.length && weights.some(w => w !== weights[0]);
  return differ ? { choice, weights } : { choice };
}

/** A typed list, "a, b, 0.5": numbers where they read as numbers, text otherwise. */
export function parseOptionList(text: string): unknown[] {
  return text.split(',').map(part => part.trim()).filter(Boolean).map(part => {
    const n = Number(part);
    return Number.isFinite(n) ? n : part;
  });
}

export const formatOptionList = (values: unknown[]) => values.map(String).join(', ');
