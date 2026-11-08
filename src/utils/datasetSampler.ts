// src/utils/datasetSampler.ts — editing a param marked `random`, which takes
// a fixed value or one of the engine's samplers (include/vk_dataset/params.py):
//
//   {"uniform": [low, high]}                  a float in [low, high]
//   {"int": [low, high]}                      an int in [low, high], both included
//   {"choice": [a, b], "weights": [3, 1]}     one of the values, weights optional

import type { DatasetParamSpec } from './datasetGraph';

export type SamplerMode = 'fixed' | 'uniform' | 'int' | 'choice';

const SAMPLER_KEYS = ['uniform', 'int', 'choice'] as const;

export function isSampler(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
    && SAMPLER_KEYS.some(key => key in (value as Record<string, unknown>));
}

/** The modes a param offers: an int param is never drawn from a float range, and a dropdown only picks. */
export function samplerModes(param: DatasetParamSpec): SamplerMode[] {
  if (param.kind === 'choice') return ['fixed', 'choice'];
  return param.kind === 'int' ? ['fixed', 'int', 'choice'] : ['fixed', 'uniform', 'int', 'choice'];
}

export function samplerModeOf(value: unknown): SamplerMode {
  if (!isSampler(value)) return 'fixed';
  if ('uniform' in value) return 'uniform';
  if ('int' in value) return 'int';
  return 'choice';
}

const finite = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

/** The numbers a value stands for: the value, a range's ends, or a list's options. */
function numbersOf(value: unknown): number[] {
  if (!isSampler(value)) {
    const n = finite(value);
    return n === null ? [] : [n];
  }
  const list = (value.uniform ?? value.int ?? value.choice) as unknown;
  return Array.isArray(list) ? list.map(finite).filter((n): n is number => n !== null) : [];
}

/**
 * Turns a value into the given mode, keeping what it can: a fixed value
 * becomes both ends of a range or a list of one; a range or list gives its
 * first number to a fixed value and its ends to the other forms.
 */
export function toSamplerMode(value: unknown, mode: SamplerMode, param: DatasetParamSpec): unknown {
  if (samplerModeOf(value) === mode) return value;
  if (param.kind === 'choice') {
    const choices = param.choices ?? [];
    if (mode === 'choice') return { choice: [...choices] };
    const picked = isSampler(value) && Array.isArray(value.choice) ? value.choice : [];
    return picked[0] ?? (isSampler(param.default) ? choices[0] : param.default) ?? choices[0];
  }
  const numbers = numbersOf(value);
  const fallback = finite(param.default) ?? finite(param.min) ?? 0;
  const low = numbers.length ? Math.min(...numbers) : fallback;
  const high = numbers.length ? Math.max(...numbers) : fallback;
  const asParam = (n: number) => (param.kind === 'int' ? Math.round(n) : n);
  switch (mode) {
    case 'fixed':
      return asParam(numbers[0] ?? fallback);
    case 'uniform':
      return { uniform: [low, high] };
    case 'int':
      return { int: [Math.round(low), Math.round(high)] };
    case 'choice':
      return { choice: numbers.length ? numbers.map(asParam) : [asParam(fallback)] };
  }
}

/** Reads "1, 2.5, 3" into numbers; int params take whole numbers only. */
export function parseNumberList(text: string, kind: string): { values: number[] } | { error: string } {
  const parts = text.split(/[,\s]+/).filter(Boolean);
  const values: number[] = [];
  for (const part of parts) {
    const n = Number(part);
    if (!Number.isFinite(n)) return { error: `"${part}" is not a number.` };
    if (kind === 'int' && !Number.isInteger(n)) return { error: `"${part}" is not a whole number.` };
    values.push(n);
  }
  return { values };
}

export function formatNumberList(values: unknown): string {
  return Array.isArray(values) ? values.map(String).join(', ') : '';
}

/** Sets the options of a choice sampler. Weights stay only while there is one per option. */
export function withChoices(value: Record<string, unknown>, choices: number[]): Record<string, unknown> {
  const weights = value.weights;
  return Array.isArray(weights) && weights.length === choices.length
    ? { choice: choices, weights }
    : { choice: choices };
}

/**
 * A dropdown's random pick: the ticked options in the dropdown's order, each
 * with its weight. Equal weights are left out, as they are the default.
 */
export function pickOf(options: unknown[], weights: number[]): Record<string, unknown> {
  return weights.every(w => w === weights[0]) ? { choice: options } : { choice: options, weights };
}

/** Sets or, given an empty list, removes a choice sampler's weights. */
export function withWeights(value: Record<string, unknown>, weights: number[]): Record<string, unknown> {
  return weights.length ? { choice: value.choice, weights } : { choice: value.choice };
}

/** Why the engine would refuse a sampler, or null when it would draw from it. */
export function samplerProblem(value: Record<string, unknown>, param: DatasetParamSpec): string | null {
  if ('uniform' in value || 'int' in value) {
    const range = (value.uniform ?? value.int) as unknown;
    if (!Array.isArray(range) || range.length !== 2 || range.some(n => finite(n) === null)) {
      return 'a range needs a low and a high number.';
    }
    if ((range[0] as number) > (range[1] as number)) return 'the low end of the range is above the high end.';
    if ('int' in value && param.kind === 'int' && range.some(n => !Number.isInteger(n))) {
      return 'an int range needs whole numbers.';
    }
    return null;
  }
  const choices = value.choice;
  if (!Array.isArray(choices) || choices.length === 0) return 'a list needs at least one option.';
  if (param.kind === 'choice') {
    const strays = choices.filter(c => !(param.choices ?? []).includes(c));
    if (strays.length) return `${strays.join(', ')} is not one of the options.`;
  }
  const weights = value.weights;
  if (weights !== undefined) {
    if (!Array.isArray(weights) || weights.length !== choices.length) return 'a list needs one weight per option.';
    if (weights.some(w => finite(w) === null || (w as number) < 0)) return 'weights must be numbers of 0 or more.';
    if (!weights.some(w => (w as number) > 0)) return 'at least one weight must be above 0.';
  }
  return null;
}

const short = (n: unknown) => (typeof n === 'number' ? String(Math.round(n * 1000) / 1000) : String(n));

/** A few words for a node card: "0.75–1", "int 2–6", "one of 3". */
export function describeValue(value: unknown): string {
  if (!isSampler(value)) {
    if (value === null || value === undefined || value === '') return '—';
    if (typeof value === 'object') return Array.isArray(value) ? `${value.length} items` : 'custom';
    return short(value);
  }
  const range = (value.uniform ?? value.int) as unknown;
  if (Array.isArray(range)) return `${'int' in value ? 'int ' : ''}${short(range[0])}–${short(range[1])}`;
  const choices = value.choice;
  return Array.isArray(choices) ? `one of ${choices.length}` : 'random';
}
