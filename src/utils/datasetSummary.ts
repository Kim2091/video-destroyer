// src/utils/datasetSummary.ts — a node's key settings in a few words for its
// card: "Scale 0.5", "bicubic / bilinear / area", "h264 23–33 (60%) · mpeg2
// (40%)", a random setting as its range or its options, and how often a node
// with a probability runs. Built from the declared kinds alone, so no node
// type is named here.

import type { DatasetNodeSpec, DatasetParamSpec } from './datasetGraph';
import { isSampler } from './datasetSampler';

export interface SettingSummary {
  name: string;
  label: string;
  text: string;
  /** A required setting that is still empty, such as a source's file. */
  missing?: boolean;
}

const short = (n: number) => String(Math.round(n * 1000) / 1000);
const isNumber = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const isPair = (v: unknown): v is [number, number] => Array.isArray(v) && v.length === 2 && v.every(isNumber);
const span = ([low, high]: [number, number]) => (low === high ? short(low) : `${short(low)}–${short(high)}`);
const percent = (share: number) => `${Math.round(share * 100)}%`;
const fileName = (path: string) => path.split(/[\\/]/).filter(Boolean).pop() ?? path;

/** Options joined with slashes, the first few only. */
function options(values: unknown[], limit = 4): string {
  const shown = values.slice(0, limit).map(v => (isNumber(v) ? short(v) : String(v)));
  return shown.join(' / ') + (values.length > limit ? ` +${values.length - limit}` : '');
}

/** A sampler: its range, or its options with their shares when weighted. */
function samplerText(value: Record<string, unknown>): string {
  const range = (value.uniform ?? value.int) as unknown;
  if (isPair(range)) return span(range);
  const choices = Array.isArray(value.choice) ? value.choice : [];
  const weights = Array.isArray(value.weights) && value.weights.length === choices.length && value.weights.every(isNumber)
    ? (value.weights as number[]) : null;
  const total = weights?.reduce((a, b) => a + b, 0) ?? 0;
  if (!weights || total <= 0) return options(choices);
  return choices.map((c, i) => `${isNumber(c) ? short(c) : String(c)} (${percent(weights[i] / total)})`).join(' / ');
}

/**
 * A table of named options, each with a probability, as a list of the ones
 * that can be picked: "h264 23–33 (60%) · mpeg2 6000–10000 (40%)".
 */
function weightedTable(value: Record<string, unknown>): string | null {
  const entries = Object.entries(value);
  if (!entries.length || !entries.every(([, v]) => v && typeof v === 'object' && !Array.isArray(v) && isNumber((v as Record<string, unknown>).probability))) {
    return null;
  }
  const picked = entries
    .map(([name, v]) => ({ name, row: v as Record<string, unknown>, share: (v as Record<string, unknown>).probability as number }))
    .filter(e => e.share > 0)
    .sort((a, b) => b.share - a.share);
  if (!picked.length) return 'none';
  const total = picked.reduce((a, e) => a + e.share, 0);
  return picked.map(({ name, row, share }) => {
    const range = Object.entries(row).find(([key, v]) => key.endsWith('_range') && isPair(v))?.[1] as [number, number] | undefined;
    const odds = picked.length > 1 ? ` (${percent(share / total)})` : '';
    return `${name}${range ? ` ${span(range)}` : ''}${odds}`;
  }).join(' · ');
}

/** A setting's value in a few words, or null when there is nothing worth showing. */
export function formatSetting(param: DatasetParamSpec, value: unknown, privacy = false): string | null {
  if (isSampler(value)) return samplerText(value);
  if (value === null || value === undefined) return null;
  switch (typeof value) {
    case 'number':
      return short(value);
    case 'boolean':
      return value ? 'on' : 'off';
    case 'string':
      if (param.kind === 'path') return value ? (privacy ? 'set' : fileName(value)) : 'not set';
      if (param.kind === 'code') return value ? 'edited' : null;
      return value || null;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) return 'none';
    if (isPair(value) && param.name.endsWith('_range')) return span(value);
    return value.every(v => typeof v !== 'object' || v === null) ? options(value) : `${value.length} items`;
  }
  const table = value as Record<string, unknown>;
  const weighted = weightedTable(table);
  if (weighted) return weighted;
  const flat = Object.entries(table);
  if (flat.length && flat.every(([, v]) => typeof v !== 'object' || v === null)) {
    return flat.map(([k, v]) => `${k} ${String(v)}`).join(' · ');
  }
  return 'custom';
}

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

/**
 * The settings a card shows, at most `limit`, in the order declared. Files
 * and folders come first in priority, then settings changed from their
 * default, then random ones, then the rest. The probability is left to
 * nodeChance.
 */
export function summarizeSettings(
  spec: DatasetNodeSpec,
  params: Record<string, unknown>,
  limit = 3,
  privacy = false,
): SettingSummary[] {
  // A node's first file or folder is what it works on, so it shows even when empty; other empty ones are optional.
  const main = spec.params.find(p => p.kind === 'path')?.name;
  const rows = spec.params
    // Weights show on the input ports they belong to (inputShares).
    .filter(p => p.name !== 'probability' && p.kind !== 'weights')
    .map((param, index) => {
      const value = params[param.name] ?? param.default;
      const empty = param.kind === 'path' && !value;
      const text = empty && param.name !== main ? null : formatSetting(param, value, privacy);
      const rank = param.kind === 'path' ? 0 : !same(value, param.default) ? 1 : param.random || isSampler(value) ? 2 : 3;
      return { param, index, text, rank, missing: empty && param.name === main };
    })
    .filter(row => row.text !== null);
  return rows
    .sort((a, b) => a.rank - b.rank || a.index - b.index)
    .slice(0, limit)
    .sort((a, b) => a.index - b.index)
    .map(({ param, text, missing }) => ({ name: param.name, label: param.label, text: text!, ...(missing ? { missing } : {}) }));
}

/** How often a node with a probability runs, 0 to 1, when it is less than always; else null. */
export function nodeChance(spec: DatasetNodeSpec | undefined, params: Record<string, unknown>): number | null {
  const param = spec?.params.find(p => p.name === 'probability');
  if (!param) return null;
  const value = params.probability ?? param.default;
  return isNumber(value) && value < 1 ? Math.max(0, value) : null;
}

/**
 * For a node with a weights setting (one weight per input, in declared
 * order), how often each connected input is picked: its share of the
 * connected inputs' weights. Null for any other node.
 */
export function inputShares(spec: DatasetNodeSpec, params: Record<string, unknown>, connected: Set<string>): Map<string, number> | null {
  const param = spec.params.find(p => p.kind === 'weights');
  if (!param) return null;
  const value = params[param.name] ?? param.default;
  const weights = Array.isArray(value) ? value : [];
  const ports = Object.keys(spec.inputs);
  const weightOf = (port: string) => {
    const w = weights[ports.indexOf(port)];
    return isNumber(w) && w > 0 ? w : 0;
  };
  const linked = ports.filter(p => connected.has(p));
  const total = linked.reduce((sum, p) => sum + weightOf(p), 0);
  return new Map(linked.map(p => [p, total > 0 ? weightOf(p) / total : 0]));
}

/**
 * A node's settings in the order a card shows them: the files and folders it
 * reads or writes, how often it runs, then the rest as the node declares them.
 */
export function cardSettings(spec: DatasetNodeSpec): DatasetParamSpec[] {
  const files = spec.params.filter(p => p.kind === 'path');
  const chance = spec.params.filter(p => p.name === 'probability');
  return [...files, ...chance, ...spec.params.filter(p => !files.includes(p) && !chance.includes(p))];
}
