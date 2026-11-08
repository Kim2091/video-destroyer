// src/utils/datasetStages.ts — the stages of a dataset job (docs.py: 1 Load
// to 8 Save) as the editor uses them: a colour each, the palette grouped and
// searched by stage, and gentle hints where an edge runs back through them.
// Also the heat colours that show how far along its chain a node is.

import { nodeLabel, type DatasetNodeSpec, type DatasetStage, type GraphState } from './datasetGraph';

/**
 * One colour per stage, spread round the hue wheel so neighbouring stages
 * differ, and mid-tone so a header tinted with it reads on dark and light ink
 * alike. Port colours (DatasetNodeCard) are pale or neutral, so a wire is
 * never mistaken for a stage.
 */
const STAGE_COLORS: Record<number, string> = {
  1: '#4f8cf7',
  2: '#22b5c4',
  3: '#e8952c',
  4: '#e5534b',
  5: '#3fb36b',
  6: '#9a6cf2',
  7: '#de5fa8',
  8: '#8c97ab',
};

/** The colour of a stage; nodes without one are drawn in neutral ink. */
export const stageColor = (stage: number | null | undefined): string =>
  (stage && STAGE_COLORS[stage]) || 'rgb(var(--ink-500))';

/**
 * The colour of a place along a chain, from 0 (its start, green) to 1 (its
 * end, red), through yellow and orange. The hue turns rather than the
 * channels mixing, so the middle stays bright.
 */
export const heatColor = (along: number): string =>
  `hsl(${Math.round(130 * (1 - Math.min(1, Math.max(0, along))))} 80% 52%)`;

/** Every stage a node may sit in: its own and its `also` stages. */
export function stagesOf(spec: DatasetNodeSpec | undefined): number[] {
  if (!spec?.stage) return [];
  return [spec.stage, ...(spec.alsoStages ?? [])];
}

/**
 * The stages to show, in order: the engine's list, or when it could not be
 * read, the stages the node types name.
 */
export function stageList(stages: DatasetStage[], specs: DatasetNodeSpec[]): DatasetStage[] {
  if (stages.length) return [...stages].sort((a, b) => a.stage - b.stage);
  const named = new Map<number, string>();
  for (const spec of specs) if (spec.stage) named.set(spec.stage, spec.stageName || `Stage ${spec.stage}`);
  return [...named].sort(([a], [b]) => a - b).map(([stage, name]) => ({ stage, name, help: '' }));
}

export interface PaletteSection {
  stage: DatasetStage;
  specs: DatasetNodeSpec[];
}

/** Lower case words of a search, so "Down  scale" matches as two words. */
const words = (text: string) => text.toLowerCase().split(/\s+/).filter(Boolean);

/** Whether a node type matches every word of a search, in its label, description or stage. */
export function matchesSearch(spec: DatasetNodeSpec, query: string): boolean {
  const haystack = `${spec.label} ${spec.description} ${spec.stageName ?? ''}`.toLowerCase();
  return words(query).every(word => haystack.includes(word));
}

/**
 * The palette: one section per stage in order, each with the node types
 * that match the search, empty sections left out. Node types without a
 * stage follow under their group (stage 0), such as "Values", in the order
 * they come, and the rest last, under "Other".
 */
export function paletteSections(specs: DatasetNodeSpec[], stages: DatasetStage[], query = ''): PaletteSection[] {
  const matching = specs.filter(spec => matchesSearch(spec, query));
  const sections: PaletteSection[] = stageList(stages, specs).map(stage => ({
    stage,
    specs: matching.filter(spec => spec.stage === stage.stage),
  }));
  const known = new Set(sections.map(s => s.stage.stage));
  const unstaged = matching.filter(spec => !spec.stage || !known.has(spec.stage));
  const groups = [...new Set(unstaged.map(spec => spec.group).filter((group): group is string => !!group))];
  for (const group of groups) {
    sections.push({ stage: { stage: 0, name: group, help: '' }, specs: unstaged.filter(spec => spec.group === group) });
  }
  const other = unstaged.filter(spec => !spec.group);
  if (other.length) sections.push({ stage: { stage: 0, name: 'Other', help: '' }, specs: other });
  return sections.filter(section => section.specs.length > 0);
}

export interface StageHint {
  message: string;
  nodeId: string;
  edgeId: string;
}

/**
 * Edges that run backwards through the stages: from a node whose earliest
 * stage comes after the latest stage of the node it feeds, such as a
 * Degrade node fed from a Pick node. The job may still run; it is a hint.
 */
export function stageHints(state: GraphState, specs: Map<string, DatasetNodeSpec>): StageHint[] {
  const byId = new Map(state.nodes.map(n => [n.id, n]));
  const hints: StageHint[] = [];
  for (const edge of state.edges) {
    const src = byId.get(edge.source);
    const dst = byId.get(edge.target);
    if (!src || !dst) continue;
    const srcSpec = specs.get(src.data.type);
    const dstSpec = specs.get(dst.data.type);
    if (!srcSpec?.outputs[edge.sourceHandle ?? ''] || !dstSpec?.inputs[edge.targetHandle ?? '']) continue;
    const from = stagesOf(srcSpec);
    const to = stagesOf(dstSpec);
    if (!from.length || !to.length || Math.min(...from) <= Math.max(...to)) continue;
    const srcName = nodeLabel(src.data, srcSpec);
    const dstName = nodeLabel(dst.data, dstSpec);
    hints.push({
      nodeId: dst.id,
      edgeId: edge.id,
      message: `${dstName} is a ${dstSpec!.stageName || `stage ${dstSpec!.stage}`} step but comes after ${srcName} `
        + `(${srcSpec!.stageName || `stage ${srcSpec!.stage}`}). Jobs usually work best in stage order: `
        + `try moving ${dstName} before ${srcName}.`,
    });
  }
  return hints;
}
