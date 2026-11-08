import { describe, expect, it } from 'vitest';
import { canConnect, connectedInputs, jobVariants, specMap, validateGraph, type DatasetNodeSpec, type GraphEdge, type GraphNode } from './datasetGraph';
import { previewPorts } from './datasetPreview';
import { inputShares, summarizeSettings } from './datasetSummary';

// Declarations as `python -m vk_dataset nodes` prints them, trimmed.
const BRANCH: DatasetNodeSpec = {
  type: 'random_branch', category: 'Combine', label: 'Random Branch', description: '',
  inputs: { in_1: 'clip', in_2: 'clip', in_3: 'clip', in_4: 'clip' }, optionalInputs: ['in_2', 'in_3', 'in_4'],
  outputs: { clip: 'clip' }, chunker: false,
  params: [{ name: 'weights', kind: 'weights', default: [1, 1, 1, 1], label: 'Weights', help: '', random: false }],
};
const MASK: DatasetNodeSpec = {
  type: 'luma_mask', category: 'Mask', label: 'Brightness Mask', description: '', inputs: { clip: 'clip' },
  optionalInputs: [], outputs: { mask: 'mask' }, chunker: false, params: [],
};
const MERGE: DatasetNodeSpec = {
  type: 'masked_merge', category: 'Combine', label: 'Masked Merge', description: '',
  inputs: { base: 'clip', overlay: 'clip', mask: 'mask' }, optionalInputs: [], outputs: { clip: 'clip' }, chunker: false, params: [],
};
const specs = specMap([BRANCH, MASK, MERGE]);

const node = (id: string, type: string, params: Record<string, unknown> = {}): GraphNode =>
  ({ id, position: { x: 0, y: 0 }, data: { type, params, enabled: true, extra: {} } });
const edge = (source: string, sourceHandle: string, target: string, targetHandle: string): GraphEdge =>
  ({ id: `${source}.${sourceHandle}->${target}.${targetHandle}`, source, sourceHandle, target, targetHandle });

describe('random branch weights', () => {
  it('give each connected input its share, ignoring unconnected ones', () => {
    const shares = inputShares(BRANCH, { weights: [3, 1, 4, 0] }, new Set(['in_1', 'in_2', 'in_4']));
    expect(Object.fromEntries(shares!)).toEqual({ in_1: 0.75, in_2: 0.25, in_4: 0 });
    expect(inputShares(BRANCH, {}, new Set(['in_1', 'in_2']))!.get('in_1')).toBe(0.5);
    expect(inputShares(MASK, {}, new Set(['clip']))).toBeNull();
  });

  it('show on the ports, not in the settings line', () => {
    expect(summarizeSettings(BRANCH, { weights: [3, 1] })).toEqual([]);
  });

  it('read the connected inputs from the edges', () => {
    const connected = connectedInputs([edge('a', 'clip', 'pick', 'in_1'), edge('b', 'clip', 'pick', 'in_3')]);
    expect([...connected.get('pick')!]).toEqual(['in_1', 'in_3']);
  });
});

describe('masks', () => {
  const nodes = [node('lr', 'luma_mask'), node('m', 'luma_mask'), node('merge', 'masked_merge')];

  it('connect only to mask inputs', () => {
    expect(canConnect({ source: 'm', sourceHandle: 'mask', target: 'merge', targetHandle: 'mask' }, nodes, [], specs)).toBe(true);
    expect(canConnect({ source: 'm', sourceHandle: 'mask', target: 'merge', targetHandle: 'base' }, nodes, [], specs)).toBe(false);
  });

  it('can be previewed', () => {
    expect(previewPorts(nodes, specs).map(p => `${p.node}.${p.port}:${p.type}`)).toContain('m.mask:mask');
  });
});

describe('variants', () => {
  it('are the most any enabled writer writes', () => {
    const job = (variants: unknown[], enabled = true) => ({
      nodes: variants.map((v, i) => ({ id: `w${i}`, type: 'write_dataset', enabled, params: { variants: v } })),
      edges: [],
    });
    expect(jobVariants(job([1, 3, 2]))).toBe(3);
    expect(jobVariants(job([3], false))).toBe(1);
    expect(jobVariants(job(['x']))).toBe(1);
    expect(jobVariants({ nodes: [], edges: [] })).toBe(1);
  });

  it('need a chunk node upstream of their writer', () => {
    const writer: DatasetNodeSpec = {
      type: 'write_dataset', category: 'Output', label: 'Save Dataset', description: '', inputs: { clip: 'clip' },
      optionalInputs: [], outputs: {}, chunker: false,
      params: [{ name: 'variants', kind: 'int', default: 1, label: 'Variants', help: '', random: false }],
    };
    const chunk: DatasetNodeSpec = { ...MERGE, type: 'chunk', inputs: {}, outputs: { clip: 'clip' }, chunker: true };
    const all = specMap([writer, chunk]);
    const state = (edges: GraphEdge[]) => ({
      seed: 0, settings: {}, extra: {}, edges,
      nodes: [node('chunks', 'chunk'), node('free', 'chunk'), node('out', 'write_dataset', { variants: 2 })],
    });
    const problems = (edges: GraphEdge[]) => validateGraph(state(edges), all, true).filter(i => i.message.includes('variants'));
    expect(problems([])).toHaveLength(1);
    expect(problems([edge('chunks', 'clip', 'out', 'clip')])).toHaveLength(0);
  });
});
