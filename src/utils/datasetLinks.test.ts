import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';
import { describe, expect, it } from 'vitest';
import {
  canConnect, edgeId, jobToGraph, specMap, validateGraph,
  type DatasetJob, type DatasetNodeSpec, type GraphEdge, type GraphNode,
} from './datasetGraph';
import { autoLayout, chainDepth, chunkScope, COLUMN } from './datasetLayout';
import { isLink, linkedParam, paramHandle, valueNodeIds, valueTargets } from './datasetLinks';
import { formatOptionList, parseOptionList, readOptions, writeOptions } from './datasetOptions';
import { paletteSections } from './datasetStages';
import { NODE_SPECS } from './datasetSpecs.fixture';

const PRESETS = join(__dirname, '..', '..', 'include', 'vk_dataset', 'presets');
const preset = (prefix: string): DatasetJob =>
  JSON.parse(readFileSync(join(PRESETS, readdirSync(PRESETS).find(f => f.startsWith(prefix))!), 'utf-8'));

const valueSpec = (type: string, params: DatasetNodeSpec['params']): DatasetNodeSpec => ({
  type, category: 'Values', label: type, description: '', stage: null, stageName: 'Values', group: 'Values',
  inputs: {}, optionalInputs: [], outputs: { value: 'value' }, params, chunker: false,
});
const PICK = valueSpec('random_pick', [
  { name: 'options', kind: 'options', default: { choice: [] }, label: 'Options', help: '', random: false, linkable: false, gives: true },
]);
const NUMBER = valueSpec('random_number', [
  { name: 'low', kind: 'float', default: 0, label: 'Low', help: '', random: false, linkable: true },
]);
const IF = valueSpec('if_equals', [
  { name: 'value', kind: 'value', default: '', label: 'Value', help: '', random: false, linkable: true },
  { name: 'then', kind: 'value', default: '', label: 'Then', help: '', random: true, linkable: true, gives: true },
]);
const CODEC: DatasetNodeSpec = {
  type: 'codec', category: 'Degrade', label: 'Compression', description: '', stage: 4, stageName: 'Degrade',
  inputs: { clip: 'clip' }, optionalInputs: [], outputs: { clip: 'clip' }, chunker: false,
  params: [
    { name: 'profile', kind: 'choice', default: 'web_h264', label: 'Profile', help: '', random: true, linkable: true, choices: ['dvd_mpeg2', 'web_h264', 'web_h265'] },
    { name: 'quality', kind: 'float', default: 0.5, label: 'Quality', help: '', random: true, linkable: true, min: 0, max: 1 },
    { name: 'table', kind: 'json', default: {}, label: 'Table', help: '', random: false, linkable: false },
  ],
};
const specs = specMap([...NODE_SPECS.filter(s => s.type !== 'codec'), CODEC, PICK, NUMBER, IF]);

const node = (id: string, type: string, params: Record<string, unknown> = {}): GraphNode =>
  ({ id, type: 'dataset', position: { x: 0, y: 0 }, data: { type, params, enabled: true, extra: {} } });
const edge = (source: string, sourceHandle: string, target: string, targetHandle: string): GraphEdge =>
  ({ id: edgeId(source, sourceHandle, target, targetHandle), source, sourceHandle, target, targetHandle });

describe('links into settings', () => {
  it('names the setting a handle carries', () => {
    expect(linkedParam(paramHandle('quality'))).toBe('quality');
    expect(linkedParam('clip')).toBeNull();
    expect(isLink(edge('p', 'value', 'c', 'param:profile'))).toBe(true);
  });

  it('round-trips a job with links as ordinary edges', () => {
    const state = jobToGraph(preset('17'));
    const link = state.edges.find(e => e.source === 'profile' && e.target === 'comp')!;
    expect(link.targetHandle).toBe('param:profile');
  });

  it('connects a value only into a linkable setting, and never into a port', () => {
    const nodes = [node('p', 'random_pick'), node('c', 'codec'), node('n', 'random_number')];
    const ok = (targetHandle: string, source = 'p') => canConnect({ source, sourceHandle: 'value', target: 'c', targetHandle }, nodes, [], specs);
    expect(ok('param:profile')).toBe(true);
    expect(ok('param:table')).toBe(false);
    expect(ok('param:nothing')).toBe(false);
    expect(ok('clip')).toBe(false);
    // A value node fed by its own value would loop.
    expect(canConnect({ source: 'n', sourceHandle: 'value', target: 'n', targetHandle: 'param:low' }, nodes, [], specs)).toBe(false);
  });

  it('flags a pick a dropdown does not offer, naming its options', () => {
    const state = {
      seed: 0, settings: {}, extra: {},
      nodes: [node('p', 'random_pick', { options: { choice: ['web_h264', 'vhs'] } }), node('c', 'codec')],
      edges: [edge('p', 'value', 'c', 'param:profile')],
    };
    const messages = validateGraph(state, specs, true).map(i => i.message);
    expect(messages.some(m => m.includes('"vhs"') && m.includes('dvd_mpeg2, web_h264, web_h265'))).toBe(true);
  });

  it('flags a link into a setting that takes none, or two links into one setting', () => {
    const state = {
      seed: 0, settings: {}, extra: {},
      nodes: [node('p', 'random_pick'), node('q', 'random_pick'), node('c', 'codec')],
      edges: [edge('p', 'value', 'c', 'param:table'), edge('p', 'value', 'c', 'param:quality'), edge('q', 'value', 'c', 'param:quality')],
    };
    const messages = validateGraph(state, specs, true).map(i => i.message);
    expect(messages.some(m => m.includes('a json setting cannot be linked'))).toBe(true);
    expect(messages.some(m => m.includes('Quality: has more than one link'))).toBe(true);
  });

  it('edits what a value node gives with the dropdown it feeds, through If Equals', () => {
    const nodes = [node('p', 'random_pick'), node('i', 'if_equals'), node('c', 'codec')];
    const edges = [edge('p', 'value', 'i', 'param:value'), edge('p', 'value', 'c', 'param:quality'), edge('i', 'value', 'c', 'param:profile')];
    const targets = valueTargets(nodes, edges, specs);
    // A link into If Equals' tested value is not where its own value goes; a dropdown wins over a number.
    expect(targets.get('p')?.name).toBe('quality');
    expect(targets.get('i')?.name).toBe('profile');
  });
});

describe('value nodes on the canvas', () => {
  it('keeps value wires out of the chain steps', () => {
    const state = jobToGraph(preset('17'));
    const depths = chainDepth(state.nodes, state.edges, specs);
    expect(depths.get('comp')).toBe(3);
    expect(depths.get('reup')).toBe(4);
    expect(depths.has('profile')).toBe(false);
    expect([...valueNodeIds(state.nodes, state.edges)].sort()).toEqual(['factor', 'profile', 'quality', 'requality']);
  });

  it('puts value nodes in rows above the band of the nodes they feed, left of them', () => {
    const state = jobToGraph(preset('17'));
    const layout = autoLayout(state.nodes, state.edges, { specs });
    const at = (id: string) => layout.get(id)!;
    expect(at('profile').y).toBeLessThan(at('comp').y);
    expect(at('profile').x).toBe(at('comp').x - COLUMN);
    // Two values for one place share a row, side by side.
    expect(at('quality').y).toBe(at('profile').y);
    expect(at('quality').x).toBe(at('profile').x - COLUMN);
    // The chain keeps its columns.
    expect(at('reup').x - at('comp').x).not.toBe(0);
  });

  it('runs a value node per chunk only when everything it feeds does, the chunk node aside', () => {
    const nodes = [node('src', 'load_video'), node('chunks', 'chunk'), node('c', 'codec'), node('n', 'random_number'), node('m', 'random_number')];
    const edges = [
      edge('src', 'clip', 'chunks', 'clip'), edge('chunks', 'clip', 'c', 'clip'),
      edge('n', 'value', 'c', 'param:quality'), edge('m', 'value', 'c', 'param:quality'), edge('m', 'value', 'chunks', 'param:max_chunks'),
    ];
    const scope = chunkScope(nodes, edges, specs);
    expect(scope.has('n')).toBe(true);
    expect(scope.has('m')).toBe(false);
  });

  it('lists value nodes under their group, after the stages', () => {
    const sections = paletteSections([...specs.values()], []);
    expect(sections[sections.length - 1].stage.name).toBe('Values');
    expect(sections[sections.length - 1].specs.map(s => s.type)).toEqual(['random_pick', 'random_number', 'if_equals']);
  });
});

describe('typed options', () => {
  it('reads numbers as numbers and keeps weights only while they differ', () => {
    expect(parseOptionList('a, 0.5 ,, 2')).toEqual(['a', 0.5, 2]);
    expect(formatOptionList(['a', 0.5])).toBe('a, 0.5');
    expect(writeOptions({ choice: ['a', 'b'], weights: [1, 1] })).toEqual({ choice: ['a', 'b'] });
    expect(writeOptions({ choice: ['a', 'b'], weights: [3, 1] })).toEqual({ choice: ['a', 'b'], weights: [3, 1] });
    expect(readOptions({ choice: ['a'], weights: [1, 2] })).toEqual({ choice: ['a'], weights: null });
    expect(readOptions('x')).toEqual({ choice: [], weights: null });
  });
});
