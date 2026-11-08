import { describe, expect, it } from 'vitest';
import {
  clampPan,
  compositeSize,
  cropsAt,
  existingPort,
  formatDrawn,
  portForNode,
  previewJob,
  previewPorts,
  rangeLabel,
  rgbToRgba,
  zoomScale,
} from './datasetPreview';
import type { DatasetNodeSpec, GraphNode } from './datasetGraph';
import type { DatasetPreviewSequences } from '../../electron/datasetPreviewProtocol';

const spec = (type: string, outputs: Record<string, string>): DatasetNodeSpec => ({
  type, category: 'Test', label: type.toUpperCase(), description: '', inputs: {}, optionalInputs: [], outputs, params: [], chunker: false,
});
const specs = new Map([
  ['src', spec('src', { clip: 'clip' })],
  ['ds', spec('ds', { hr: 'clip', lr: 'clip', runs: 'static_runs' })],
  ['seq', spec('seq', { sequences: 'sequences' })],
  ['out', spec('out', {})],
]);
const node = (id: string, type: string, label?: string): GraphNode => ({
  id, position: { x: 0, y: 0 }, data: { type, params: {}, enabled: true, extra: {}, ...(label ? { label } : {}) },
});
const nodes = [node('a', 'src'), node('b', 'ds', 'Pairs'), node('c', 'seq'), node('d', 'out'), node('e', 'gone')];

describe('previewPorts', () => {
  it('lists clip and sequences outputs, naming the port only for nodes with several', () => {
    expect(previewPorts(nodes, specs)).toEqual([
      { node: 'a', port: 'clip', type: 'clip', label: 'SRC' },
      { node: 'b', port: 'hr', type: 'clip', label: 'Pairs · hr' },
      { node: 'b', port: 'lr', type: 'clip', label: 'Pairs · lr' },
      { node: 'c', port: 'sequences', type: 'sequences', label: 'SEQ' },
    ]);
  });
});

describe('portForNode and existingPort', () => {
  const ports = previewPorts(nodes, specs);
  it('keeps the shown port of the picked node, else takes its first', () => {
    expect(portForNode('b', { node: 'b', port: 'lr' }, ports)).toEqual({ node: 'b', port: 'lr' });
    expect(portForNode('b', { node: 'a', port: 'clip' }, ports)).toEqual({ node: 'b', port: 'hr' });
  });
  it('has nothing for a node without a previewable output', () => {
    expect(portForNode('d', null, ports)).toBeNull();
  });
  it('drops a port that left the graph', () => {
    expect(existingPort({ node: 'z', port: 'clip' }, ports)).toBeNull();
    expect(existingPort({ node: 'a', port: 'clip' }, ports)).toEqual({ node: 'a', port: 'clip' });
  });
});

describe('previewJob', () => {
  it('drops canvas positions and notes and keeps everything else', () => {
    const job = { seed: 1, nodes: [{ id: 'a', type: 'src', params: { path: 'x' }, ui: { x: 4, y: 5 }, note: 'n' }], edges: [] };
    expect(previewJob(job)).toEqual({ seed: 1, nodes: [{ id: 'a', type: 'src', params: { path: 'x' } }], edges: [] });
    expect(job.nodes[0].ui).toEqual({ x: 4, y: 5 });
  });
});

describe('rgbToRgba', () => {
  it('adds an opaque alpha to each pixel', () => {
    expect(Array.from(rgbToRgba(new Uint8Array([1, 2, 3, 4, 5, 6]), 2, 1))).toEqual([1, 2, 3, 255, 4, 5, 6, 255]);
  });
  it('refuses a buffer too short for the size', () => {
    expect(() => rgbToRgba(new Uint8Array(5), 2, 1)).toThrow();
  });
});

describe('layout and zoom', () => {
  it('puts a compare picture beside the frame at the frame’s size, or under it for a wipe', () => {
    expect(compositeSize({ width: 320, height: 240 }, true, 'side')).toEqual({ width: 640, height: 240 });
    expect(compositeSize({ width: 320, height: 240 }, true, 'wipe')).toEqual({ width: 320, height: 240 });
    expect(compositeSize({ width: 320, height: 240 }, false, 'side')).toEqual({ width: 320, height: 240 });
  });
  it('fits up or down, and otherwise uses the fixed zoom', () => {
    expect(zoomScale('fit', { width: 160, height: 120 }, { width: 640, height: 240 })).toBe(2);
    expect(zoomScale('fit', { width: 1920, height: 1080 }, { width: 960, height: 1080 })).toBe(0.5);
    expect(zoomScale(2, { width: 1920, height: 1080 }, { width: 10, height: 10 })).toBe(2);
    expect(zoomScale('fit', { width: 0, height: 0 }, { width: 10, height: 10 })).toBe(1);
  });
  it('centres a picture that fits and stops a larger one at its edges', () => {
    expect(clampPan({ x: 50, y: -50 }, { width: 100, height: 100 }, { width: 200, height: 200 })).toEqual({ x: 0, y: 0 });
    expect(clampPan({ x: 500, y: -30 }, { width: 400, height: 300 }, { width: 200, height: 200 })).toEqual({ x: 100, y: -30 });
  });
});

describe('cropsAt', () => {
  const sequences: DatasetPreviewSequences = {
    clips: { hr: { width: 320, height: 240, frames: 20 }, lr: { width: 160, height: 120, frames: 20 } },
    items: [
      { frames: { hr: [2], lr: [0, 1, 2, 3, 4] }, crops: { hr: [0, 0, 64, 64], lr: [0, 0, 32, 32] }, info: {} },
      { frames: { hr: [7], lr: [5, 6, 7, 8, 9] }, crops: {}, info: {} },
    ],
    rejected: [{ frames: { hr: [2], lr: [0, 1, 2, 3, 4] }, crops: {}, info: { rejected: 'too little motion' } }],
  };
  it('returns the kept and rejected sequences on the frame, with their crops in the shown clip', () => {
    expect(cropsAt(sequences, 'lr', 3)).toEqual([
      { index: 0, rejected: false, reason: null, rect: [0, 0, 32, 32] },
      { index: 0, rejected: true, reason: 'too little motion', rect: [0, 0, 160, 120] },
    ]);
    expect(cropsAt(sequences, 'hr', 3)).toEqual([]);
    expect(cropsAt(sequences, 'hr', 7)).toEqual([{ index: 1, rejected: false, reason: null, rect: [0, 0, 320, 240] }]);
    expect(cropsAt(sequences, 'nope', 3)).toEqual([]);
  });
});

describe('formatDrawn and rangeLabel', () => {
  it('rounds numbers and writes structures compactly', () => {
    expect(formatDrawn(0.123456789)).toBe('0.1235');
    expect(formatDrawn(33)).toBe('33');
    expect(formatDrawn('h264')).toBe('h264');
    expect(formatDrawn({ scale: 0.876543, kernel: 'area' })).toBe('{"scale":0.8765,"kernel":"area"}');
    expect(formatDrawn(null)).toBe('—');
  });
  it('labels a chunk by its inclusive frames', () => {
    expect(rangeLabel([72, 144])).toBe('frames 72–143');
    expect(rangeLabel(null)).toBe('whole video');
  });
});
