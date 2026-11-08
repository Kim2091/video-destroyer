import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';
import { describe, expect, it } from 'vitest';
import { jobToGraph, specMap, type DatasetJob, type GraphNode } from './datasetGraph';
import { autoConnect, autoLayout, BAND_COLUMNS, boundsOf, CARD_HEIGHT, CARD_WIDTH, chunkScope, COLUMN, placeNewNode } from './datasetLayout';
import { NODE_SPECS } from './datasetSpecs.fixture';

const specs = specMap(NODE_SPECS);
const PRESETS = join(__dirname, '..', '..', 'include', 'vk_dataset', 'presets');
const preset = (prefix: string): DatasetJob =>
  JSON.parse(readFileSync(join(PRESETS, readdirSync(PRESETS).find(f => f.startsWith(prefix))!), 'utf-8'));

/** Column and row of every node, from a layout. */
function cells(job: DatasetJob) {
  const state = jobToGraph(job);
  const layout = autoLayout(state.nodes, state.edges, { specs });
  const rows = [...new Set([...layout.values()].map(p => p.y))].sort((a, b) => a - b);
  return Object.fromEntries([...layout].map(([id, p]) => [id, [p.x / COLUMN, rows.indexOf(p.y)]]));
}

const node = (id: string, type: string, x = 0, y = 0): GraphNode =>
  ({ id, type: 'dataset', position: { x, y }, data: { type, params: {}, enabled: true, extra: {} } });

describe('auto layout', () => {
  it('puts the HR path on top and the degraded LR path under it, one column per step', () => {
    expect(cells(preset('01'))).toEqual({
      src: [0, 0], chunks: [1, 0], down: [2, 1], comp: [3, 1], seq: [4, 0], out: [5, 0],
    });
  });

  it('keeps two releases on their own rows, by the ports they feed', () => {
    expect(cells(preset('06'))).toEqual({
      hr_src: [0, 0], lr_src: [0, 1], hr_crop: [1, 0], lr_crop: [1, 1], time: [2, 1], space: [3, 0],
      chunks: [4, 0], seq: [1, 2], tiles: [2, 2], aligned: [3, 2], dedup: [4, 2], out: [5, 2],
    });
  });

  it('wraps a long chain onto bands of even length, each a column further right', () => {
    const ids = Array.from({ length: 14 }, (_, i) => `n${i}`);
    const nodes = ids.map(id => node(id, 'blur'));
    const edges = ids.slice(1).map((id, i) => ({ id: `e${i}`, source: ids[i], target: id, sourceHandle: 'clip', targetHandle: 'clip' }));
    const layout = autoLayout(nodes, edges, { specs });
    const xs = new Set([...layout.values()].map(p => p.x / COLUMN));
    const ys = new Set([...layout.values()].map(p => p.y));
    expect(Math.max(...xs)).toBeLessThan(BAND_COLUMNS + 2);
    expect(ys.size).toBe(3);
    expect(layout.get('n0')!.x).toBe(0);
    expect(layout.get('n5')!.x).toBe(COLUMN);
    expect(layout.get('n13')!.x).toBe(Math.max(...xs) * COLUMN);
    expect(layout.get('n5')!.y).toBeGreaterThan(layout.get('n4')!.y);
  });

  it('keeps a node that feeds both paths on the HR path', () => {
    // The centre frames are saved as LR and, denoised, guide the HR average.
    const at = cells(preset('09'));
    expect([at.src, at.find, at.centers, at.denoise, at.stack, at.out]).toEqual([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]);
  });

  it('lays out every example without two cards in one place', () => {
    for (const file of readdirSync(PRESETS)) {
      const state = jobToGraph(JSON.parse(readFileSync(join(PRESETS, file), 'utf-8')));
      const layout = autoLayout(state.nodes, state.edges, { specs });
      const spots = [...layout.values()].map(p => `${p.x},${p.y}`);
      expect(new Set(spots).size, file).toBe(spots.length);
    }
  });

  it('spaces rows by the tallest measured card', () => {
    const state = jobToGraph(preset('01'));
    const layout = autoLayout(state.nodes, state.edges, { heightOf: id => (id === 'seq' ? 400 : 150) });
    expect(layout.get('down')!.y).toBe(400 + 48);
  });

  it('puts nodes with no edges in a last row, in stage order', () => {
    const nodes = [node('out', 'write_dataset'), node('src', 'load_video'), node('blur', 'blur')];
    const layout = autoLayout(nodes, [], { specs });
    expect(['src', 'blur', 'out'].map(id => layout.get(id)!.x)).toEqual([0, COLUMN, 2 * COLUMN]);
  });
});

describe('adding a node from the palette', () => {
  const src = node('src', 'load_video', 0, 0);
  const chunks = node('chunks', 'chunk', COLUMN, 0);

  it('goes right of the selected node, below anything already there', () => {
    const taken = node('down', 'resize', 2 * COLUMN, 0);
    const at = placeNewNode(specs.get('blur')!, [src, chunks, taken], specs, 'chunks');
    expect(at.x).toBe(2 * COLUMN);
    expect(at.y).toBeGreaterThan(CARD_HEIGHT);
  });

  it('with nothing selected, goes right of the last node of an earlier stage', () => {
    expect(placeNewNode(specs.get('make_sequences')!, [src, chunks], specs, null)).toEqual({ x: 2 * COLUMN, y: 0 });
    expect(placeNewNode(specs.get('load_video')!, [src, chunks], specs, null).x).toBe(0);
    expect(placeNewNode(specs.get('load_video')!, [], specs, null)).toEqual({ x: 0, y: 0 });
  });

  it('joins the one free output to the one input that takes it', () => {
    expect(autoConnect(specs.get('blur')!, src, [], specs)).toEqual({ sourceHandle: 'clip', targetHandle: 'clip' });
    // Split into Chunks has an optional second input; the required one is meant.
    expect(autoConnect(specs.get('chunk')!, src, [], specs)).toEqual({ sourceHandle: 'clip', targetHandle: 'clip' });
    // Cut into Sequences needs HR; its LR is optional.
    expect(autoConnect(specs.get('make_sequences')!, src, [], specs)).toEqual({ sourceHandle: 'clip', targetHandle: 'hr' });
    // Without a second video, the second chunk output carries nothing.
    expect(autoConnect(specs.get('blur')!, chunks, [], specs)).toEqual({ sourceHandle: 'clip', targetHandle: 'clip' });
    expect(autoConnect(specs.get('write_dataset')!, node('seq', 'make_sequences'), [], specs))
      .toEqual({ sourceHandle: 'sequences', targetHandle: 'sequences' });
  });

  it('joins nothing when there is more than one way, or none', () => {
    const used = [{ id: 'e', source: 'src', sourceHandle: 'clip', target: 'x', targetHandle: 'clip' }];
    expect(autoConnect(specs.get('blur')!, src, used, specs)).toBeNull();
    // HR or LR: the new node's two inputs are both required.
    expect(autoConnect(specs.get('pair_sequences')!, src, [], specs)).toBeNull();
    // Split into Chunks with a second video has two free outputs.
    const second = [{ id: 'e', source: 'hr', sourceHandle: 'clip', target: 'chunks', targetHandle: 'aux' }];
    expect(autoConnect(specs.get('blur')!, chunks, second, specs)).toBeNull();
    expect(autoConnect(specs.get('load_video')!, src, [], specs)).toBeNull();
    expect(autoConnect(specs.get('blur')!, null, [], specs)).toBeNull();
  });
});

describe('the per-chunk region', () => {
  it('holds the chunk node and everything that reads from it, as graph.py works it out', () => {
    const state = jobToGraph(preset('06'));
    expect([...chunkScope(state.nodes, state.edges, specs)].sort()).toEqual(['aligned', 'chunks', 'dedup', 'out', 'seq', 'tiles']);
    const plain = jobToGraph(preset('09'));
    expect(chunkScope(plain.nodes, plain.edges, specs).size).toBe(0);
  });

  it('is drawn around its cards with room for its label', () => {
    const a = node('a', 'blur', 0, 0);
    const b = node('b', 'blur', COLUMN, 100);
    expect(boundsOf([a, b], new Set(['a', 'b']), () => 100, 10, 20)).toEqual({
      x: -10, y: -30, width: COLUMN + CARD_WIDTH + 20, height: 200 + 30 + 10,
    });
    expect(boundsOf([a], new Set(), () => 100)).toBeNull();
  });
});
