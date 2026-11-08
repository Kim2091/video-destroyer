import { describe, expect, it } from 'vitest';
import { COLUMN } from './datasetLayout';
import {
  canConnect,
  connectEdge,
  createNode,
  exampleTitle,
  findCycle,
  graphToJob,
  jobToGraph,
  jobHasLayout,
  newNodeId,
  specMap,
  splitPortRef,
  validateGraph,
  type DatasetJob,
  type DatasetNodeSpec,
  type GraphState,
} from './datasetGraph';

// Declarations as `python -m vk_dataset nodes` prints them, trimmed.
const SPECS: DatasetNodeSpec[] = [
  { type: 'load_video', category: 'Source', label: 'Load Video', description: '', inputs: {}, optionalInputs: [],
    outputs: { clip: 'clip' }, chunker: false,
    params: [{ name: 'path', kind: 'path', default: '', label: 'Path', help: '', random: false }] },
  { type: 'chunk', category: 'Chunk', label: 'Split into Chunks', description: '', inputs: { clip: 'clip', aux: 'clip' },
    optionalInputs: ['aux'], outputs: { clip: 'clip', aux: 'clip' }, chunker: true,
    params: [{ name: 'strategy', kind: 'choice', default: 'scene', label: 'Strategy', help: '', random: false, choices: ['scene', 'frames'] }] },
  { type: 'resize', category: 'Degrade', label: 'Resize', description: '', inputs: { clip: 'clip' }, optionalInputs: [],
    outputs: { clip: 'clip' }, chunker: false,
    params: [
      { name: 'probability', kind: 'float', default: 1, label: 'Probability', help: '', random: false, min: 0, max: 1 },
      { name: 'down_up_scale', kind: 'float', default: { uniform: [0.75, 1] }, label: 'Down up scale', help: '', random: true },
    ] },
  { type: 'make_sequences', category: 'Sequences', label: 'Pick Sequences', description: '', inputs: { hr: 'clip', lr: 'clip' },
    optionalInputs: ['lr'], outputs: { sequences: 'sequences' }, chunker: false,
    params: [{ name: 'length', kind: 'int', default: 5, label: 'Length', help: '', random: false, min: 1 }] },
  { type: 'telecine', category: 'Degrade', label: 'Telecine', description: '', inputs: { clip: 'clip' }, optionalInputs: [],
    outputs: { clip: 'clip' }, chunker: false, needsAfter: 'telecine_fix', params: [] },
  { type: 'telecine_fix', category: 'Degrade', label: 'Bad Telecine Fix', description: '', inputs: { clip: 'clip' },
    optionalInputs: [], outputs: { clip: 'clip' }, chunker: false, needsBefore: 'telecine', params: [] },
  { type: 'write_dataset', category: 'Output', label: 'Write Dataset', description: '', inputs: { sequences: 'sequences' },
    optionalInputs: [], outputs: {}, chunker: false,
    params: [{ name: 'directory', kind: 'path', default: 'frames', label: 'Directory', help: '', random: false }] },
];
const specs = specMap(SPECS);

const JOB: DatasetJob = {
  version: 1,
  seed: 42,
  settings: { work_dir: 'D:/tmp' },
  note: 'kept',
  nodes: [
    { id: 'src', type: 'load_video', params: { path: 'D:/in.mkv' }, ui: { x: 0, y: 0 } },
    { id: 'chunk', type: 'chunk', params: { strategy: 'frames' }, ui: { x: 280, y: 0 } },
    { id: 'down', type: 'resize', params: { down_up_scale: { int: [1, 2] } }, enabled: false, label: 'Half', ui: { x: 560, y: 0 }, tag: 1 },
    { id: 'seq', type: 'make_sequences', params: { length: 3 }, ui: { x: 840, y: 0 } },
    { id: 'out', type: 'write_dataset', params: { directory: 'D:/out' }, ui: { x: 1120, y: 0 } },
  ],
  edges: [
    { from: 'src.clip', to: 'chunk.clip' },
    { from: 'chunk.clip', to: 'down.clip' },
    { from: 'chunk.clip', to: 'seq.hr' },
    { from: 'down.clip', to: 'seq.lr' },
    { from: 'seq.sequences', to: 'out.sequences' },
  ],
};

const graph = (job: DatasetJob = JOB): GraphState => jobToGraph(job);
const messages = (state: GraphState, saved = true) => validateGraph(state, specs, saved).map(i => i.message);

describe('job files', () => {
  it('round-trip exactly, positions under ui and unknown keys kept', () => {
    const job = graphToJob(graph());
    expect(job).toEqual({
      ...JOB,
      nodes: JOB.nodes.map(n => ({ ...n, enabled: n.enabled ?? true })),
    });
    expect(job.nodes[2]).toMatchObject({ label: 'Half', enabled: false, tag: 1 });
  });

  it('lay out nodes saved without a position, one column per step, the LR path under the HR path', () => {
    const job = { ...JOB, nodes: JOB.nodes.map(({ ui: _ui, ...n }) => n) };
    expect(jobHasLayout(JOB)).toBe(true);
    expect(jobHasLayout(job)).toBe(false);
    const state = graph(job);
    const at = Object.fromEntries(state.nodes.map(n => [n.id, n.position]));
    expect([at.src, at.chunk, at.down, at.seq, at.out].map(p => p.x)).toEqual([0, 1, 2, 3, 4].map(column => column * COLUMN));
    expect([at.src, at.chunk, at.seq, at.out].map(p => p.y)).toEqual([0, 0, 0, 0]);
    expect(at.down.y).toBeGreaterThan(0);
  });

  it('keep the positions a job was saved with', () => {
    const job = { ...JOB, nodes: JOB.nodes.map((n, i) => (i === 2 ? { ...n, ui: undefined } : n)) };
    expect(jobHasLayout(job)).toBe(true);
    expect(graph(job).nodes.find(n => n.id === 'seq')!.position).toEqual({ x: 840, y: 0 });
  });

  it('read a job with only nodes and edges, and write it as version 1 with seed 0', () => {
    const job = graphToJob(jobToGraph({ nodes: [], edges: [] }));
    expect(job).toEqual({ version: 1, seed: 0, settings: {}, nodes: [], edges: [] });
  });

  it('number examples by their name, else by their place', () => {
    expect(exampleTitle('3. video-destroyer classic', 0)).toEqual({ number: 3, title: 'video-destroyer classic' });
    expect(exampleTitle('Tiles', 4)).toEqual({ number: 5, title: 'Tiles' });
  });

  it('split port references at the last dot, as graph.py does', () => {
    expect(splitPortRef('a.b.clip')).toEqual({ node: 'a.b', port: 'clip' });
    expect(splitPortRef('clip')).toBeNull();
    expect(splitPortRef('a.')).toBeNull();
  });
});

describe('new nodes', () => {
  it('get a free id and their declared defaults', () => {
    expect(newNodeId('resize', ['resize', 'resize_2'])).toBe('resize_3');
    const node = createNode(specs.get('resize')!, { x: 1, y: 2 }, ['src']);
    expect(node.id).toBe('resize');
    expect(node.data.params).toEqual({ probability: 1, down_up_scale: { uniform: [0.75, 1] } });
    // Defaults are copies: editing a node never edits the declaration.
    (node.data.params.down_up_scale as { uniform: number[] }).uniform[0] = 0;
    expect(specs.get('resize')!.params[1].default).toEqual({ uniform: [0.75, 1] });
  });
});

describe('connecting ports', () => {
  const state = graph();
  const connection = (source: string, sourceHandle: string, target: string, targetHandle: string) =>
    ({ source, sourceHandle, target, targetHandle });

  it('joins ports of the same type only', () => {
    expect(canConnect(connection('src', 'clip', 'chunk', 'aux'), state.nodes, state.edges, specs)).toBe(true);
    expect(canConnect(connection('src', 'clip', 'out', 'sequences'), state.nodes, state.edges, specs)).toBe(false);
    expect(canConnect(connection('src', 'nope', 'chunk', 'aux'), state.nodes, state.edges, specs)).toBe(false);
  });

  it('refuses a loop and a node joined to itself', () => {
    expect(canConnect(connection('down', 'clip', 'chunk', 'aux'), state.nodes, state.edges, specs)).toBe(false);
    expect(canConnect(connection('down', 'clip', 'down', 'clip'), state.nodes, state.edges, specs)).toBe(false);
  });

  it('replaces the edge already on an input, since an input takes one', () => {
    const edges = connectEdge(state.edges, connection('src', 'clip', 'seq', 'hr'));
    expect(edges.filter(e => e.target === 'seq' && e.targetHandle === 'hr')).toEqual([
      expect.objectContaining({ source: 'src', sourceHandle: 'clip' }),
    ]);
    expect(edges).toHaveLength(state.edges.length);
  });

  it('lets an occupied input take a new edge that would only loop through the edge it replaces', () => {
    expect(canConnect(connection('src', 'clip', 'down', 'clip'), state.nodes, state.edges, specs)).toBe(true);
  });
});

describe('validation, as graph.py checks a job', () => {
  it('passes a complete job', () => {
    expect(messages(graph())).toEqual([]);
  });

  it('names a missing required input but not a missing optional one', () => {
    const state = graph();
    state.edges = state.edges.filter(e => e.targetHandle !== 'hr' && e.targetHandle !== 'lr');
    expect(messages(state)).toEqual(['Pick Sequences (seq) needs an input on: hr.']);
  });

  it('finds a cycle and names it in the direction data flows', () => {
    const state = graph();
    state.edges.push({ id: 'x', source: 'down', sourceHandle: 'clip', target: 'chunk', targetHandle: 'aux' });
    expect(findCycle(state)).toEqual(['chunk', 'down', 'chunk']);
    expect(messages(state)).toContain('The graph has a cycle through chunk -> down -> chunk.');
  });

  it('allows one chunk node only', () => {
    const state = graph();
    state.nodes.push({ ...state.nodes[1], id: 'chunk_2' });
    state.edges.push({ id: 'y', source: 'src', sourceHandle: 'clip', target: 'chunk_2', targetHandle: 'clip' });
    expect(messages(state).filter(m => m.startsWith('A job can have only one chunk node'))).toHaveLength(2);
  });

  it('needs a writer', () => {
    const state = graph();
    state.nodes = state.nodes.filter(n => n.id !== 'out');
    state.edges = state.edges.filter(e => e.target !== 'out');
    expect(messages(state)).toEqual(['A job needs at least one writer: a node with no outputs.']);
  });

  it('reports edges that join the wrong types or crowd one input', () => {
    const state = graph();
    state.edges.push({ id: 'a', source: 'seq', sourceHandle: 'sequences', target: 'chunk', targetHandle: 'aux' });
    state.edges.push({ id: 'b', source: 'src', sourceHandle: 'clip', target: 'down', targetHandle: 'clip' });
    expect(messages(state)).toEqual([
      'seq.sequences is a sequences, but chunk.aux takes a clip.',
      'down.clip has more than one input.',
    ]);
  });

  it('reports unknown node types and settings', () => {
    const state = graph();
    state.nodes[0].data.params.colour = 1;
    state.nodes.push({ id: 'old', position: { x: 0, y: 0 }, data: { type: 'gone', params: {}, enabled: true, extra: {} } });
    expect(messages(state)).toEqual([
      'Load Video (src) has unknown settings: colour.',
      'old is a node type this version does not have: gone.',
    ]);
  });

  it('needs a file on a source, and full paths until the job is saved', () => {
    const state = graph();
    state.nodes[0].data.params.path = '';
    state.nodes[4].data.params.directory = 'frames';
    expect(messages(state, true)).toEqual(['Load Video (src) has no file set.']);
    expect(messages(state, false)).toEqual([
      'Load Video (src) has no file set.',
      'Write Dataset (out), Directory: "frames" is a relative path. Save the job first, or give a full path.',
    ]);
    state.nodes[4].data.params.directory = '\\\\server\\share\\frames';
    expect(messages(state, false)).toEqual(['Load Video (src) has no file set.']);
  });

  it('needs a Bad Telecine Fix between a Telecine and the sequences, and a Telecine before every fix', () => {
    const job = (nodes: string[]): DatasetJob => {
      const chain = ['chunk', ...nodes];
      return {
        ...JOB,
        nodes: [...JOB.nodes.filter(n => n.id !== 'down'), ...nodes.map(type => ({ id: type, type, params: {} }))],
        edges: [
          ...JOB.edges.filter(e => !e.from.startsWith('down.') && !e.to.startsWith('down.')),
          ...nodes.map((type, i) => ({ from: `${chain[i]}.clip`, to: `${type}.clip` })),
          { from: `${chain[chain.length - 1]}.clip`, to: 'seq.lr' },
        ],
      };
    };
    expect(messages(graph(job(['telecine', 'telecine_fix'])))).toEqual([]);
    expect(messages(graph(job(['telecine'])))).toEqual([
      'Telecine (telecine) reaches Pick Sequences (seq) without a Bad Telecine Fix between them, so its frames no longer line up with HR.',
    ]);
    expect(messages(graph(job(['telecine_fix'])))).toEqual(['Bad Telecine Fix (telecine_fix) needs a Telecine before it.']);
    const off = graph(job(['telecine', 'telecine_fix']));
    off.nodes.find(n => n.id === 'telecine')!.data.enabled = false;
    expect(messages(off)).toEqual(['Bad Telecine Fix (telecine_fix) needs a Telecine before it.']);
  });

  it('reports a random setting the engine could not draw from', () => {
    const state = graph();
    state.nodes[2].data.params.down_up_scale = { uniform: [1, 0.5] };
    expect(messages(state)).toEqual(['Half (down), Down up scale: the low end of the range is above the high end.']);
  });
});
