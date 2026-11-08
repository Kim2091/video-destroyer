// src/utils/datasetGraph.ts — the Dataset workspace's graph, between the
// canvas and the engine's job JSON (include/vk_dataset/graph.py).
//
// Node types are never named here. Everything the editor knows about a node
// comes from the declarations `python -m vk_dataset nodes` prints, so a node
// added to the engine shows up in the palette with no change to the app.

import { autoLayout, chunkScope } from './datasetLayout';
import { linkedParam, VALUE_PORT } from './datasetLinks';
import { isSampler, samplerProblem } from './datasetSampler';

export interface DatasetParamSpec {
  name: string;
  /**
   * float, int, bool, str, path, choice, json, code (several lines of a
   * script), weights (one number per input, in order), and on value nodes,
   * value (any number, text or bool) or options (a choice sampler of them).
   */
  kind: string;
  default: unknown;
  label: string;
  help: string;
  /** A float or int param that may be given a sampler instead of a value (params.py). */
  random: boolean;
  choices?: unknown[];
  min?: number;
  max?: number;
  /**
   * A code setting's text per value of the choice `presetFrom` names: while
   * the setting is empty, it shows and runs the chosen value's text.
   */
  presets?: Record<string, string>;
  presetFrom?: string;
  /** Whether a value wire may drive the setting (links.py). */
  linkable?: boolean;
  /** On a value node: the setting holds what the node gives, so it is edited with the choices of the setting it feeds. */
  gives?: boolean;
}

/** What a port means, for its label on the card and its tooltip. */
export interface DatasetPortInfo {
  label: string;
  help: string;
}

export interface DatasetNodeSpec {
  type: string;
  category: string;
  label: string;
  /** One line for the node card and the palette. */
  description: string;
  /** What the node is for and where it goes, for the node's "?" popover. */
  help?: string;
  /** The stage of a job the node belongs to (docs.py), 1 Load to 8 Save. */
  stage?: number | null;
  stageName?: string;
  /** Other stages the node is commonly used in. */
  alsoStages?: number[];
  /** For a node with no stage, such as a value node: the palette group it is listed under. */
  group?: string;
  /** Port name → port type ("clip", "sequences"). */
  inputs: Record<string, string>;
  optionalInputs: string[];
  outputs: Record<string, string>;
  inputInfo?: Record<string, DatasetPortInfo>;
  outputInfo?: Record<string, DatasetPortInfo>;
  params: DatasetParamSpec[];
  chunker: boolean;
  /** A node type the output must pass through before it becomes sequences or is written. */
  needsAfter?: string;
  /** A node type that must be upstream of this node. */
  needsBefore?: string;
}

/** One stage of a job, as `python -m vk_dataset stages` prints it. */
export interface DatasetStage {
  stage: number;
  name: string;
  help: string;
}

export interface DatasetJobNode {
  id: string;
  type: string;
  params?: Record<string, unknown>;
  enabled?: boolean;
  label?: string;
  /** The node's place on the canvas. The engine never reads it. */
  ui?: { x: number; y: number };
  [key: string]: unknown;
}

export interface DatasetJobEdge {
  from: string;
  to: string;
  [key: string]: unknown;
}

/** A .vkdataset file: exactly the job the engine runs. */
export interface DatasetJob {
  version?: number;
  seed?: number;
  settings?: Record<string, unknown>;
  nodes: DatasetJobNode[];
  edges: DatasetJobEdge[];
  [key: string]: unknown;
}

export const JOB_VERSION = 1;

/** An example job from the presets folder, by the name it gives itself. */
export interface DatasetExample {
  name: string;
  job: DatasetJob;
}

/** An example's number and title, from a name like "3. video-destroyer classic"; unnumbered ones count from 1. */
export function exampleTitle(name: string, index: number): { number: number; title: string } {
  const match = /^\s*(\d+)[.)]\s*(.+)$/.exec(name);
  return match ? { number: Number(match[1]), title: match[2] } : { number: index + 1, title: name };
}

/** What a canvas node carries. `extra` holds job keys the editor does not edit, so they survive a save. */
export interface DatasetNodeData extends Record<string, unknown> {
  type: string;
  params: Record<string, unknown>;
  enabled: boolean;
  label?: string;
  extra: Record<string, unknown>;
}

/** The parts of a React Flow node and edge this module reads and writes. */
export interface GraphNode {
  id: string;
  type?: string;
  position: { x: number; y: number };
  data: DatasetNodeData;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string | null;
  targetHandle?: string | null;
}

export interface GraphState {
  seed: number;
  settings: Record<string, unknown>;
  /** Top-level job keys the editor does not edit. */
  extra: Record<string, unknown>;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface GraphIssue {
  message: string;
  nodeId?: string;
}

export const NODE_KIND = 'dataset';

export function specMap(specs: DatasetNodeSpec[]): Map<string, DatasetNodeSpec> {
  return new Map(specs.map(spec => [spec.type, spec]));
}

const clone = <T>(value: T): T => (value === undefined ? value : JSON.parse(JSON.stringify(value)));

export function defaultParams(spec: DatasetNodeSpec): Record<string, unknown> {
  return Object.fromEntries(spec.params.map(param => [param.name, clone(param.default)]));
}

/** The label a node shows: its own, else its type's. */
export function nodeLabel(data: DatasetNodeData, spec: DatasetNodeSpec | undefined): string {
  return data.label || spec?.label || data.type;
}

/** A node id no other node has: the type, then type_2, type_3, ... */
export function newNodeId(type: string, taken: Iterable<string>): string {
  const ids = new Set(taken);
  if (!ids.has(type)) return type;
  let n = 2;
  while (ids.has(`${type}_${n}`)) n++;
  return `${type}_${n}`;
}

/** Node id -> the input ports that have an edge. */
export function connectedInputs(edges: GraphEdge[]): Map<string, Set<string>> {
  const out = new Map<string, Set<string>>();
  for (const edge of edges) {
    if (!edge.targetHandle) continue;
    const ports = out.get(edge.target) ?? new Set<string>();
    ports.add(edge.targetHandle);
    out.set(edge.target, ports);
  }
  return out;
}

/** The most variants an enabled writer of the job writes; 1 when none has variants. */
export function jobVariants(job: DatasetJob): number {
  const counts = job.nodes
    .filter(n => n.enabled !== false)
    .map(n => n.params?.variants)
    .filter((v): v is number => typeof v === 'number' && Number.isInteger(v) && v > 1);
  return counts.length ? Math.max(...counts) : 1;
}

export function edgeId(source: string, sourcePort: string, target: string, targetPort: string): string {
  return `${source}.${sourcePort}->${target}.${targetPort}`;
}

export function createNode(spec: DatasetNodeSpec, position: { x: number; y: number }, taken: Iterable<string>): GraphNode {
  return {
    id: newNodeId(spec.type, taken),
    type: NODE_KIND,
    position,
    data: { type: spec.type, params: defaultParams(spec), enabled: true, extra: {} },
  };
}

/** Splits "node.port" at its last dot, as graph.py does. */
export function splitPortRef(ref: string): { node: string; port: string } | null {
  const at = ref.lastIndexOf('.');
  if (at <= 0 || at === ref.length - 1) return null;
  return { node: ref.slice(0, at), port: ref.slice(at + 1) };
}

/** A fresh job: no nodes, seed 0. */
export function emptyGraph(): GraphState {
  return { seed: 0, settings: {}, extra: {}, nodes: [], edges: [] };
}

const hasPosition = (node: DatasetJobNode) => !!node.ui && Number.isFinite(node.ui.x) && Number.isFinite(node.ui.y);

/** Reads a job into the editor. Edges that do not name a node and a port are dropped. */
export function jobToGraph(job: DatasetJob): GraphState {
  const { version: _version, seed, settings, nodes, edges, ...extra } = job;
  void _version;
  const placed = new Set<string>();
  const graphNodes: GraphNode[] = nodes.map(node => {
    const { id, type, params, enabled, label, ui, ...nodeExtra } = node;
    if (hasPosition(node)) placed.add(id);
    const position = hasPosition(node) ? { x: ui!.x, y: ui!.y } : { x: 0, y: 0 };
    return {
      id,
      type: NODE_KIND,
      position,
      data: {
        type,
        params: clone(params ?? {}),
        enabled: enabled ?? true,
        ...(label ? { label } : {}),
        extra: clone(nodeExtra),
      },
    };
  });
  const graphEdges: GraphEdge[] = [];
  for (const edge of edges) {
    const from = splitPortRef(edge.from);
    const to = splitPortRef(edge.to);
    if (!from || !to) continue;
    graphEdges.push({
      id: edgeId(from.node, from.port, to.node, to.port),
      source: from.node,
      sourceHandle: from.port,
      target: to.node,
      targetHandle: to.port,
    });
  }
  // Nodes saved without a position are placed by the auto layout.
  if (placed.size < graphNodes.length) {
    const layout = autoLayout(graphNodes, graphEdges);
    for (const node of graphNodes) if (!placed.has(node.id)) node.position = layout.get(node.id) ?? node.position;
  }
  return {
    seed: Number.isInteger(seed) ? seed! : 0,
    settings: clone(settings ?? {}),
    extra: clone(extra),
    nodes: graphNodes,
    edges: graphEdges,
  };
}

/** Whether any node of a job has a saved place on the canvas; a job written by hand has none. */
export function jobHasLayout(job: DatasetJob): boolean {
  return job.nodes.some(hasPosition);
}

/** Writes the editor's graph as the engine's job, with each node's position under `ui`. */
export function graphToJob(state: GraphState): DatasetJob {
  return {
    ...clone(state.extra),
    version: JOB_VERSION,
    seed: state.seed,
    settings: clone(state.settings),
    nodes: state.nodes.map(node => ({
      ...clone(node.data.extra),
      id: node.id,
      type: node.data.type,
      ...(node.data.label ? { label: node.data.label } : {}),
      enabled: node.data.enabled,
      params: clone(node.data.params),
      ui: { x: Math.round(node.position.x), y: Math.round(node.position.y) },
    })),
    edges: state.edges.map(edge => ({
      from: `${edge.source}.${edge.sourceHandle ?? ''}`,
      to: `${edge.target}.${edge.targetHandle ?? ''}`,
    })),
  };
}

/** Whether `from` can be reached by following edges forward from `start`. */
function reaches(edges: GraphEdge[], start: string, from: string): boolean {
  const seen = new Set<string>();
  const stack = [start];
  while (stack.length) {
    const id = stack.pop()!;
    if (id === from) return true;
    if (seen.has(id)) continue;
    seen.add(id);
    for (const edge of edges) if (edge.source === id) stack.push(edge.target);
  }
  return false;
}

export interface Connection {
  source: string | null;
  target: string | null;
  sourceHandle?: string | null;
  targetHandle?: string | null;
}

/**
 * Whether a new edge may be drawn: an output to an input of the same port
 * type, between two different nodes, without closing a loop. An input that
 * already has an edge may still be connected; connectEdge replaces its edge.
 */
export function canConnect(connection: Connection, nodes: GraphNode[], edges: GraphEdge[], specs: Map<string, DatasetNodeSpec>): boolean {
  const { source, target, sourceHandle, targetHandle } = connection;
  if (!source || !target || !sourceHandle || !targetHandle || source === target) return false;
  const sourceNode = nodes.find(n => n.id === source);
  const targetNode = nodes.find(n => n.id === target);
  if (!sourceNode || !targetNode) return false;
  const outType = specs.get(sourceNode.data.type)?.outputs[sourceHandle];
  const setting = linkedParam(targetHandle);
  const inType = setting !== null
    ? (specs.get(targetNode.data.type)?.params.some(p => p.name === setting && p.linkable) ? VALUE_PORT : undefined)
    : specs.get(targetNode.data.type)?.inputs[targetHandle];
  if (!outType || outType !== inType) return false;
  const others = edges.filter(e => !(e.target === target && e.targetHandle === targetHandle));
  return !reaches(others, target, source);
}

/** Adds an edge, replacing the one already on its input: an input takes one edge. */
export function connectEdge(edges: GraphEdge[], connection: Connection): GraphEdge[] {
  const { source, target, sourceHandle, targetHandle } = connection;
  if (!source || !target || !sourceHandle || !targetHandle) return edges;
  return [
    ...edges.filter(e => !(e.target === target && e.targetHandle === targetHandle)),
    { id: edgeId(source, sourceHandle, target, targetHandle), source, sourceHandle, target, targetHandle },
  ];
}

const isAbsolutePath = (value: string) => /^([a-zA-Z]:[\\/]|[\\/]{2}|\/)/.test(value);

/**
 * Everything that would stop the job, before it is sent: the rules graph.py
 * enforces, in the order it checks them, plus what the runner refuses at
 * the start (a source with no file) and malformed random settings. `saved`
 * says whether the job has a file; relative paths are read against its
 * folder, so an unsaved job needs full ones.
 */
export function validateGraph(state: GraphState, specs: Map<string, DatasetNodeSpec>, saved: boolean): GraphIssue[] {
  const issues: GraphIssue[] = [];
  const byId = new Map(state.nodes.map(n => [n.id, n]));
  const name = (node: GraphNode) => `${nodeLabel(node.data, specs.get(node.data.type))} (${node.id})`;

  const ids = new Set<string>();
  for (const node of state.nodes) {
    if (ids.has(node.id)) issues.push({ nodeId: node.id, message: `Two nodes have the id ${node.id}.` });
    ids.add(node.id);
    const spec = specs.get(node.data.type);
    if (!spec) {
      issues.push({ nodeId: node.id, message: `${node.id} is a node type this version does not have: ${node.data.type}.` });
      continue;
    }
    // graph.py takes a probability on any node, as the runner ignores it where it has no meaning.
    const known = new Set([...spec.params.map(p => p.name), 'probability']);
    const unknown = Object.keys(node.data.params).filter(key => !known.has(key)).sort();
    if (unknown.length) issues.push({ nodeId: node.id, message: `${name(node)} has unknown settings: ${unknown.join(', ')}.` });
    for (const param of spec.params) {
      const value = node.data.params[param.name] ?? param.default;
      if (param.random && isSampler(value)) {
        const problem = samplerProblem(value, param);
        if (problem) issues.push({ nodeId: node.id, message: `${name(node)}, ${param.label}: ${problem}` });
      }
      if (param.kind === 'path' && typeof value === 'string') {
        if (!value && spec.category === 'Source' && param.name === 'path' && node.data.enabled) {
          issues.push({ nodeId: node.id, message: `${name(node)} has no file set.` });
        } else if (value && !saved && !isAbsolutePath(value)) {
          issues.push({ nodeId: node.id, message: `${name(node)}, ${param.label}: "${value}" is a relative path. Save the job first, or give a full path.` });
        }
      }
    }
  }

  const sources = new Map<string, Set<string>>();
  const linksIn = new Map<string, Set<string>>();
  // Only edges that pass are followed for cycles, as graph.py stops at the first bad one.
  const valid: GraphEdge[] = [];
  for (const edge of state.edges) {
    const src = byId.get(edge.source);
    const dst = byId.get(edge.target);
    if (!src || !dst) {
      issues.push({ message: `An edge names a node that does not exist: ${src ? edge.target : edge.source}.` });
      continue;
    }
    const srcSpec = specs.get(src.data.type);
    const dstSpec = specs.get(dst.data.type);
    if (!srcSpec || !dstSpec) continue;
    const outType = srcSpec.outputs[edge.sourceHandle ?? ''];
    const inType = dstSpec.inputs[edge.targetHandle ?? ''];
    if (outType === undefined) {
      issues.push({ nodeId: src.id, message: `${name(src)} has no output ${edge.sourceHandle}.` });
      continue;
    }
    const setting = linkedParam(edge.targetHandle);
    if (setting !== null) {
      const problem = linkProblem(setting, outType, dstSpec, `${src.id}.${edge.sourceHandle}`, linksIn.get(dst.id));
      if (problem) {
        issues.push({ nodeId: dst.id, message: `${name(dst)}${problem}` });
        continue;
      }
      linksIn.set(dst.id, new Set([...(linksIn.get(dst.id) ?? []), setting]));
      valid.push(edge);
      continue;
    }
    if (inType === undefined) {
      issues.push({ nodeId: dst.id, message: `${name(dst)} has no input ${edge.targetHandle}.` });
      continue;
    }
    if (outType !== inType) {
      issues.push({ nodeId: dst.id, message: `${src.id}.${edge.sourceHandle} is a ${outType}, but ${dst.id}.${edge.targetHandle} takes a ${inType}.` });
      continue;
    }
    const taken = sources.get(dst.id) ?? new Set<string>();
    if (taken.has(edge.targetHandle!)) {
      issues.push({ nodeId: dst.id, message: `${dst.id}.${edge.targetHandle} has more than one input.` });
      continue;
    }
    taken.add(edge.targetHandle!);
    sources.set(dst.id, taken);
    valid.push(edge);
  }

  for (const node of state.nodes) {
    const spec = specs.get(node.data.type);
    if (!spec) continue;
    const connected = sources.get(node.id) ?? new Set<string>();
    const missing = Object.keys(spec.inputs).filter(port => !connected.has(port) && !spec.optionalInputs.includes(port));
    if (missing.length) issues.push({ nodeId: node.id, message: `${name(node)} needs an input on: ${missing.join(', ')}.` });
  }

  const cycle = findCycle({ ...state, edges: valid });
  if (cycle) issues.push({ nodeId: cycle[0], message: `The graph has a cycle through ${cycle.join(' -> ')}.` });
  issues.push(...givenValueIssues({ ...state, edges: valid }, specs));

  const chunkers = state.nodes.filter(n => specs.get(n.data.type)?.chunker);
  if (chunkers.length > 1) {
    for (const node of chunkers) issues.push({ nodeId: node.id, message: `A job can have only one chunk node; ${name(node)} is one of ${chunkers.length}.` });
  }

  const perChunk = chunkScope(state.nodes, valid, specs);
  for (const node of state.nodes) {
    const variants = node.data.params.variants;
    if (typeof variants === 'number' && variants > 1 && specs.get(node.data.type)?.params.some(p => p.name === 'variants') && !perChunk.has(node.id)) {
      issues.push({ nodeId: node.id, message: `${name(node)} writes variants, which needs a chunk node upstream.` });
    }
  }

  // Pairs of nodes that undo each other's change of the frame count, as registry.py declares them.
  const consumers = new Map<string, string[]>();
  const producers = new Map<string, string[]>();
  for (const edge of valid) {
    consumers.set(edge.source, [...(consumers.get(edge.source) ?? []), edge.target]);
    producers.set(edge.target, [...(producers.get(edge.target) ?? []), edge.source]);
  }
  const typeLabel = (type: string) => specs.get(type)?.label ?? type;
  for (const node of state.nodes) {
    const spec = specs.get(node.data.type);
    if (!spec || !node.data.enabled) continue;
    if (spec.needsAfter) {
      const queue = [...(consumers.get(node.id) ?? [])];
      const seen = new Set<string>();
      while (queue.length) {
        const reached = byId.get(queue.shift()!);
        const reachedSpec = reached && specs.get(reached.data.type);
        if (!reached || !reachedSpec || seen.has(reached.id)) continue;
        if (reached.data.type === spec.needsAfter && reached.data.enabled) continue;
        seen.add(reached.id);
        const outputs = Object.values(reachedSpec.outputs);
        if (!outputs.length || outputs.includes('sequences')) {
          issues.push({ nodeId: node.id, message: `${name(node)} reaches ${name(reached)} without a ${typeLabel(spec.needsAfter)} between them, so its frames no longer line up with HR.` });
          break;
        }
        queue.push(...(consumers.get(reached.id) ?? []));
      }
    }
    if (spec.needsBefore) {
      const queue = [...(producers.get(node.id) ?? [])];
      const seen = new Set<string>();
      let found = false;
      while (queue.length && !found) {
        const current = byId.get(queue.shift()!);
        if (!current || seen.has(current.id)) continue;
        seen.add(current.id);
        found = current.data.type === spec.needsBefore && current.data.enabled;
        queue.push(...(producers.get(current.id) ?? []));
      }
      if (!found) issues.push({ nodeId: node.id, message: `${name(node)} needs a ${typeLabel(spec.needsBefore)} before it.` });
    }
  }

  const known = state.nodes.filter(n => specs.has(n.data.type));
  if (known.length === state.nodes.length && !known.some(n => Object.keys(specs.get(n.data.type)!.outputs).length === 0)) {
    issues.push({ message: 'A job needs at least one writer: a node with no outputs.' });
  }
  return issues;
}

/** What is wrong with a link into a setting, worded as graph.py does after the node's name, or null. */
function linkProblem(setting: string, outType: string, spec: DatasetNodeSpec, from: string, linked?: Set<string>): string | null {
  const param = spec.params.find(p => p.name === setting);
  if (!param) return ` has no setting ${setting} to link.`;
  if (!param.linkable) return `, ${param.label}: a ${param.kind} setting cannot be linked.`;
  if (outType !== VALUE_PORT) return `: ${from} is a ${outType}, but a setting takes a value.`;
  if (linked?.has(setting)) return `, ${param.label}: has more than one link.`;
  return null;
}

/**
 * The values a value node can give that its settings say outright: the
 * options it picks from and the typed values it gives (its `gives`
 * settings, none of them linked). Null when it may give anything else, such
 * as a number it draws or works out, or when it is off.
 */
export function givenValues(node: GraphNode, spec: DatasetNodeSpec, edges: GraphEdge[]): unknown[] | null {
  const gives = spec.params.filter(p => p.gives);
  if (!gives.length || !node.data.enabled) return null;
  const values: unknown[] = [];
  for (const param of gives) {
    if (edges.some(e => e.target === node.id && linkedParam(e.targetHandle) === param.name)) return null;
    const value = node.data.params[param.name] ?? param.default;
    if (isSampler(value)) {
      if (!Array.isArray(value.choice)) return null;
      values.push(...value.choice);
    } else if (param.kind === 'options') {
      return null;
    } else {
      values.push(value);
    }
  }
  return values;
}

/** Whether two values are the same as links.py compares them: numbers as numbers, anything else by its text. */
export function sameValue(a: unknown, b: unknown): boolean {
  const number = (v: unknown) => (typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN);
  const x = number(a);
  const y = number(b);
  if (Number.isFinite(x) && Number.isFinite(y)) return Math.abs(x - y) <= 1e-9 * Math.max(1, Math.abs(x), Math.abs(y));
  return String(a) === String(b);
}

/**
 * A value node that gives a value a dropdown it feeds does not offer, or
 * nothing at all: one issue per link, naming the dropdown's options. The
 * engine checks the rest (ranges, numbers worked out) when the job loads.
 */
function givenValueIssues(state: GraphState, specs: Map<string, DatasetNodeSpec>): GraphIssue[] {
  const issues: GraphIssue[] = [];
  const byId = new Map(state.nodes.map(n => [n.id, n]));
  for (const edge of state.edges) {
    const setting = linkedParam(edge.targetHandle);
    const src = byId.get(edge.source);
    const dst = byId.get(edge.target);
    const srcSpec = src && specs.get(src.data.type);
    const dstSpec = dst && specs.get(dst.data.type);
    const param = dstSpec?.params.find(p => p.name === setting);
    if (!src || !srcSpec || !dst || param?.kind !== 'choice') continue;
    const values = givenValues(src, srcSpec, state.edges);
    if (!values) continue;
    const choices = param.choices ?? [];
    const target = `${nodeLabel(dst.data, dstSpec)}, ${param.label}`;
    const wrong = values.filter(v => !choices.some(c => sameValue(c, v)));
    if (values.length === 0) {
      issues.push({ nodeId: src.id, message: `${nodeLabel(src.data, srcSpec)} (${src.id}) has no options to give ${target}.` });
    } else if (wrong.length) {
      issues.push({
        nodeId: src.id,
        message: `${nodeLabel(src.data, srcSpec)} (${src.id}) gives ${wrong.map(v => `"${String(v)}"`).join(', ')}, which `
          + `${target} does not offer. Its options: ${choices.map(String).join(', ')}.`,
      });
    }
  }
  return issues;
}

/** One cycle in the graph as a list of node ids, first id repeated at the end, or null. */
export function findCycle(state: GraphState): string[] | null {
  const inputs = new Map<string, string[]>();
  for (const edge of state.edges) inputs.set(edge.target, [...(inputs.get(edge.target) ?? []), edge.source]);
  const mark = new Map<string, 'visiting' | 'done'>();
  const visit = (id: string, path: string[]): string[] | null => {
    const seen = mark.get(id);
    if (seen === 'done') return null;
    if (seen === 'visiting') return [...path.slice(path.indexOf(id)), id];
    mark.set(id, 'visiting');
    for (const src of inputs.get(id) ?? []) {
      const found = visit(src, [...path, id]);
      if (found) return found;
    }
    mark.set(id, 'done');
    return null;
  };
  for (const node of state.nodes) {
    const found = visit(node.id, []);
    if (found) return found.reverse();
  }
  return null;
}
