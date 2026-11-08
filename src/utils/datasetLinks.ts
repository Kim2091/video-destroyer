// src/utils/datasetLinks.ts — settings driven by value wires (links.py):
// the handle a link arrives at on a setting's row, and which nodes are value
// nodes. Imports only types, so the layout can use it as well as the graph.

import type { DatasetNodeSpec, DatasetParamSpec, GraphEdge, GraphNode } from './datasetGraph';

/**
 * The handle a setting's link arrives at: "param:<name>". An edge into it
 * carries a value node's output into the setting, and is saved like any
 * edge, as {"from": "pick.value", "to": "comp.param:profile"}.
 */
export const PARAM_PORT = 'param:';
export const paramHandle = (name: string) => `${PARAM_PORT}${name}`;

/** The setting a handle names, or null for an ordinary port. */
export const linkedParam = (handle: string | null | undefined): string | null =>
  handle?.startsWith(PARAM_PORT) ? handle.slice(PARAM_PORT.length) : null;

/** Whether an edge links a value into a setting rather than joining two ports. */
export const isLink = (edge: GraphEdge) => linkedParam(edge.targetHandle) !== null;

/** The port type value nodes give and settings take. */
export const VALUE_PORT = 'value';

/** Whether a node type is a value node: its only output is a value. */
export const isValueSpec = (spec: DatasetNodeSpec | undefined): boolean =>
  !!spec && Object.keys(spec.outputs).length > 0 && Object.values(spec.outputs).every(type => type === VALUE_PORT);

/**
 * The value nodes of a graph: those `specs` declares as value nodes, and
 * any other whose only edges are links, at least one of them its own.
 */
export function valueNodeIds(nodes: GraphNode[], edges: GraphEdge[], specs?: Map<string, DatasetNodeSpec>): Set<string> {
  const gives = new Set(edges.filter(isLink).map(e => e.source));
  const joined = new Set(edges.filter(e => !isLink(e)).flatMap(e => [e.source, e.target]));
  return new Set(nodes.filter(n => isValueSpec(specs?.get(n.data.type)) || (gives.has(n.id) && !joined.has(n.id))).map(n => n.id));
}

/**
 * The setting each value node's value goes into, for editing what it gives
 * with that setting's choices. A link into a setting of another value node
 * that gives what it is given (an If Equals "then") is followed on to where
 * that node's value goes. Of several, a dropdown wins, then a setting of a
 * fixed kind, then the first link.
 */
export function valueTargets(nodes: GraphNode[], edges: GraphEdge[], specs: Map<string, DatasetNodeSpec>): Map<string, DatasetParamSpec> {
  const types = new Map(nodes.map(n => [n.id, n.data.type]));
  const links = new Map<string, GraphEdge[]>();
  for (const edge of edges) if (isLink(edge)) links.set(edge.source, [...(links.get(edge.source) ?? []), edge]);
  const reached = (id: string, seen: Set<string>): DatasetParamSpec[] => {
    if (seen.has(id)) return [];
    return (links.get(id) ?? []).flatMap(edge => {
      const param = specs.get(types.get(edge.target) ?? '')?.params.find(p => p.name === linkedParam(edge.targetHandle));
      if (!param) return [];
      return param.gives ? reached(edge.target, new Set([...seen, id])) : [param];
    });
  };
  const out = new Map<string, DatasetParamSpec>();
  for (const id of links.keys()) {
    const params = reached(id, new Set());
    const best = params.find(p => p.kind === 'choice') ?? params.find(p => p.kind !== 'value') ?? params[0];
    if (best) out.set(id, best);
  }
  return out;
}
