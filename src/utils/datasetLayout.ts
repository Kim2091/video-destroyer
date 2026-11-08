// src/utils/datasetLayout.ts — where nodes sit on the dataset canvas: the
// left-to-right layout Arrange applies, the place a node added from the
// palette goes and the edge that joins it, and the region of nodes that run
// once per chunk.
//
// Value nodes and their links into settings (datasetLinks.ts) stay off the
// chains: they sit above the nodes they feed and never count as a step.
//
// Nothing is imported from datasetGraph.ts but types, so it can use the
// layout for jobs saved without positions.

import type { DatasetNodeSpec, GraphEdge, GraphNode } from './datasetGraph';
import { isLink, valueNodeIds } from './datasetLinks';

/** Every card is this wide, so columns line up. */
export const CARD_WIDTH = 232;
/** The distance from one column to the next. */
export const COLUMN = CARD_WIDTH + 56;
/** The height assumed for a card that has not been measured yet. */
export const CARD_HEIGHT = 150;
const ROW_GAP = 48;
/** The most columns in one band; a longer graph wraps onto further bands, like lines of text. */
export const BAND_COLUMNS = 6;
const BAND_GAP = 96;

export interface Point {
  x: number;
  y: number;
}

export interface Rect extends Point {
  width: number;
  height: number;
}

type Lane = 'hr' | 'lr';

/** The path a port name belongs to: hr and hr_* ports to HR, lr and lr_* to LR. */
function portLane(port: string | null | undefined): Lane | null {
  if (!port) return null;
  if (/^lr(_|$)/i.test(port)) return 'lr';
  if (/^hr(_|$)/i.test(port)) return 'hr';
  return null;
}

/**
 * How far along its chain each node is: 0 for a node that reads from
 * nothing, otherwise one more than the furthest node it reads from. Links
 * into settings are not followed and value nodes have no depth, so a value
 * wire never moves a node along its chain.
 */
export function chainDepth(nodes: GraphNode[], edges: GraphEdge[], specs?: Map<string, DatasetNodeSpec>): Map<string, number> {
  const values = valueNodeIds(nodes, edges, specs);
  const ids = new Set(nodes.filter(n => !values.has(n.id)).map(n => n.id));
  const inputs = new Map<string, string[]>();
  for (const edge of edges) {
    if (isLink(edge) || !ids.has(edge.source) || !ids.has(edge.target) || edge.source === edge.target) continue;
    inputs.set(edge.target, [...(inputs.get(edge.target) ?? []), edge.source]);
  }
  const depths = new Map<string, number>();
  const visiting = new Set<string>();
  const depthOf = (id: string): number => {
    const known = depths.get(id);
    if (known !== undefined) return known;
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const value = Math.max(-1, ...(inputs.get(id) ?? []).map(depthOf)) + 1;
    visiting.delete(id);
    depths.set(id, value);
    return value;
  };
  for (const node of nodes) if (ids.has(node.id)) depthOf(node.id);
  return depths;
}

export interface LayoutOptions {
  /** Orders nodes with no edges by stage. */
  specs?: Map<string, DatasetNodeSpec>;
  /** A card's measured height, when it has one. */
  heightOf?: (id: string) => number | undefined;
}

/**
 * Lays a graph out left to right. Each node goes one column after the
 * furthest node it reads from. Nodes on the HR path fill the top rows and
 * nodes on the LR path the rows under them: a node is on the LR path when
 * all of its edges leave from LR outputs, or when every edge out of it goes
 * into an LR input or into another node on the LR path. Nodes with no edges
 * go in a last row, in stage order. A graph more than BAND_COLUMNS columns
 * long wraps onto bands of even length, each under the one before and one
 * column further right, so the whole job fits on screen and still reads from
 * top left to bottom right. Value nodes go in rows above each band, a column
 * left of the leftmost node they feed, or of the value node they feed.
 */
export function autoLayout(nodes: GraphNode[], edges: GraphEdge[], options: LayoutOptions = {}): Map<string, Point> {
  const ids = new Set(nodes.map(n => n.id));
  const order = new Map(nodes.map((n, i) => [n.id, i]));
  const values = valueNodeIds(nodes, edges, options.specs);
  const live = edges.filter(e => ids.has(e.source) && ids.has(e.target) && e.source !== e.target && !isLink(e));
  const inputs = new Map<string, string[]>();
  const outputs = new Map<string, GraphEdge[]>();
  for (const edge of live) {
    inputs.set(edge.target, [...(inputs.get(edge.target) ?? []), edge.source]);
    outputs.set(edge.source, [...(outputs.get(edge.source) ?? []), edge]);
  }

  const depths = chainDepth(nodes, edges, options.specs);
  const columnOf = (id: string) => depths.get(id) ?? 0;

  const lanes = new Map<string, Lane>();
  const inLane = new Set<string>();
  const laneOf = (id: string): Lane => {
    const known = lanes.get(id);
    if (known) return known;
    if (inLane.has(id)) return 'hr';
    inLane.add(id);
    const out = outputs.get(id) ?? [];
    const named = [...new Set(out.map(e => portLane(e.sourceHandle)))];
    let lane: Lane;
    if (named.length === 1 && named[0]) {
      lane = named[0];
    } else {
      const votes = out.map(e => portLane(e.targetHandle) ?? laneOf(e.target));
      lane = votes.length > 0 && votes.every(v => v === 'lr') ? 'lr' : 'hr';
    }
    inLane.delete(id);
    lanes.set(id, lane);
    return lane;
  };

  const linked = nodes.filter(n => !values.has(n.id) && (inputs.has(n.id) || outputs.has(n.id)));
  const loose = nodes.filter(n => !values.has(n.id) && !inputs.has(n.id) && !outputs.has(n.id));
  const stageOf = (node: GraphNode) => options.specs?.get(node.data.type)?.stage ?? 99;

  // Slots per lane and column, in the order the job lists the nodes.
  const slot = new Map<string, number>();
  const used = new Map<string, number>();
  const rowsIn: Record<Lane, number> = { hr: 0, lr: 0 };
  for (const node of linked) {
    const lane = laneOf(node.id);
    const key = `${lane}:${columnOf(node.id)}`;
    const n = used.get(key) ?? 0;
    used.set(key, n + 1);
    slot.set(node.id, n);
    rowsIn[lane] = Math.max(rowsIn[lane], n + 1);
  }

  const columnCount = Math.max(0, ...linked.map(n => columnOf(n.id) + 1));
  const bandCount = Math.max(1, Math.ceil(columnCount / BAND_COLUMNS));
  const perBand = Math.max(1, Math.ceil(columnCount / bandCount));

  // Value nodes are placed nearest first: those feeding a chain node, then
  // those feeding them. Each goes one column left of the leftmost place it
  // feeds, or further left in that band when that spot is taken, and only
  // then on a further row.
  const feeds = new Map<string, string[]>();
  for (const edge of edges) {
    if (isLink(edge) && values.has(edge.source) && ids.has(edge.target) && edge.source !== edge.target) {
      feeds.set(edge.source, [...(feeds.get(edge.source) ?? []), edge.target]);
    }
  }
  const placed = new Set(linked.map(n => n.id));
  const hops = new Map<string, number>();
  const hopsOf = (id: string, seen: Set<string>): number => {
    if (hops.has(id)) return hops.get(id)!;
    if (seen.has(id)) return Infinity;
    const next = new Set([...seen, id]);
    const found = Math.min(Infinity, ...(feeds.get(id) ?? []).map(t => (values.has(t) ? hopsOf(t, next) + 1 : placed.has(t) ? 1 : Infinity)));
    hops.set(id, found);
    return found;
  };
  const valueCells = new Map<string, { column: number; slot: number }>();
  const taken = new Set<string>();
  const queue = nodes.filter(n => values.has(n.id)).map(n => n.id).sort((a, b) => hopsOf(a, new Set()) - hopsOf(b, new Set()));
  let valueRows = 0;
  for (const id of queue) {
    const targets = (feeds.get(id) ?? [])
      .map(t => (values.has(t) ? valueCells.get(t)?.column : placed.has(t) ? columnOf(t) : undefined))
      .filter((column): column is number => column !== undefined);
    if (!targets.length) {
      loose.push(nodes.find(n => n.id === id)!);
      continue;
    }
    const target = Math.min(...targets);
    const start = Math.floor(target / perBand) * perBand;
    const want = Math.max(start, target - 1);
    let spot: { column: number; slot: number } | null = null;
    for (let slot = 0; !spot; slot++) {
      for (let column = want; column >= start && !spot; column--) {
        if (!taken.has(`${column}:${slot}`)) spot = { column, slot };
      }
    }
    taken.add(`${spot.column}:${spot.slot}`);
    valueCells.set(id, spot);
    valueRows = Math.max(valueRows, spot.slot + 1);
  }
  loose.sort((a, b) => stageOf(a) - stageOf(b) || order.get(a.id)! - order.get(b.id)!);
  const bandRows = valueRows + rowsIn.hr + rowsIn.lr;

  const cells = new Map<string, { row: number; column: number }>();
  for (const node of linked) {
    const lane = laneOf(node.id);
    const column = columnOf(node.id);
    const band = Math.floor(column / perBand);
    cells.set(node.id, {
      row: band * bandRows + valueRows + (lane === 'lr' ? rowsIn.hr : 0) + slot.get(node.id)!,
      column: column % perBand + band,
    });
  }
  for (const [id, { column, slot: n }] of valueCells) {
    const band = Math.floor(column / perBand);
    cells.set(id, { row: band * bandRows + n, column: column % perBand + band });
  }
  const looseRow = bandCount * bandRows;
  loose.forEach((node, i) => cells.set(node.id, { row: looseRow + Math.floor(i / BAND_COLUMNS), column: i % BAND_COLUMNS }));

  // Each row is as tall as its tallest card; a row left empty in a band takes no room.
  const rowCount = Math.max(0, ...[...cells.values()].map(c => c.row + 1));
  const heights = Array.from({ length: rowCount }, () => 0);
  for (const [id, cell] of cells) heights[cell.row] = Math.max(heights[cell.row], options.heightOf?.(id) ?? CARD_HEIGHT);
  const tops: number[] = [];
  heights.reduce((y, height, row) => {
    const startsBand = row > 0 && bandRows > 0 && row <= looseRow && row % bandRows === 0;
    const top = y + (startsBand ? BAND_GAP : 0);
    tops[row] = top;
    return height ? top + height + ROW_GAP : top;
  }, 0);

  return new Map([...cells].map(([id, cell]) => [id, { x: cell.column * COLUMN, y: tops[cell.row] }]));
}

const overlaps = (a: Rect, b: Rect, margin = 16) =>
  a.x < b.x + b.width + margin && b.x < a.x + a.width + margin && a.y < b.y + b.height + margin && b.y < a.y + a.height + margin;

/** The rectangle a card covers. */
export function cardRect(node: GraphNode, heightOf?: (id: string) => number | undefined): Rect {
  return { x: node.position.x, y: node.position.y, width: CARD_WIDTH, height: heightOf?.(node.id) ?? CARD_HEIGHT };
}

/**
 * Where a node added from the palette goes: one column right of the
 * selected node, else right of the last node of an earlier stage, moved
 * down until it covers no other card.
 */
export function placeNewNode(
  spec: DatasetNodeSpec,
  nodes: GraphNode[],
  specs: Map<string, DatasetNodeSpec>,
  selectedId: string | null,
  heightOf?: (id: string) => number | undefined,
): Point {
  if (nodes.length === 0) return { x: 0, y: 0 };
  const selected = selectedId ? nodes.find(n => n.id === selectedId) : undefined;
  let at: Point;
  if (selected) {
    at = { x: selected.position.x + COLUMN, y: selected.position.y };
  } else {
    const stage = spec.stage ?? 99;
    const before = nodes.filter(n => (specs.get(n.data.type)?.stage ?? 99) < stage);
    at = {
      x: before.length ? Math.max(...before.map(n => n.position.x)) + COLUMN : Math.min(...nodes.map(n => n.position.x)),
      y: Math.min(...nodes.map(n => n.position.y)),
    };
  }
  const rects = nodes.map(n => cardRect(n, heightOf));
  for (let tries = 0; tries < 200; tries++) {
    const rect = { ...at, width: CARD_WIDTH, height: CARD_HEIGHT };
    const hit = rects.filter(r => overlaps(rect, r));
    if (hit.length === 0) break;
    at = { x: at.x, y: Math.max(...hit.map(r => r.y + r.height)) + ROW_GAP };
  }
  return at;
}

/**
 * The edge that joins a new node to the selected one, when there is only
 * one way to draw it: the selected node has one output that nothing reads
 * yet and whose type the new node takes, and the new node has one input of
 * that type (counting only required inputs when it has several). An output
 * named like an optional input that has no edge carries nothing, so it is
 * not counted.
 */
export function autoConnect(
  spec: DatasetNodeSpec,
  selected: GraphNode | null,
  edges: GraphEdge[],
  specs: Map<string, DatasetNodeSpec>,
): { sourceHandle: string; targetHandle: string } | null {
  const from = selected ? specs.get(selected.data.type) : undefined;
  if (!selected || !from) return null;
  const pairs: { sourceHandle: string; targetHandle: string }[] = [];
  for (const [port, type] of Object.entries(from.outputs)) {
    if (edges.some(e => e.source === selected.id && e.sourceHandle === port)) continue;
    if (from.optionalInputs.includes(port) && !edges.some(e => e.target === selected.id && e.targetHandle === port)) continue;
    let fits = Object.entries(spec.inputs).filter(([, inType]) => inType === type).map(([name]) => name);
    if (fits.length > 1) fits = fits.filter(name => !spec.optionalInputs.includes(name));
    if (fits.length === 1) pairs.push({ sourceHandle: port, targetHandle: fits[0] });
    else if (fits.length > 1) return null;
  }
  return pairs.length === 1 ? pairs[0] : null;
}

/**
 * The nodes that run once per chunk, as graph.py works them out: the chunk
 * node and every node that reads from one of them, directly or not, and
 * each value node whose every link goes into a node that runs per chunk
 * other than the chunk node, which cuts the video once. A value node that
 * feeds nothing is left out, as it changes nothing.
 */
export function chunkScope(nodes: GraphNode[], edges: GraphEdge[], specs: Map<string, DatasetNodeSpec>): Set<string> {
  const chunkers = new Set(nodes.filter(n => specs.get(n.data.type)?.chunker).map(n => n.id));
  const scope = new Set(chunkers);
  const stack = [...scope];
  while (stack.length) {
    const id = stack.pop()!;
    for (const edge of edges) {
      if (edge.source === id && !scope.has(edge.target) && !isLink(edge)) {
        scope.add(edge.target);
        stack.push(edge.target);
      }
    }
  }
  const values = valueNodeIds(nodes, edges, specs);
  const feeds = new Map<string, string[]>();
  for (const edge of edges) {
    if (isLink(edge) && values.has(edge.source)) feeds.set(edge.source, [...(feeds.get(edge.source) ?? []), edge.target]);
  }
  for (let grew = true; grew;) {
    grew = false;
    for (const [id, targets] of feeds) {
      if (!scope.has(id) && targets.every(t => scope.has(t) && !chunkers.has(t))) {
        scope.add(id);
        grew = true;
      }
    }
  }
  return scope;
}

/** The rectangle around the given cards, with room for a label above, or null when there are none. */
export function boundsOf(
  nodes: GraphNode[],
  ids: Set<string>,
  heightOf?: (id: string) => number | undefined,
  pad = 24,
  label = 28,
): Rect | null {
  const rects = nodes.filter(n => ids.has(n.id)).map(n => cardRect(n, heightOf));
  if (rects.length === 0) return null;
  const left = Math.min(...rects.map(r => r.x)) - pad;
  const top = Math.min(...rects.map(r => r.y)) - pad - label;
  const right = Math.max(...rects.map(r => r.x + r.width)) + pad;
  const bottom = Math.max(...rects.map(r => r.y + r.height)) + pad;
  return { x: left, y: top, width: right - left, height: bottom - top };
}
