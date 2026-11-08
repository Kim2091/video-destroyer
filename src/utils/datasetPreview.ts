// src/utils/datasetPreview.ts — the pure parts of the Dataset workspace's
// preview panel: which outputs can be shown, how a frame is laid out and
// zoomed, and which sequence crops sit on the frame shown.

import type { DatasetPreviewSequences } from '../../electron/datasetPreviewProtocol';
import type { DatasetJob, DatasetNodeSpec, GraphNode } from './datasetGraph';
import { nodeLabel } from './datasetGraph';

/** Output port types the preview can draw. */
export const PREVIEW_PORT_TYPES = ['clip', 'sequences', 'mask', 'signal'];

export interface PortRef {
  node: string;
  port: string;
}

export interface PreviewPort extends PortRef {
  type: string;
  label: string;
}

export const samePort = (a: PortRef | null, b: PortRef | null) =>
  a === b || (!!a && !!b && a.node === b.node && a.port === b.port);

/**
 * The job as the preview loads it: without the canvas positions and notes,
 * so moving a node or writing a note is not an edit the preview reloads for.
 */
export function previewJob(job: DatasetJob): DatasetJob {
  return { ...job, nodes: job.nodes.map(({ ui: _ui, note: _note, ...node }) => { void _ui; void _note; return node; }) };
}

/** Every output the preview can show, in canvas order, labelled "Node label · port". */
export function previewPorts(nodes: GraphNode[], specs: Map<string, DatasetNodeSpec>): PreviewPort[] {
  const ports: PreviewPort[] = [];
  for (const node of nodes) {
    const spec = specs.get(node.data.type);
    if (!spec) continue;
    const outputs = Object.entries(spec.outputs).filter(([, type]) => PREVIEW_PORT_TYPES.includes(type));
    for (const [port, type] of outputs) {
      const label = nodeLabel(node.data, spec);
      ports.push({ node: node.id, port, type, label: outputs.length > 1 ? `${label} · ${port}` : label });
    }
  }
  return ports;
}

/**
 * The output to show for a node picked on the canvas: the port already shown
 * when it is that node's, else its first previewable port, else null.
 */
export function portForNode(nodeId: string, current: PortRef | null, ports: PreviewPort[]): PortRef | null {
  if (current?.node === nodeId && ports.some(p => samePort(p, current))) return current;
  const first = ports.find(p => p.node === nodeId);
  return first ? { node: first.node, port: first.port } : null;
}

/** A port that still exists in the graph, else null. */
export function existingPort(ref: PortRef | null, ports: PreviewPort[]): PortRef | null {
  return ref && ports.some(p => samePort(p, ref)) ? ref : null;
}

/** Packed RGB24 to the RGBA an ImageData takes. */
export function rgbToRgba(rgb: Uint8Array, width: number, height: number): Uint8ClampedArray<ArrayBuffer> {
  const pixels = width * height;
  if (rgb.length < pixels * 3) throw new Error(`${rgb.length} bytes is not a ${width}x${height} RGB picture`);
  const out = new Uint8ClampedArray(pixels * 4);
  for (let i = 0, j = 0; i < pixels; i++, j += 3) {
    const k = i * 4;
    out[k] = rgb[j];
    out[k + 1] = rgb[j + 1];
    out[k + 2] = rgb[j + 2];
    out[k + 3] = 255;
  }
  return out;
}

export type CompareMode = 'side' | 'wipe';
export type ZoomMode = 'fit' | 1 | 2;

export interface Size {
  width: number;
  height: number;
}

/**
 * The picture drawn: the frame, or with a compare picture, both at the
 * frame's size (the compare one scaled to it) side by side or one over the
 * other for a wipe.
 */
export function compositeSize(main: Size, compare: boolean, mode: CompareMode): Size {
  return compare && mode === 'side' ? { width: main.width * 2, height: main.height } : main;
}

/** Screen pixels per picture pixel: fit the box (up or down), or a fixed zoom. */
export function zoomScale(zoom: ZoomMode, picture: Size, box: Size): number {
  if (zoom !== 'fit') return zoom;
  if (picture.width <= 0 || picture.height <= 0 || box.width <= 0 || box.height <= 0) return 1;
  return Math.min(box.width / picture.width, box.height / picture.height);
}

/**
 * Keeps a pan offset (from the centred position) inside the picture: along
 * an axis where the scaled picture fits the box it stays centred, else it
 * can move until an edge meets the box's edge.
 */
export function clampPan(pan: { x: number; y: number }, scaled: Size, box: Size): { x: number; y: number } {
  const limit = (picture: number, room: number) => Math.max(0, (picture - room) / 2);
  const clamp = (value: number, max: number) => (max === 0 ? 0 : Math.min(max, Math.max(-max, value)));
  return {
    x: clamp(pan.x, limit(scaled.width, box.width)),
    y: clamp(pan.y, limit(scaled.height, box.height)),
  };
}

export interface CropOverlay {
  /** Index in the kept or the rejected list. */
  index: number;
  rejected: boolean;
  reason: string | null;
  /** [left, top, width, height] in the shown clip's pixels; the whole frame when the sequence has no crop. */
  rect: [number, number, number, number];
}

/** The sequences, kept and rejected, that include frame `n` of clip `view`, with their crops. */
export function cropsAt(sequences: DatasetPreviewSequences, view: string, n: number): CropOverlay[] {
  const clip = sequences.clips[view];
  if (!clip) return [];
  const whole: [number, number, number, number] = [0, 0, clip.width, clip.height];
  const overlays: CropOverlay[] = [];
  const add = (list: DatasetPreviewSequences['items'], rejected: boolean) => list.forEach((item, index) => {
    if (!item.frames[view]?.includes(n)) return;
    const reason = rejected ? String(item.info?.rejected ?? 'rejected') : null;
    overlays.push({ index, rejected, reason, rect: item.crops[view] ?? whole });
  });
  add(sequences.items, false);
  add(sequences.rejected, true);
  return overlays;
}

/** A drawn setting for the settings list: numbers to 4 decimals, structures as compact JSON. */
export function formatDrawn(value: unknown): string {
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(4)));
  if (typeof value === 'string') return value;
  if (value === null || value === undefined) return '—';
  return JSON.stringify(value, (_key, inner) => (typeof inner === 'number' && !Number.isInteger(inner) ? Number(inner.toFixed(4)) : inner));
}

/** "Frames 120–239" for a chunk range (end exclusive), or the whole video. */
export function rangeLabel(range: [number, number] | null): string {
  return range ? `frames ${range[0]}–${range[1] - 1}` : 'whole video';
}

/** A seed for a reroll, in the range the job's seed field offers. */
export function randomSeed(): number {
  return Math.floor(Math.random() * 2 ** 31);
}
