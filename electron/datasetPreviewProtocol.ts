// electron/datasetPreviewProtocol.ts — what the app and the dataset preview
// server (include/vk_dataset/preview.py) say to each other.
//
// Commands go out as JSON lines; replies come back as
// {uint32 big-endian header length}{header JSON}{payload}, where a frame's
// payload is its packed RGB24 picture followed by the compare picture, if
// one was asked for. Imports nothing from Electron, so the reader and the
// schemas are testable on their own.

import { z } from 'zod';
import { datasetJobSchema } from './datasetProtocol';

export interface PreviewReply {
  header: Record<string, any>;
  payload: Buffer | null;
}

type ReadState = 'length' | 'header' | 'payload';

/**
 * Incremental reader for {uint32 length}{header}{payload}. Every byte is
 * copied once, into a buffer allocated when its size becomes known, so a
 * large frame split over many pipe chunks costs one copy.
 */
export class PreviewFrameReader {
  private state: ReadState = 'length';
  private readonly lengthBuf = Buffer.alloc(4);
  private lengthFilled = 0;
  private headerBuf: Buffer | null = null;
  private headerFilled = 0;
  private payloadBuf: Buffer | null = null;
  private payloadFilled = 0;
  private header: Record<string, any> | null = null;

  constructor(private readonly onReply: (reply: PreviewReply) => void) {}

  push(chunk: Buffer): void {
    let offset = 0;
    while (offset < chunk.length) {
      if (this.state === 'length') {
        const take = Math.min(4 - this.lengthFilled, chunk.length - offset);
        chunk.copy(this.lengthBuf, this.lengthFilled, offset, offset + take);
        this.lengthFilled += take;
        offset += take;
        if (this.lengthFilled === 4) {
          this.headerBuf = Buffer.allocUnsafe(this.lengthBuf.readUInt32BE(0));
          this.headerFilled = 0;
          this.state = 'header';
        }
        continue;
      }

      if (this.state === 'header') {
        const target = this.headerBuf!;
        const take = Math.min(target.length - this.headerFilled, chunk.length - offset);
        chunk.copy(target, this.headerFilled, offset, offset + take);
        this.headerFilled += take;
        offset += take;
        if (this.headerFilled === target.length) {
          this.header = JSON.parse(target.toString('utf8'));
          const bytes: number = this.header?.bytes ?? 0;
          if (bytes > 0) {
            this.payloadBuf = Buffer.allocUnsafe(bytes);
            this.payloadFilled = 0;
            this.state = 'payload';
          } else {
            this.deliver(null);
          }
        }
        continue;
      }

      const target = this.payloadBuf!;
      const take = Math.min(target.length - this.payloadFilled, chunk.length - offset);
      chunk.copy(target, this.payloadFilled, offset, offset + take);
      this.payloadFilled += take;
      offset += take;
      if (this.payloadFilled === target.length) this.deliver(target);
    }
  }

  private deliver(payload: Buffer | null): void {
    const header = this.header!;
    this.state = 'length';
    this.lengthFilled = 0;
    this.headerBuf = null;
    this.payloadBuf = null;
    this.header = null;
    this.onReply({ header, payload });
  }
}

const portRefSchema = z.object({ node: z.string().min(1), port: z.string().min(1) });
const seedSchema = z.number().int().nullable();

/** Loads (or reloads, after an edit) the job the preview shows; relative paths are read against jobPath's folder. */
export const datasetPreviewLoadSchema = z.object({
  job: datasetJobSchema,
  jobPath: z.string().min(1).nullable(),
});

export const datasetPreviewChunksSchema = z.object({
  video: z.number().int().min(0),
  seed: seedSchema,
});

export const datasetPreviewFrameSchema = z.object({
  node: z.string().min(1),
  port: z.string().min(1),
  video: z.number().int().min(0),
  chunk: z.number().int().min(0),
  frame: z.number().int().min(0),
  compare: portRefSchema.nullable(),
  /** For a sequences port, the clip to show ("hr", "lr"); null shows the first. */
  view: z.string().min(1).nullable(),
  compareView: z.string().min(1).nullable(),
  /** Replaces the job seed for this frame's draws; null uses the job's. */
  seed: seedSchema,
  /** Which of a writer's variants to draw for; 0, the default, is what a job without variants draws. */
  variant: z.number().int().min(0).max(15).optional(),
});

export type DatasetPreviewFrameRequest = z.infer<typeof datasetPreviewFrameSchema>;

export interface DatasetPreviewVideo {
  index: number;
  name: string;
}

export interface DatasetPreviewLoaded {
  seed: number;
  videos: DatasetPreviewVideo[];
  /** Nodes whose cached outputs survived the reload. */
  reused: number;
}

export interface DatasetPreviewChunks {
  video: number;
  name: string;
  /** False when the job has no chunk node; its one chunk is then the whole video (a null range). */
  chunked: boolean;
  chunks: ([number, number] | null)[];
  messages: string[];
}

/** One picture of a frame reply: which output it is and how big. */
export interface DatasetPreviewPicture {
  node: string;
  port: string;
  kind: string;
  /** "video" for a per-video node, shown over the chunk's frames from `offset`. */
  scope: 'chunk' | 'video';
  /** The clip shown, and the clips there are, for a sequences port. */
  view?: string;
  clips?: string[];
  n: number;
  frames: number;
  offset: number;
  width: number;
  height: number;
  format: string;
}

export interface DatasetPreviewSequence {
  /** Clip name → frame numbers within the chunk. */
  frames: Record<string, number[]>;
  /** Clip name → [left, top, width, height]; no entry is the whole frame. */
  crops: Record<string, [number, number, number, number]>;
  /** What nodes noted; a rejected sequence has its reason in `rejected`. */
  info: Record<string, unknown>;
}

export interface DatasetPreviewSequences {
  clips: Record<string, { width: number; height: number; frames: number }>;
  items: DatasetPreviewSequence[];
  rejected: DatasetPreviewSequence[];
}

/** What a node drew for the chunk, as the manifest records it. */
export interface DatasetDrawnSettings {
  type: string;
  applied: boolean;
  params?: Record<string, unknown>;
}

export interface DatasetPreviewFrame extends DatasetPreviewPicture {
  video: number;
  chunk: number;
  range: [number, number] | null;
  seed: number;
  variant: number;
  settings: Record<string, DatasetDrawnSettings>;
  /** Upstream nodes not applied on this chunk (disabled, or their probability said no). */
  skipped: string[];
  /** Every enabled value node's value for the chunk, or why it failed, by node id. */
  values?: Record<string, { value?: unknown; error?: string }>;
  /** True when the chunk's values were already computed. */
  cached: boolean;
  messages: string[];
  /** Server time for this frame, in milliseconds. */
  ms: number;
  sequences?: DatasetPreviewSequences;
  compare?: DatasetPreviewPicture;
  /** Packed RGB24, width * height * 3 bytes. */
  data: Uint8Array;
  compareData?: Uint8Array;
}

const pictureBytes = (picture: { width: number; height: number }) => picture.width * picture.height * 3;

/** Splits a frame reply into its header fields and its one or two pictures. */
export function frameFromReply(header: Record<string, any>, payload: Buffer | null): DatasetPreviewFrame {
  const { type: _type, seq: _seq, bytes: _bytes, ...fields } = header;
  void _type; void _seq; void _bytes;
  const main = pictureBytes(header as DatasetPreviewPicture);
  const second = header.compare ? pictureBytes(header.compare as DatasetPreviewPicture) : 0;
  if (!payload || payload.length !== main + second) {
    throw new Error(`The preview frame has ${payload?.length ?? 0} bytes of pixels; its header says ${main + second}`);
  }
  return {
    ...(fields as Omit<DatasetPreviewFrame, 'data' | 'compareData'>),
    data: payload.subarray(0, main),
    ...(second ? { compareData: payload.subarray(main) } : {}),
  };
}
