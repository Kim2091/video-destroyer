// electron/datasetProtocol.ts — what the app and the vk_dataset engine say to
// each other (include/vk_dataset, copied to data/config/vk_dataset).
//
// Imports nothing from Electron, so the reader, the schemas and the command
// line are testable on their own. datasetHandlers.ts does the spawning.

import { z } from 'zod';
import { LineSplitter } from './lineSplitter';

/**
 * The job as the engine reads it (graph.py). Only the shape is checked here;
 * graph.py checks the node types, ports and rules. Unknown keys are kept, so
 * a node's `ui` (its place on the canvas) and settings this app does not edit
 * reach the file unchanged.
 */
export const datasetJobSchema = z.looseObject({
  version: z.number().int().optional(),
  seed: z.number().int().optional(),
  settings: z.record(z.string(), z.unknown()).optional(),
  nodes: z.array(z.looseObject({
    id: z.string().min(1),
    type: z.string().min(1),
    params: z.record(z.string(), z.unknown()).optional(),
    enabled: z.boolean().optional(),
    label: z.string().optional(),
  })),
  edges: z.array(z.looseObject({ from: z.string().min(1), to: z.string().min(1) })),
});

export type DatasetJobFile = z.infer<typeof datasetJobSchema>;

/** A run or check: the job, and the file it was saved to, which relative paths in it are read against. */
export const datasetRunRequestSchema = z.object({
  job: datasetJobSchema,
  jobPath: z.string().min(1).nullable(),
});

export const datasetSaveRequestSchema = z.object({
  job: datasetJobSchema,
  filePath: z.string().min(1).nullable(),
});

export type DatasetEvent =
  | { kind: 'progress'; progress: number; message: string }
  | { kind: 'message'; message: string };

/**
 * Reads the engine's stdout, one JSON object per line: {"progress", "message"}
 * or a bare {"message"} while it works, then one {"result"} or {"error"}.
 * Lines that are not JSON are VapourSynth or plugin chatter, passed on as is.
 */
export class DatasetOutputReader {
  private readonly lines = new LineSplitter(line => this.line(line));
  result: unknown = undefined;
  error: string | null = null;

  constructor(
    private readonly onEvent: (event: DatasetEvent) => void,
    private readonly onOther: (line: string) => void = () => {},
  ) {}

  get finished(): boolean {
    return this.result !== undefined || this.error !== null;
  }

  push(chunk: string): void {
    this.lines.push(chunk);
  }

  /** Reads a last line that had no newline after it. */
  end(): void {
    this.lines.flush();
  }

  private line(line: string): void {
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch {
      this.onOther(line);
      return;
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      this.onOther(line);
      return;
    }
    const message = parsed as Record<string, unknown>;
    if (typeof message.error === 'string') this.error = message.error;
    else if ('result' in message) this.result = message.result;
    else if (typeof message.progress === 'number') {
      this.onEvent({
        kind: 'progress',
        progress: Math.min(1, Math.max(0, message.progress)),
        message: String(message.message ?? ''),
      });
    } else if (typeof message.message === 'string') {
      this.onEvent({ kind: 'message', message: message.message });
    }
  }
}

/**
 * The app's embedded Python ignores PYTHONPATH and the working folder (its
 * ._pth file fixes sys.path), so `python -m vk_dataset` cannot find the
 * package. This puts the folder holding it on sys.path, then runs the
 * engine's own command line with the remaining arguments.
 */
const BOOTSTRAP = 'import sys; sys.path.insert(0, sys.argv.pop(1)); '
  + 'from vk_dataset.__main__ import main; sys.exit(main())';

/** Python arguments for one engine command, with the package found in `packageParent`. */
export function datasetEngineArgs(packageParent: string, command: 'nodes' | 'stages' | 'check' | 'run' | 'preview', jobFile?: string): string[] {
  return ['-u', '-c', BOOTSTRAP, packageParent, command, ...(jobFile ? [jobFile] : [])];
}

/** A preset's display name, from its file name: "hr_lr-pairs.json" → "Hr lr pairs". */
export function presetName(fileName: string): string {
  const words = fileName.replace(/\.json$/i, '').replace(/[_-]+/g, ' ').trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}
