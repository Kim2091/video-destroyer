// electron/datasetPreviewSession.ts — one long-lived dataset preview server
// (python -m vk_dataset preview) and the requests sent to it.
//
// Replies carry the sequence number of their request. The server answers a
// frame request that a newer one overtook with "superseded", which rejects
// with PreviewSupersededError so the caller can tell it from a failure.

import { spawn, type ChildProcess, type SpawnOptions } from 'child_process';
import { logger } from './logger';
import { terminateProcessTree } from './processLifecycle';
import {
  PreviewFrameReader,
  frameFromReply,
  type DatasetPreviewChunks,
  type DatasetPreviewFrame,
  type DatasetPreviewFrameRequest,
  type DatasetPreviewLoaded,
  type PreviewReply,
} from './datasetPreviewProtocol';
import type { DatasetJobFile } from './datasetProtocol';

/** How much of the server's stderr is kept for explaining an exit. */
const STDERR_TAIL = 16 * 1024;

export class PreviewSupersededError extends Error {
  constructor() {
    super('A newer preview request replaced this one');
    this.name = 'PreviewSupersededError';
  }
}

export interface DatasetPreviewLaunch {
  command: string;
  args: string[];
  options: SpawnOptions;
}

export class DatasetPreviewSession {
  private child: ChildProcess | null = null;
  private seq = 0;
  private readonly pending = new Map<number, { resolve: (reply: PreviewReply) => void; reject: (error: Error) => void }>();
  private stderrTail = '';
  private exitReason: string | null = null;
  private exited: Promise<void> = Promise.resolve();

  /** Injected so the protocol can be tested against a fake child. */
  constructor(private readonly spawner: typeof spawn = spawn) {}

  get isRunning(): boolean {
    return this.child !== null;
  }

  start(launch: DatasetPreviewLaunch): void {
    if (this.child) return;
    this.exitReason = null;
    this.stderrTail = '';
    const reader = new PreviewFrameReader(reply => this.onReply(reply));
    const child = this.spawner(launch.command, launch.args, { ...launch.options, stdio: ['pipe', 'pipe', 'pipe'] });
    this.exited = new Promise(resolve => {
      child.once('exit', () => resolve());
      child.once('error', () => resolve());
    });
    child.stdout?.on('data', (chunk: Buffer) => reader.push(chunk));
    child.stderr?.on('data', (chunk: Buffer) => {
      const text = chunk.toString();
      this.stderrTail = (this.stderrTail + text).slice(-STDERR_TAIL);
      const trimmed = text.trimEnd();
      if (trimmed) logger.debug(`[dataset-preview] ${trimmed}`);
    });
    child.on('exit', (code, signal) => {
      if (this.child !== child) return;
      this.exitReason = signal ? `signal ${signal}` : `code ${code}`;
      this.child = null;
      const last = this.stderrTail.trim().split(/\r?\n/).filter(Boolean).pop();
      this.failAllPending(new Error(`The preview stopped (${this.exitReason})${last ? `: ${last}` : ''}`));
    });
    child.on('error', error => {
      if (this.child !== child) return;
      this.child = null;
      this.failAllPending(new Error(`The preview could not start: ${error.message}`));
    });
    this.child = child;
  }

  async load(job: DatasetJobFile, cwd: string): Promise<DatasetPreviewLoaded> {
    const { header } = await this.send({ cmd: 'load', job, cwd });
    return { seed: header.seed, videos: header.videos, reused: header.reused };
  }

  async chunks(video: number, seed: number | null): Promise<DatasetPreviewChunks> {
    const { header } = await this.send({ cmd: 'chunks', video, seed });
    return { video: header.video, name: header.name, chunked: header.chunked, chunks: header.chunks, messages: header.messages ?? [] };
  }

  async frame(request: DatasetPreviewFrameRequest): Promise<DatasetPreviewFrame> {
    const { header, payload } = await this.send({ cmd: 'frame', ...request });
    return frameFromReply(header, payload);
  }

  /** Stops the server; the promise settles once its process has exited. */
  dispose(): Promise<void> {
    const child = this.child;
    this.child = null;
    this.failAllPending(new Error('The preview was closed'));
    if (!child) return Promise.resolve();
    try {
      child.stdin?.write(`${JSON.stringify({ cmd: 'close', seq: -1 })}\n`);
      child.stdin?.end();
    } catch {
      // The pipe is already gone.
    }
    // A request that renders a long chunk does not read the close; end the tree.
    terminateProcessTree(child);
    return this.exited;
  }

  private send(command: Record<string, unknown>): Promise<PreviewReply> {
    const child = this.child;
    if (!child?.stdin) {
      return Promise.reject(new Error(this.exitReason ? `The preview is not running (it stopped with ${this.exitReason})` : 'The preview is not running'));
    }
    const seq = ++this.seq;
    return new Promise<PreviewReply>((resolve, reject) => {
      this.pending.set(seq, { resolve, reject });
      child.stdin!.write(`${JSON.stringify({ ...command, seq })}\n`, error => {
        if (!error) return;
        this.pending.delete(seq);
        reject(new Error(`Could not reach the preview: ${error.message}`));
      });
    });
  }

  private onReply(reply: PreviewReply): void {
    const seq: number = reply.header.seq ?? -1;
    const waiter = this.pending.get(seq);
    if (!waiter) return;
    this.pending.delete(seq);
    if (reply.header.type === 'error') waiter.reject(new Error(String(reply.header.error ?? 'The preview failed')));
    else if (reply.header.type === 'superseded') waiter.reject(new PreviewSupersededError());
    else waiter.resolve(reply);
  }

  private failAllPending(error: Error): void {
    for (const waiter of this.pending.values()) waiter.reject(error);
    this.pending.clear();
  }
}
