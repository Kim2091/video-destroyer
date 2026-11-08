import { describe, expect, it, vi } from 'vitest';
import { EventEmitter } from 'events';
import { PassThrough } from 'stream';

vi.mock('./logger', () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
}));

// The fake child's pid is not a real process, so teardown must never reach taskkill.
vi.mock('./processLifecycle', async importOriginal => ({
  ...(await importOriginal<typeof import('./processLifecycle')>()),
  terminateProcessTree: vi.fn(),
}));

import { DatasetPreviewSession, PreviewSupersededError } from './datasetPreviewSession';

function fakeChild() {
  const child = new EventEmitter() as any;
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();
  const written: string[] = [];
  child.stdin = {
    write(chunk: string, cb?: (error?: Error | null) => void) {
      written.push(chunk);
      cb?.(null);
      return true;
    },
    end() {},
  };
  child.pid = 4242;
  child.commands = () => written.map(line => JSON.parse(line.trim()));
  child.reply = (header: Record<string, unknown>, payload = Buffer.alloc(0)) => {
    const body = Buffer.from(JSON.stringify({ ...header, bytes: payload.length }), 'utf8');
    const length = Buffer.alloc(4);
    length.writeUInt32BE(body.length, 0);
    child.stdout.write(Buffer.concat([length, body, payload]));
  };
  return child;
}

function started() {
  const child = fakeChild();
  const session = new DatasetPreviewSession((() => child) as never);
  session.start({ command: 'python', args: [], options: {} });
  return { child, session };
}

const picture = { node: 'down', port: 'clip', kind: 'clip', scope: 'chunk', n: 0, frames: 1, offset: 0, width: 1, height: 1, format: 'RGB24' };
const request = { node: 'down', port: 'clip', video: 0, chunk: 0, frame: 0, compare: null, view: null, compareView: null, seed: null };

describe('DatasetPreviewSession', () => {
  it('sends commands with a seq and resolves each with its own reply', async () => {
    const { child, session } = started();
    const loaded = session.load({ nodes: [], edges: [] }, 'C:/jobs');
    const chunks = session.chunks(0, null);
    const [load, chunk] = child.commands();
    expect(load).toMatchObject({ cmd: 'load', cwd: 'C:/jobs' });
    expect(chunk).toMatchObject({ cmd: 'chunks', video: 0, seed: null });
    // Replies are matched by seq, not by order.
    child.reply({ type: 'chunks', seq: chunk.seq, video: 0, name: 'clip', chunked: true, chunks: [[0, 5]], messages: [] });
    child.reply({ type: 'loaded', seq: load.seq, seed: 3, videos: [{ index: 0, name: 'clip' }], reused: 0 });
    expect(await loaded).toEqual({ seed: 3, videos: [{ index: 0, name: 'clip' }], reused: 0 });
    expect((await chunks).chunks).toEqual([[0, 5]]);
  });

  it('returns frames with their pixels', async () => {
    const { child, session } = started();
    const frame = session.frame(request);
    const [command] = child.commands();
    expect(command).toMatchObject({ cmd: 'frame', node: 'down', port: 'clip', compare: null });
    child.reply({ type: 'frame', seq: command.seq, ...picture, settings: {}, skipped: [] }, Buffer.from([9, 8, 7]));
    expect(Array.from((await frame).data)).toEqual([9, 8, 7]);
  });

  it('rejects an error reply with its message, and a superseded one with PreviewSupersededError', async () => {
    const { child, session } = started();
    const failed = session.frame(request);
    const replaced = session.frame(request);
    const [first, second] = child.commands();
    child.reply({ type: 'error', seq: first.seq, error: 'there is no chunk 4' });
    child.reply({ type: 'superseded', seq: second.seq });
    await expect(failed).rejects.toThrow('there is no chunk 4');
    await expect(replaced).rejects.toBeInstanceOf(PreviewSupersededError);
  });

  it('fails what is pending when the server exits, naming its last words', async () => {
    const { child, session } = started();
    const pending = session.chunks(0, null);
    child.stderr.write('Traceback...\nMemoryError\n');
    await new Promise(resolve => setImmediate(resolve));
    child.emit('exit', 1, null);
    await expect(pending).rejects.toThrow('The preview stopped (code 1): MemoryError');
    expect(session.isRunning).toBe(false);
    await expect(session.chunks(0, null)).rejects.toThrow(/not running/);
  });

  it('asks the server to close and ends its tree on dispose', async () => {
    const { terminateProcessTree } = await import('./processLifecycle');
    const { child, session } = started();
    const pending = session.chunks(0, null);
    session.dispose();
    await expect(pending).rejects.toThrow('The preview was closed');
    expect(child.commands().at(-1)).toMatchObject({ cmd: 'close' });
    expect(terminateProcessTree).toHaveBeenCalledWith(child);
  });
});
