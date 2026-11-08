import { describe, expect, it } from 'vitest';
import {
  DatasetOutputReader,
  datasetEngineArgs,
  datasetJobSchema,
  datasetRunRequestSchema,
  presetName,
  type DatasetEvent,
} from './datasetProtocol';

describe('reading the dataset engine', () => {
  it('reports progress and bare messages, passes chatter on, and keeps the result', () => {
    const events: DatasetEvent[] = [];
    const other: string[] = [];
    const reader = new DatasetOutputReader(e => events.push(e), line => other.push(line));

    reader.push('{"message": "a: 3 chunks"}\nScript evaluation done\r\n{"progress": 0.25, "mess');
    reader.push('age": "a: chunk 1 of 3"}\n{"result": {"videos": 1, "sequences": 12}}');
    expect(reader.finished).toBe(false);
    reader.end();

    expect(events).toEqual([
      { kind: 'message', message: 'a: 3 chunks' },
      { kind: 'progress', progress: 0.25, message: 'a: chunk 1 of 3' },
    ]);
    expect(other).toEqual(['Script evaluation done']);
    expect(reader.result).toEqual({ videos: 1, sequences: 12 });
    expect(reader.finished).toBe(true);
    expect(reader.error).toBeNull();
  });

  it('keeps the error, and a result that is not an object', () => {
    const failed = new DatasetOutputReader(() => {});
    failed.push('{"error": "a job needs at least one writer node"}\n');
    expect(failed.error).toBe('a job needs at least one writer node');
    expect(failed.finished).toBe(true);

    const checked = new DatasetOutputReader(() => {});
    checked.push('{"result": "ok"}\n');
    expect(checked.result).toBe('ok');
  });

  it('clamps progress into 0..1 and treats JSON that is not an object as chatter', () => {
    const seen: number[] = [];
    const other: string[] = [];
    const reader = new DatasetOutputReader(e => { if (e.kind === 'progress') seen.push(e.progress); }, l => other.push(l));
    reader.push('{"progress": 1.5}\n{"progress": -2}\n42\n[1]\n');
    expect(seen).toEqual([1, 0]);
    expect(other).toEqual(['42', '[1]']);
  });
});

describe('the dataset engine command line', () => {
  it('puts the package folder on sys.path before the engine reads its arguments', () => {
    const args = datasetEngineArgs('C:/vk/data/config', 'run', 'C:/tmp/job.json');
    expect(args.slice(0, 2)).toEqual(['-u', '-c']);
    expect(args[2]).toContain('sys.path.insert(0, sys.argv.pop(1))');
    expect(args.slice(3)).toEqual(['C:/vk/data/config', 'run', 'C:/tmp/job.json']);
    expect(datasetEngineArgs('/c', 'nodes')).toHaveLength(5);
  });
});

describe('job validation at the IPC boundary', () => {
  const job = {
    version: 1,
    seed: 7,
    nodes: [{ id: 'src', type: 'load_video', params: { path: 'a.mkv' }, enabled: true, ui: { x: 10, y: 20 } }],
    edges: [{ from: 'src.clip', to: 'w.clip' }],
  };

  it('accepts a job and keeps keys it does not know', () => {
    const parsed = datasetRunRequestSchema.parse({ job: { ...job, note: 'x' }, jobPath: null });
    expect(parsed.job.nodes[0]).toMatchObject({ ui: { x: 10, y: 20 } });
    expect((parsed.job as Record<string, unknown>).note).toBe('x');
  });

  it('rejects a node without an id or type, a bad edge, and a fractional seed', () => {
    expect(datasetJobSchema.safeParse({ ...job, nodes: [{ type: 'x' }] }).success).toBe(false);
    expect(datasetJobSchema.safeParse({ ...job, nodes: [{ id: 'a', type: '' }] }).success).toBe(false);
    expect(datasetJobSchema.safeParse({ ...job, edges: [{ from: 'a.b' }] }).success).toBe(false);
    expect(datasetJobSchema.safeParse({ ...job, seed: 1.5 }).success).toBe(false);
    expect(datasetRunRequestSchema.safeParse({ job }).success).toBe(false);
  });
});

describe('preset names', () => {
  it('come from the file name', () => {
    expect(presetName('video_sr-pairs.json')).toBe('Video sr pairs');
    expect(presetName('Basic.JSON')).toBe('Basic');
  });
});
