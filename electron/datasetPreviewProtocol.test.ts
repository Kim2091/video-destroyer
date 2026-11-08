import { describe, expect, it } from 'vitest';
import {
  PreviewFrameReader,
  datasetPreviewFrameSchema,
  frameFromReply,
  type PreviewReply,
} from './datasetPreviewProtocol';

function encode(header: Record<string, unknown>, payload = Buffer.alloc(0)): Buffer {
  const body = Buffer.from(JSON.stringify({ ...header, bytes: payload.length }), 'utf8');
  const length = Buffer.alloc(4);
  length.writeUInt32BE(body.length, 0);
  return Buffer.concat([length, body, payload]);
}

function collect(chunks: Buffer[]): PreviewReply[] {
  const replies: PreviewReply[] = [];
  const reader = new PreviewFrameReader(reply => replies.push(reply));
  for (const chunk of chunks) reader.push(chunk);
  return replies;
}

describe('PreviewFrameReader', () => {
  const pixels = Buffer.from([1, 2, 3, 4, 5, 6]);

  it('reads replies with and without payloads packed into one chunk', () => {
    const replies = collect([Buffer.concat([encode({ type: 'frame', seq: 1 }, pixels), encode({ type: 'ok', seq: 2 })])]);
    expect(replies.map(r => r.header.seq)).toEqual([1, 2]);
    expect(replies[0].payload).toEqual(pixels);
    expect(replies[1].payload).toBeNull();
  });

  it('survives every split point, down to one byte at a time', () => {
    const wire = Buffer.concat([encode({ type: 'frame', seq: 1 }, pixels), encode({ type: 'frame', seq: 2 }, pixels)]);
    for (let split = 1; split < wire.length; split++) {
      const replies = collect([wire.subarray(0, split), wire.subarray(split)]);
      expect(replies.map(r => r.header.seq)).toEqual([1, 2]);
    }
    const bytes = collect(Array.from({ length: wire.length }, (_, i) => wire.subarray(i, i + 1)));
    expect(bytes[1].payload).toEqual(pixels);
  });
});

describe('frameFromReply', () => {
  const header = {
    type: 'frame', seq: 4, bytes: 2 * 1 * 3 + 1 * 1 * 3,
    node: 'down', port: 'clip', kind: 'clip', scope: 'chunk', n: 0, frames: 5, offset: 0, width: 2, height: 1, format: 'YUV420P8',
    video: 0, chunk: 1, range: [10, 15], seed: 7, settings: {}, skipped: [], cached: false, messages: [], ms: 3,
    compare: { node: 'src', port: 'clip', kind: 'clip', scope: 'video', n: 0, frames: 5, offset: 10, width: 1, height: 1, format: 'YUV420P8' },
  };

  it('splits the payload into the picture and the compare picture, dropping transport fields', () => {
    const frame = frameFromReply(header, Buffer.from([1, 2, 3, 4, 5, 6, 7, 8, 9]));
    expect(Array.from(frame.data)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(Array.from(frame.compareData!)).toEqual([7, 8, 9]);
    expect(frame).not.toHaveProperty('seq');
    expect(frame).not.toHaveProperty('type');
    expect(frame.range).toEqual([10, 15]);
  });

  it('has no compare data without a compare picture', () => {
    const { compare: _compare, ...single } = header;
    void _compare;
    const frame = frameFromReply(single, Buffer.from([1, 2, 3, 4, 5, 6]));
    expect(frame.compareData).toBeUndefined();
  });

  it('refuses a payload that does not match the sizes in the header', () => {
    expect(() => frameFromReply(header, Buffer.from([1, 2, 3]))).toThrow(/3 bytes of pixels; its header says 9/);
    expect(() => frameFromReply(header, null)).toThrow(/0 bytes/);
  });
});

describe('datasetPreviewFrameSchema', () => {
  const request = { node: 'down', port: 'clip', video: 0, chunk: 0, frame: 0, compare: null, view: null, compareView: null, seed: null };

  it('takes a full request', () => {
    expect(datasetPreviewFrameSchema.safeParse(request).success).toBe(true);
    expect(datasetPreviewFrameSchema.safeParse({ ...request, compare: { node: 'src', port: 'clip' }, seed: 12 }).success).toBe(true);
  });

  it('refuses negative or fractional positions and empty ids', () => {
    expect(datasetPreviewFrameSchema.safeParse({ ...request, frame: -1 }).success).toBe(false);
    expect(datasetPreviewFrameSchema.safeParse({ ...request, chunk: 1.5 }).success).toBe(false);
    expect(datasetPreviewFrameSchema.safeParse({ ...request, node: '' }).success).toBe(false);
    expect(datasetPreviewFrameSchema.safeParse({ ...request, compare: { node: 'src' } }).success).toBe(false);
  });
});
