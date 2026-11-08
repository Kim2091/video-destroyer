// src/hooks/useDatasetPreview.ts — the Dataset workspace's preview panel and
// the warm preview server behind it (include/vk_dataset/preview.py).
//
// While the panel is open the job is loaded into the server, again after
// each edit once the edits pause; moving nodes on the canvas is not an edit.
// Frame requests are sent as the selection changes and only the newest
// answer is shown: the server skips requests a newer one overtook. Closing
// the panel, or leaving the workspace, stops the server.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { DatasetPreviewFrame, DatasetPreviewVideo } from '../../electron/datasetPreviewProtocol';
import type { DatasetJob } from '../utils/datasetGraph';
import { previewJob, samePort, type PortRef } from '../utils/datasetPreview';

/** How long edits must pause before the job is loaded again. */
const RELOAD_DELAY_MS = 350;

export interface DatasetPreviewSelection {
  target: PortRef | null;
  compare: PortRef | null;
  /** The clip shown for a sequences port; null is its first. */
  view: string | null;
  compareView: string | null;
  video: number;
  chunk: number;
  /** Frame within the chunk. */
  frame: number;
  /** A seed that replaces the job's for the preview's draws; null uses the job's. */
  seed: number | null;
  /** Which variant of a writer with variants to draw for; 0 is the first. */
  variant: number;
}

type LoadState =
  | { status: 'idle' }
  | { status: 'loading'; videos: DatasetPreviewVideo[] }
  | { status: 'ready'; videos: DatasetPreviewVideo[]; version: number }
  | { status: 'error'; error: string; videos: DatasetPreviewVideo[] };

type ChunksState =
  | { status: 'idle' | 'loading' }
  | { status: 'ready'; chunked: boolean; chunks: ([number, number] | null)[]; key: string }
  | { status: 'error'; error: string };

export type FrameState =
  | { status: 'idle' }
  | { status: 'loading'; frame: DatasetPreviewFrame | null }
  | { status: 'ready'; frame: DatasetPreviewFrame }
  | { status: 'error'; error: string; frame: DatasetPreviewFrame | null };

const INITIAL: DatasetPreviewSelection = {
  target: null, compare: null, view: null, compareView: null, video: 0, chunk: 0, frame: 0, seed: null, variant: 0,
};

const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error));

export function useDatasetPreview({ enabled, job, jobPath }: { enabled: boolean; job: DatasetJob; jobPath: string | null }) {
  const [selection, setSelection] = useState<DatasetPreviewSelection>(INITIAL);
  const [load, setLoad] = useState<LoadState>({ status: 'idle' });
  const [chunks, setChunks] = useState<ChunksState>({ status: 'idle' });
  const [frame, setFrame] = useState<FrameState>({ status: 'idle' });
  const [retries, setRetries] = useState(0);
  const loadVersion = useRef(0);
  const frameToken = useRef(0);

  const jobText = useMemo(() => JSON.stringify(previewJob(job)), [job]);

  useEffect(() => {
    if (!enabled) return;
    return () => {
      void window.electronAPI.datasetPreviewClose();
      setLoad({ status: 'idle' });
      setChunks({ status: 'idle' });
      setFrame({ status: 'idle' });
    };
  }, [enabled]);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    setLoad(current => ({ status: 'loading', videos: current.status === 'idle' ? [] : current.videos }));
    const first = loadVersion.current === 0;
    const timer = setTimeout(() => {
      window.electronAPI.datasetPreviewLoad({ job: JSON.parse(jobText) as DatasetJob, jobPath })
        .then(outcome => {
          if (cancelled) return;
          if (outcome.success) {
            loadVersion.current += 1;
            setLoad({ status: 'ready', videos: outcome.value.videos, version: loadVersion.current });
          } else {
            setLoad({ status: 'error', error: outcome.error, videos: [] });
          }
        })
        .catch(error => { if (!cancelled) setLoad({ status: 'error', error: errorText(error), videos: [] }); });
    }, first ? 0 : RELOAD_DELAY_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [enabled, jobText, jobPath, retries]);

  const version = load.status === 'ready' ? load.version : 0;
  const videoCount = load.status === 'ready' ? load.videos.length : 0;

  useEffect(() => {
    if (videoCount > 0 && selection.video >= videoCount) setSelection(s => ({ ...s, video: 0, chunk: 0, frame: 0 }));
  }, [videoCount, selection.video]);

  const chunksKey = `${version}:${selection.video}:${selection.seed}`;
  useEffect(() => {
    if (!version || selection.video >= videoCount) return;
    let cancelled = false;
    setChunks({ status: 'loading' });
    window.electronAPI.datasetPreviewChunks({ video: selection.video, seed: selection.seed })
      .then(outcome => {
        if (cancelled) return;
        if (!outcome.success) {
          setChunks({ status: 'error', error: outcome.error });
          return;
        }
        const list = outcome.value.chunks;
        setChunks({ status: 'ready', chunked: outcome.value.chunked, chunks: list, key: chunksKey });
        setSelection(s => (s.chunk >= list.length ? { ...s, chunk: 0, frame: 0 } : s));
      })
      .catch(error => { if (!cancelled) setChunks({ status: 'error', error: errorText(error) }); });
    return () => { cancelled = true; };
    // chunksKey stands for version, video and seed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chunksKey, videoCount]);

  const chunksReady = chunks.status === 'ready' && chunks.key === chunksKey;
  const chunkCount = chunks.status === 'ready' ? chunks.chunks.length : 0;
  const { target, compare, view, compareView, video, chunk, frame: frameIndex, seed, variant } = selection;
  useEffect(() => {
    if (!enabled || !version || !chunksReady || !target || chunk >= chunkCount) return;
    const token = ++frameToken.current;
    setFrame(current => ({ status: 'loading', frame: current.status === 'idle' ? null : current.frame }));
    window.electronAPI.datasetPreviewFrame({
      node: target.node, port: target.port, video, chunk, frame: frameIndex,
      compare: compare && !samePort(compare, target) ? compare : null, view, compareView, seed, variant,
    })
      .then(outcome => {
        if (token !== frameToken.current) return;
        if (outcome.success) setFrame({ status: 'ready', frame: outcome.value });
        else if (!outcome.superseded) setFrame(current => ({ status: 'error', error: outcome.error, frame: 'frame' in current ? current.frame : null }));
      })
      .catch(error => {
        if (token === frameToken.current) setFrame(current => ({ status: 'error', error: errorText(error), frame: 'frame' in current ? current.frame : null }));
      });
  }, [enabled, version, chunksReady, chunkCount, target, compare, view, compareView, video, chunk, frameIndex, seed, variant]);

  const update = useCallback((change: Partial<DatasetPreviewSelection>) => setSelection(s => ({ ...s, ...change })), []);
  const retry = useCallback(() => setRetries(n => n + 1), []);

  return { selection, update, load, chunks, frame, retry };
}
