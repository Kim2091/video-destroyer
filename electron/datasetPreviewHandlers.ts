// electron/datasetPreviewHandlers.ts — the Dataset workspace's preview panel,
// main-process side.
//
// One preview server at a time (datasetPreviewSession.ts), started by the
// first load and kept warm while the panel is open, so a chunk's work is
// cached between frames. It is closed when the panel or the workspace
// closes, and on quit. A server that stopped is started again by the next
// load.

import { z } from 'zod';
import { PATHS } from './constants';
import { logger } from './logger';
import { createWorkloadSpawnOptions } from './processLifecycle';
import { handleValidated } from './ipcValidation';
import { datasetEngineEnv, engineMissing, jobFolder, type DatasetOutcome } from './datasetHandlers';
import { datasetEngineArgs, type DatasetJobFile } from './datasetProtocol';
import {
  datasetPreviewChunksSchema,
  datasetPreviewFrameSchema,
  datasetPreviewLoadSchema,
  type DatasetPreviewChunks,
  type DatasetPreviewFrame,
  type DatasetPreviewFrameRequest,
  type DatasetPreviewLoaded,
} from './datasetPreviewProtocol';
import { DatasetPreviewSession, PreviewSupersededError } from './datasetPreviewSession';

export type DatasetPreviewFrameOutcome =
  | { success: true; value: DatasetPreviewFrame }
  | { success: false; error: string; superseded?: boolean };

let session: DatasetPreviewSession | null = null;

const failure = (error: unknown) => ({ success: false as const, error: error instanceof Error ? error.message : String(error) });

function runningSession(): DatasetPreviewSession {
  if (!session?.isRunning) throw new Error('The preview is not running; load the job again');
  return session;
}

export async function loadPreview(job: DatasetJobFile, jobPath: string | null): Promise<DatasetOutcome<DatasetPreviewLoaded>> {
  const missing = engineMissing();
  if (missing) return { success: false, error: missing };
  try {
    if (!session?.isRunning) {
      session = new DatasetPreviewSession();
      session.start({
        command: PATHS.PYTHON,
        args: datasetEngineArgs(PATHS.CONFIG, 'preview'),
        options: createWorkloadSpawnOptions({ cwd: PATHS.CONFIG, env: datasetEngineEnv(), windowsHide: true }),
      });
      logger.info('Started the dataset preview');
    }
    return { success: true, value: await session.load(job, jobFolder(jobPath)) };
  } catch (error) {
    return failure(error);
  }
}

export async function previewChunks(video: number, seed: number | null): Promise<DatasetOutcome<DatasetPreviewChunks>> {
  try {
    return { success: true, value: await runningSession().chunks(video, seed) };
  } catch (error) {
    return failure(error);
  }
}

export async function previewFrame(request: DatasetPreviewFrameRequest): Promise<DatasetPreviewFrameOutcome> {
  try {
    const frame = await runningSession().frame(request);
    // Each picture gets its own buffer: IPC copies a view's whole backing
    // buffer, and both pictures share the reply's.
    return {
      success: true,
      value: {
        ...frame,
        data: new Uint8Array(frame.data),
        ...(frame.compareData ? { compareData: new Uint8Array(frame.compareData) } : {}),
      },
    };
  } catch (error) {
    if (error instanceof PreviewSupersededError) return { success: false, error: error.message, superseded: true };
    return failure(error);
  }
}

/** Stops the preview server, if one is running; the promise settles once it has exited. */
export function closeDatasetPreview(): Promise<void> {
  if (!session) return Promise.resolve();
  const exited = session.dispose();
  session = null;
  logger.info('Closed the dataset preview');
  return exited;
}

export function registerDatasetPreviewHandlers(): void {
  handleValidated('dataset-preview-load', datasetPreviewLoadSchema, async ({ job, jobPath }) => loadPreview(job, jobPath));
  handleValidated('dataset-preview-chunks', datasetPreviewChunksSchema, async ({ video, seed }) => previewChunks(video, seed));
  handleValidated('dataset-preview-frame', datasetPreviewFrameSchema, async request => previewFrame(request));
  handleValidated('dataset-preview-close', z.undefined(), async () => {
    await closeDatasetPreview();
    return { success: true };
  });
}
