// electron/datasetHandlers.ts — the Dataset workspace's main-process side.
//
// Runs the vk_dataset engine (data/config/vk_dataset, see datasetEngine.ts)
// in the app's Python: `nodes` and `stages` for the node palette, `check` to validate a
// job, and `run` to build a dataset, reading the JSON-lines protocol in
// datasetProtocol.ts. Saving and opening .vkdataset files and the presets
// live here too.
//
// One run at a time, held in its own JobSlot.

import { BrowserWindow, dialog, type WebContents } from 'electron';
import { spawn, type ChildProcess } from 'child_process';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs-extra';
import { z } from 'zod';
import { PATHS } from './constants';
import { logger } from './logger';
import { setupVSEnvironment } from './utils';
import { createWorkloadSpawnOptions, terminateProcessTree } from './processLifecycle';
import { handleValidated } from './ipcValidation';
import { JobSlot } from './jobSlot';
import { datasetEnginePath, datasetPresetsPath } from './datasetEngine';
import { engineRuntimeEnvValue, RUNTIME_SETTINGS_ENV } from './inferenceBackend';
import { createBuildLogReporter } from './engineBuildProtocol';
import {
  DatasetOutputReader,
  datasetEngineArgs,
  datasetJobSchema,
  datasetRunRequestSchema,
  datasetSaveRequestSchema,
  presetName,
  type DatasetEvent,
  type DatasetJobFile,
} from './datasetProtocol';

export type DatasetOutcome<T> =
  | { success: true; value: T }
  | { success: false; error: string; cancelled?: boolean };

const JOB_FILTERS = [
  { name: 'Video Destroyer Dataset Job', extensions: ['vkdataset'] },
  { name: 'All Files', extensions: ['*'] },
];

interface EngineRun {
  child: ChildProcess;
  reader: DatasetOutputReader;
  done: Promise<{ code: number | null; stderr: string; spawnError?: Error }>;
}

/**
 * The engine's environment: the app's VapourSynth, ffmpeg on PATH for the
 * nodes that encode, and the inference settings its model nodes run with
 * (inferenceBackend.ts).
 */
export function datasetEngineEnv(): NodeJS.ProcessEnv {
  const env = setupVSEnvironment(PATHS.PYTHON);
  env.PATH = `${path.dirname(PATHS.FFMPEG)}${path.delimiter}${env.PATH ?? ''}`;
  env[RUNTIME_SETTINGS_ENV] = engineRuntimeEnvValue();
  return env;
}

function startEngine(
  command: 'nodes' | 'stages' | 'check' | 'run',
  jobFile: string | undefined,
  cwd: string,
  onEvent: (event: DatasetEvent) => void = () => {},
): EngineRun {
  const reader = new DatasetOutputReader(onEvent, line => logger.debug(`[dataset] ${line}`));
  const child = spawn(
    PATHS.PYTHON,
    datasetEngineArgs(PATHS.CONFIG, command, jobFile),
    createWorkloadSpawnOptions({
      cwd,
      env: datasetEngineEnv(),
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    }),
  );
  // TensorRT engine builds announce themselves on stderr ([vk-build]); the
  // run log shows them as messages.
  const builds = createBuildLogReporter(message => onEvent({ kind: 'message', message }));
  const done = new Promise<{ code: number | null; stderr: string; spawnError?: Error }>(resolve => {
    let stderr = '';
    child.stdout?.setEncoding('utf8');
    child.stdout?.on('data', (chunk: string) => reader.push(chunk));
    child.stderr?.on('data', (chunk: Buffer) => {
      const text = chunk.toString();
      builds.push(text);
      stderr += text;
      if (stderr.length > 20000) stderr = stderr.slice(-20000);
    });
    child.on('error', spawnError => resolve({ code: null, stderr, spawnError }));
    child.on('close', code => {
      builds.reset();
      reader.end();
      resolve({ code, stderr });
    });
  });
  return { child, reader, done };
}

/** The engine's own error, else the last line it wrote to stderr, else its exit code. */
function failureOf(run: EngineRun, outcome: { code: number | null; stderr: string; spawnError?: Error }): string {
  if (outcome.spawnError) return `The dataset engine could not start: ${outcome.spawnError.message}`;
  if (outcome.stderr.trim()) logger.debug(`[dataset] stderr:\n${outcome.stderr.trim()}`);
  return run.reader.error
    ?? (outcome.stderr.trim().split(/\r?\n/).filter(Boolean).pop() || `The dataset engine exited with code ${outcome.code}.`);
}

export function engineMissing(): string | null {
  return fs.existsSync(path.join(datasetEnginePath(), '__main__.py'))
    ? null
    : `The dataset engine is missing from ${PATHS.CONFIG}. Restart Video Destroyer to restore it.`;
}

/**
 * Writes a job where the engine can read it; the caller removes the file.
 * Synchronous, so a second run request cannot slip in between the busy check
 * and the run taking the slot.
 */
function writeJobFile(job: DatasetJobFile): string {
  const file = path.join(os.tmpdir(), `vk-dataset-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.json`);
  fs.writeFileSync(file, JSON.stringify(job), 'utf-8');
  return file;
}

/** Relative paths in a job are read against the folder of the file it was saved to. */
export function jobFolder(jobPath: string | null): string {
  return jobPath ? path.dirname(jobPath) : os.homedir();
}

/** Runs a command that prints one list: the node types or the stages. */
async function loadList(command: 'nodes' | 'stages'): Promise<DatasetOutcome<unknown[]>> {
  const missing = engineMissing();
  if (missing) return { success: false, error: missing };
  const run = startEngine(command, undefined, PATHS.CONFIG);
  const outcome = await run.done;
  if (Array.isArray(run.reader.result)) {
    logger.info(`Dataset engine declares ${run.reader.result.length} ${command === 'nodes' ? 'node types' : 'stages'}`);
    return { success: true, value: run.reader.result };
  }
  const error = failureOf(run, outcome);
  logger.warn(`Loading the dataset ${command} failed: ${error}`);
  return { success: false, error };
}

// The declarations do not change while the app runs: the engine is only
// replaced at launch.
const lists = new Map<'nodes' | 'stages', Promise<DatasetOutcome<unknown[]>>>();

function cachedList(command: 'nodes' | 'stages'): Promise<DatasetOutcome<unknown[]>> {
  let list = lists.get(command);
  if (!list) {
    list = loadList(command);
    lists.set(command, list);
    // A failure is not kept, so fixing the install and opening the
    // workspace again tries afresh.
    void list.then(outcome => { if (!outcome.success) lists.delete(command); });
  }
  return list;
}

export const getNodeSpecs = () => cachedList('nodes');

/** The stages of a job, in order, with a sentence each. */
export const getStages = () => cachedList('stages');

export async function checkJob(job: DatasetJobFile, jobPath: string | null): Promise<DatasetOutcome<null>> {
  const missing = engineMissing();
  if (missing) return { success: false, error: missing };
  const jobFile = writeJobFile(job);
  try {
    const run = startEngine('check', jobFile, jobFolder(jobPath));
    const outcome = await run.done;
    if (run.reader.result !== undefined && run.reader.error === null) return { success: true, value: null };
    return { success: false, error: failureOf(run, outcome) };
  } finally {
    await fs.remove(jobFile).catch(() => undefined);
  }
}

const runningJob = new JobSlot<ChildProcess>(child => terminateProcessTree(child));

export async function runJob(
  job: DatasetJobFile,
  jobPath: string | null,
  sender: WebContents | null,
): Promise<DatasetOutcome<unknown>> {
  const missing = engineMissing();
  if (missing) return { success: false, error: missing };
  if (runningJob.current) return { success: false, error: 'A dataset job is already running.' };

  const jobFile = writeJobFile(job);
  const report = (event: DatasetEvent) => {
    if (sender && !sender.isDestroyed()) sender.send('dataset-progress', event);
  };
  try {
    const run = startEngine('run', jobFile, jobFolder(jobPath), report);
    runningJob.start(run.child);
    logger.info(`Running dataset job${jobPath ? ` ${jobPath}` : ''} (${job.nodes.length} nodes)`);
    const outcome = await run.done;
    const stopped = !runningJob.holds(run.child);
    runningJob.release(run.child);
    if (stopped && !run.reader.finished) {
      logger.info('Dataset job stopped');
      return { success: false, error: 'The dataset job was stopped.', cancelled: true };
    }
    if (run.reader.error === null && run.reader.result !== undefined) {
      logger.info(`Dataset job finished: ${JSON.stringify(run.reader.result)}`);
      return { success: true, value: run.reader.result };
    }
    const error = failureOf(run, outcome);
    logger.warn(`Dataset job failed: ${error}`);
    return { success: false, error };
  } finally {
    await fs.remove(jobFile).catch(() => undefined);
  }
}

/** Stops the running dataset job, if any. */
export function stopDatasetJob(): boolean {
  return runningJob.stop();
}

/** Every preset job in the engine's presets folder; a file that does not read as a job is skipped. */
export async function listPresets(): Promise<{ name: string; job: DatasetJobFile }[]> {
  const folder = datasetPresetsPath();
  if (!await fs.pathExists(folder)) return [];
  const presets: { name: string; job: DatasetJobFile }[] = [];
  for (const file of (await fs.readdir(folder)).filter(f => f.toLowerCase().endsWith('.json')).sort()) {
    try {
      const parsed = datasetJobSchema.safeParse(JSON.parse(await fs.readFile(path.join(folder, file), 'utf-8')));
      if (parsed.success) {
        // A preset names itself; the file name is the fallback.
        const own = typeof parsed.data.name === 'string' ? parsed.data.name.trim() : '';
        presets.push({ name: own || presetName(file), job: parsed.data });
      }
      else logger.warn(`Dataset preset ${file} is not a job: ${parsed.error.issues[0]?.message}`);
    } catch (error) {
      logger.warn(`Could not read dataset preset ${file}:`, error);
    }
  }
  return presets;
}

export async function readJobFile(filePath: string): Promise<DatasetOutcome<DatasetJobFile>> {
  try {
    const parsed = datasetJobSchema.safeParse(JSON.parse(await fs.readFile(filePath, 'utf-8')));
    if (!parsed.success) {
      const issue = parsed.error.issues[0];
      return { success: false, error: `This is not a dataset job (${issue?.path.join('.') || 'file'}: ${issue?.message}).` };
    }
    return { success: true, value: parsed.data };
  } catch (error) {
    return { success: false, error: error instanceof Error ? error.message : String(error) };
  }
}

export async function writeJob(filePath: string, job: DatasetJobFile): Promise<void> {
  await fs.writeFile(filePath, `${JSON.stringify(job, null, 2)}\n`, 'utf-8');
}

export function registerDatasetHandlers(mainWindow: BrowserWindow | null): void {
  const sender = () => (mainWindow && !mainWindow.isDestroyed() ? mainWindow.webContents : null);

  handleValidated('dataset-get-nodes', z.undefined(), async () => getNodeSpecs());

  handleValidated('dataset-get-stages', z.undefined(), async () => getStages());

  handleValidated('dataset-check', datasetRunRequestSchema, async ({ job, jobPath }) => checkJob(job, jobPath));

  handleValidated('dataset-run', datasetRunRequestSchema, async ({ job, jobPath }) => runJob(job, jobPath, sender()));

  handleValidated('dataset-cancel', z.undefined(), async () => ({ success: stopDatasetJob() }));

  handleValidated('dataset-get-presets', z.undefined(), async () => listPresets());

  handleValidated('dataset-open-job', z.undefined(), async () => {
    const result = await dialog.showOpenDialog({ properties: ['openFile'], filters: JOB_FILTERS });
    if (result.canceled || result.filePaths.length === 0) return null;
    const filePath = result.filePaths[0];
    const read = await readJobFile(filePath);
    return read.success ? { success: true as const, filePath, job: read.value } : read;
  });

  handleValidated('dataset-save-job', datasetSaveRequestSchema, async ({ job, filePath }) => {
    let target = filePath;
    if (!target) {
      const result = await dialog.showSaveDialog({ filters: JOB_FILTERS, defaultPath: 'Dataset.vkdataset' });
      if (result.canceled || !result.filePath) return null;
      target = result.filePath;
    }
    try {
      await writeJob(target, job);
      logger.info(`Saved dataset job to ${target}`);
      return { success: true as const, filePath: target };
    } catch (error) {
      return { success: false as const, error: error instanceof Error ? error.message : String(error) };
    }
  });

  handleValidated('dataset-select-path', z.enum(['file', 'folder']), async mode => {
    const result = await dialog.showOpenDialog({
      properties: [mode === 'folder' ? 'openDirectory' : 'openFile', 'createDirectory'],
    });
    return result.canceled || result.filePaths.length === 0 ? null : result.filePaths[0];
  });
}
