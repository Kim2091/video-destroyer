// src/hooks/useDatasetRun.ts — checking and running a dataset job, and the
// log the Dataset workspace shows of it.

import { useCallback, useEffect, useRef, useState } from 'react';
import type { DatasetJob } from '../utils/datasetGraph';

export type DatasetRunStatus = 'idle' | 'checking' | 'running' | 'stopping';

export interface DatasetLogLine {
  id: number;
  text: string;
  tone: 'info' | 'ok' | 'bad';
}

/** The log keeps this many lines; a long run reports every chunk. */
const LOG_LIMIT = 1000;

const summarize = (result: Record<string, unknown>) =>
  Object.entries(result).map(([key, value]) => `${key.replace(/_/g, ' ')} ${value}`).join(', ');

export function useDatasetRun() {
  const [status, setStatus] = useState<DatasetRunStatus>('idle');
  const [progress, setProgress] = useState<number | null>(null);
  const [message, setMessage] = useState('');
  const [log, setLog] = useState<DatasetLogLine[]>([]);
  const nextId = useRef(0);

  const append = useCallback((text: string, tone: DatasetLogLine['tone'] = 'info') => {
    setLog(lines => [...lines, { id: nextId.current++, text, tone }].slice(-LOG_LIMIT));
  }, []);

  useEffect(() => window.electronAPI.onDatasetProgress(event => {
    if (event.kind === 'progress') {
      setProgress(event.progress);
      if (event.message) setMessage(event.message);
    }
    if (event.message) append(event.message);
  }), [append]);

  /** The engine's own check of the job; true when it would run. */
  const check = useCallback(async (job: DatasetJob, jobPath: string | null): Promise<boolean> => {
    setStatus('checking');
    setMessage('Checking the job');
    try {
      const checked = await window.electronAPI.datasetCheck({ job, jobPath });
      if (checked.success) {
        setMessage('The job is ready to run');
        append('Check passed.', 'ok');
        return true;
      }
      setMessage(checked.error);
      append(`Check failed: ${checked.error}`, 'bad');
      return false;
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error);
      setMessage(text);
      append(`Check failed: ${text}`, 'bad');
      return false;
    } finally {
      setStatus('idle');
    }
  }, [append]);

  const run = useCallback(async (job: DatasetJob, jobPath: string | null) => {
    if (!await check(job, jobPath)) return;
    setStatus('running');
    setProgress(0);
    setMessage('Starting');
    append('Run started.');
    try {
      const outcome = await window.electronAPI.datasetRun({ job, jobPath });
      if (outcome.success) {
        setProgress(1);
        setMessage('Finished');
        append(`Finished: ${summarize(outcome.value)}.`, 'ok');
      } else {
        setProgress(null);
        setMessage(outcome.cancelled ? 'Stopped' : outcome.error);
        append(outcome.cancelled ? 'Stopped.' : `Failed: ${outcome.error}`, outcome.cancelled ? 'info' : 'bad');
      }
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error);
      setProgress(null);
      setMessage(text);
      append(`Failed: ${text}`, 'bad');
    } finally {
      setStatus('idle');
    }
  }, [append, check]);

  const cancel = useCallback(async () => {
    setStatus('stopping');
    await window.electronAPI.datasetCancel();
  }, []);

  const clearLog = useCallback(() => setLog([]), []);

  return { status, progress, message, log, append, check, run, cancel, clearLog };
}
