// electron/installLedger.ts
//
// What this install got from us.
//
// The ledger (installed-components.json) records the version of each set of
// VapourSynth scripts we put into data\ - a digest per file, and the app
// version that put it there - so a later update can tell what it may replace
// from what the user has changed. Python packages need no entry: pip's
// dist-info already is that ledger.

import * as path from 'path';
import * as fs from 'fs-extra';
import { PATHS } from './constants';
import { writeJsonAtomic } from './jsonFile';
import { logger } from './logger';
import type { ScriptLedgerEntry } from './scriptSync';

export interface InstallLedger {
  schema: 1;
  /** vs-scripts, by source (scriptSources.ts) */
  scripts: Record<string, ScriptLedgerEntry>;
  /** The last launch-time package install that failed, so launch backs off */
  packageFailure?: { at: string; specs: string[] };
}

const ledgerPath = () => path.join(PATHS.CONFIG, 'installed-components.json');

export async function readLedger(): Promise<InstallLedger> {
  try {
    const ledger = await fs.readJson(ledgerPath()) as Partial<InstallLedger>;
    return {
      schema: 1,
      scripts: ledger.scripts ?? {},
      ...(ledger.packageFailure ? { packageFailure: ledger.packageFailure } : {}),
    };
  } catch (error) {
    // Absent is the normal state of a fresh install. An unreadable one is
    // treated the same way: every script falls back to being compared by
    // content.
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
      logger.warn('Install ledger unreadable, rebuilding it:', error);
    }
    return { schema: 1, scripts: {} };
  }
}

export async function writeLedger(ledger: InstallLedger): Promise<void> {
  await writeJsonAtomic(ledgerPath(), ledger);
}
