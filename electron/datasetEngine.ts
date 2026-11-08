// electron/datasetEngine.ts — keeps data/config/vk_dataset the same as the
// bundled include/vk_dataset, the dataset engine the Dataset workspace runs.

import * as path from 'path';
import * as fs from 'fs-extra';
import { PATHS } from './constants';
import { logger } from './logger';

export const DATASET_PACKAGE = 'vk_dataset';

/** The installed engine package. Its parent folder is what goes on sys.path. */
export function datasetEnginePath(): string {
  return path.join(PATHS.CONFIG, DATASET_PACKAGE);
}

/** Job files offered under "New from preset". */
export function datasetPresetsPath(): string {
  return path.join(datasetEnginePath(), 'presets');
}

/**
 * Replaces the installed package with the bundled one as a whole, so a node
 * module dropped from the app is dropped from the install too: the package
 * imports every module in nodes/, and a stale one would still register its
 * nodes. The copy is made beside the target and swapped in.
 */
export async function syncDatasetEngine(bundledBasePath: string): Promise<void> {
  const bundled = path.join(bundledBasePath, 'include', DATASET_PACKAGE);
  if (!await fs.pathExists(bundled)) return;
  const target = datasetEnginePath();
  const staged = `${target}.${process.pid}.tmp`;
  await fs.remove(staged);
  await fs.copy(bundled, staged, {
    filter: source => path.basename(source) !== '__pycache__' && !source.endsWith('.pyc'),
  });
  await fs.remove(target);
  await fs.move(staged, target);
  logger.dependency(`Updated ${DATASET_PACKAGE} from bundled source`);
}
