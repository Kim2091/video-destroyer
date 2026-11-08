// electron/installPreflight.ts
//
// What to check before setup or a plugin install starts, so a doomed install
// is refused up front instead of dying halfway through a multi-GB download.
//
// The failures this answers were seen in the field: pip hitting Errno 28 (disk
// full) partway through the CUDA/TensorRT stack, a data\pip-cache that had
// grown to 2.8 GB because nothing ever pruned it, and a portable copy under
// "E:\Program Files\..." that ran into write and path problems. On Windows
// data\ sits beside the exe (APP_DATA_PATH in constants.ts), so wherever the
// user unpacked the portable folder is where all of this lands.
//
// Nothing here changes anything except the probe file and the pip cache: the
// checks report, and the caller decides what to show.

import * as fs from 'fs';
import * as path from 'path';
import { execFile } from 'child_process';
import { logger } from './logger';

export type PreflightProblemKind = 'disk-space' | 'not-writable' | 'path-length' | 'protected-location';

export interface PreflightProblem {
  kind: PreflightProblemKind;
  /** blocking: the install will certainly fail; warning: it might */
  severity: 'blocking' | 'warning';
  /** One actionable sentence for the UI */
  message: string;
}

export interface PreflightOptions {
  dataDir: string;         // PATHS.APP_DATA
  tempDir?: string;        // os.tmpdir(); pip stages its downloads here
  requiredBytes: number;   // on the data drive
  requiredTempBytes?: number;
  /**
   * Whether Windows long paths are switched on. Left out, it is read from the
   * registry; tests pass it so the answer does not depend on the machine.
   */
  longPathsEnabled?: boolean;
}

const GB = 1024 ** 3;

// ---------------------------------------------------------------------------
// How much room an install needs
//
// Measured on 2026-11-07 from a fresh first-run setup on an NVIDIA machine
// (electron/setupInstall.integration.test.ts):
//
//   vapoursynth-portable       9.0 GB, of which
//     torch (cu130)             2.9 GB
//     tensorrt_libs             2.2 GB
//     nvidia (CUDA libs)        1.7 GB  vsjetpack's NVIDIA extra
//     vapoursynth + plugins     1.1 GB
//     everything else           ~1.1 GB  cv2, scipy, deepdeinterlace weights,
//                                        onnxruntime, TensorRT-RTX, ModelOpt...
//   vs-scripts                  2 MB    the Hybrid scripts
//   ffmpeg                      0.65 GB
//   downloaded                  ~6.3 GB in all, held in data\pip-cache until
//                                        the install succeeds and prunes it
//
// TensorRT engines are what the user builds by using the app, not what the
// install writes, so none count here. On the way the install needs more:
// pip holds every downloaded wheel in the temp folder (pip-unpack-*) until
// the whole command ends, and writes a second copy into data\pip-cache.
//
// Non-NVIDIA installs were not measured. They are the NVIDIA figure without
// the CUDA-only parts (torch cu130, tensorrt_libs, nvidia, TensorRT-RTX and
// the engine builder's packages: ~7.3 GB) plus CPU torch and the DirectML,
// NCNN and AMD HIP plugins (~1 GB, estimated): ~3 GB installed and ~1.5 GB
// downloaded.
//
// Only what was measured refuses an install: the installed packages. pip's
// cache copy is an estimate, so it sits in the low-space
// margin below and warns rather than refuses - a guess must not stop an
// install that would have fitted.
// ---------------------------------------------------------------------------

/**
 * Data-drive bytes a plugin install certainly writes: the installed packages,
 * with no built engines. NVIDIA: 9.0, rounded to 10 GB. Others: 3, and
 * 3.45 on Windows AMD with the MIGraphX runtime (0.38 GB unpacked plus its
 * 0.07 GB archive); both are estimates, so they get a margin: 5 GB.
 */
export const PLUGIN_INSTALL_REQUIRED_BYTES: Record<'nvidia' | 'amd' | 'intel' | 'unknown', number> = {
  nvidia: 10 * GB,
  amd: 5 * GB,
  intel: 5 * GB,
  unknown: 5 * GB,
};

/** Temp-folder bytes a plugin install needs: every downloaded wheel, held until pip exits. */
export const PLUGIN_INSTALL_REQUIRED_TEMP_BYTES: Record<'nvidia' | 'amd' | 'intel' | 'unknown', number> = {
  nvidia: 7 * GB,
  amd: 1.5 * GB,
  intel: 1.5 * GB,
  unknown: 1.5 * GB,
};

/**
 * Data-drive bytes core setup writes: embedded Python and the VapourSynth
 * core (~0.25 GB), ffmpeg (0.65 GB measured, plus its .7z downloaded beside
 * it), video-compare (0.2 GB measured, plus its zip). About 1.4 GB, rounded
 * up to 2 GB because the archive sizes were not measured.
 */
export const CORE_SETUP_REQUIRED_BYTES = 2 * GB;

/** Core setup's pip downloads are small; this is room for them and nothing more. */
export const CORE_SETUP_REQUIRED_TEMP_BYTES = 0.5 * GB;

/**
 * An install that fits but leaves less than this is let through with a
 * warning. It covers what is likely but not certain: pip's cache copy of the
 * downloads (2.8 GB measured, up to ~5 GB fresh on NVIDIA; cleared after a
 * successful install), and the engines the first TensorRT runs build (2.9 GB
 * in site-packages on the measured install) - and renders need room too.
 */
export const LOW_SPACE_MARGIN_BYTES = 6 * GB;

// ---------------------------------------------------------------------------
// Path length
//
// Without long-path support a Windows path is limited to MAX_PATH = 260
// characters including the terminating NUL, so 259 usable. The deepest path
// in the measured install, relative to data\, is 212 characters:
//
//   vapoursynth-portable\Lib\site-packages\torch-2.13.0+cu130.dist-info\
//   licenses\third_party\kineto\libkineto\third_party\dynolog\third_party\
//   prometheus-cpp\3rdparty\civetweb\src\third_party\duktape-1.5.2\LICENSE.txt
//
// which leaves 259 - 1 - 212 = 46 characters for data\ itself. The deepest
// directory there (200 characters) against CreateDirectory's 248-character
// limit gives the same 46, so the file figure alone is used.
//
// pluginInstaller.ts already sidesteps this for its own pip runs by installing
// through a short junction in the temp folder - which moves the problem there,
// so the temp folder is checked against the same tree seen from the junction.
// What still goes through the real path is launch-time package top-ups, and
// anyone copying, moving or deleting the folder in Explorer.
//
// With LongPathsEnabled set in the registry, python.exe (3.6+, whose manifest
// declares longPathAware) and so pip are not held to MAX_PATH; that is
// Python's documented behaviour, and the measured install above lives on a
// machine with it set. Node's fs is not held to it either. Whether 7-Zip and
// the NSIS installer honour it was not verified, and nothing here relies on
// them doing so - none of what they unpack comes near the torch tree.
// ---------------------------------------------------------------------------

/** Characters Windows allows in a path when long paths are off (260 less the NUL). */
export const WINDOWS_MAX_PATH_CHARS = 259;

/** Deepest path under data\ in the measured NVIDIA install (see above). */
export const DEEPEST_DATA_RELATIVE_PATH = 212;

/** The same tree as seen from pluginInstaller's junction, which stands in for vapoursynth-portable. */
const DEEPEST_UNDER_VS = DEEPEST_DATA_RELATIVE_PATH - 'vapoursynth-portable\\'.length;

/** pluginInstaller's junction: mkdtemp(tmp\vkp-) adds six characters, then \p. */
const PIP_JUNCTION_SUFFIX = '\\vkp-XXXXXX\\p'.length;

export const MAX_DATA_DIR_LENGTH = WINDOWS_MAX_PATH_CHARS - 1 - DEEPEST_DATA_RELATIVE_PATH;
export const MAX_TEMP_DIR_LENGTH = WINDOWS_MAX_PATH_CHARS - PIP_JUNCTION_SUFFIX - 1 - DEEPEST_UNDER_VS;

// ---------------------------------------------------------------------------

function formatGB(bytes: number): string {
  return `${(bytes / GB).toFixed(1)} GB`;
}

/** The directory itself if it exists, else its nearest ancestor that does. */
async function nearestExistingDir(dir: string): Promise<string | null> {
  let current = path.resolve(dir);
  for (;;) {
    try {
      const stat = await fs.promises.stat(current);
      if (stat.isDirectory()) return current;
    } catch {
      // Missing (or unreadable): keep walking towards the root
    }
    const parent = path.dirname(current);
    if (parent === current) return null;
    current = parent;
  }
}

/** Free bytes on the volume holding `dir` (walks up to the nearest existing ancestor), or null if unknown. */
export async function freeBytes(dir: string): Promise<number | null> {
  const existing = await nearestExistingDir(dir);
  if (!existing) return null;
  try {
    const stats = await fs.promises.statfs(existing);
    // bavail rather than bfree: what this user may write, not what the
    // filesystem holds back for others or for root.
    return Number(stats.bavail) * Number(stats.bsize);
  } catch (error) {
    logger.warn(`Could not read free space for ${existing}:`, error);
    return null;
  }
}

/** Something that names a volume, so two folders can be told apart or together. */
async function volumeKey(dir: string): Promise<string | null> {
  const existing = await nearestExistingDir(dir);
  if (!existing) return null;
  try {
    // On Windows Node reports the volume serial number as dev, so this holds
    // for folders mounted into another drive as well as for drive letters.
    return `dev:${(await fs.promises.stat(existing)).dev}`;
  } catch {
    const root = path.parse(existing).root;
    return `root:${process.platform === 'win32' ? root.toLowerCase() : root}`;
  }
}

function volumeLabel(dir: string): string {
  const root = path.parse(path.resolve(dir)).root;
  // "C:" reads better in a sentence than "C:\"; elsewhere say the folder
  return /^[a-zA-Z]:[\\/]$/.test(root) ? root.slice(0, 2) : `the drive holding ${dir}`;
}

async function checkDiskSpace(options: PreflightOptions): Promise<PreflightProblem[]> {
  const { dataDir, tempDir } = options;
  const requiredTemp = tempDir ? options.requiredTempBytes ?? 0 : 0;

  // pip holds its downloads in temp while it installs into data\, so at the
  // peak both are full at once. On one volume that means the volume needs
  // the sum; checking each figure against the same free space separately
  // would pass an install that cannot fit. On two volumes each is its own
  // check, and a problem on one says nothing about the other.
  const sameVolume = tempDir && requiredTemp > 0
    ? await volumeKey(dataDir).then(async key => key !== null && key === await volumeKey(tempDir))
    : false;

  const checks: Array<{ dir: string; required: number; describe: (need: string, have: string) => string }> = [];
  if (sameVolume) {
    checks.push({
      dir: dataDir,
      required: options.requiredBytes + requiredTemp,
      describe: (need, have) =>
        `This install needs about ${need} free on ${volumeLabel(dataDir)} (Video Destroyer's data folder and the ` +
        `system temp folder are both there), but only ${have} is free; free up space or move the Video Destroyer ` +
        `folder to a larger drive.`,
    });
  } else {
    checks.push({
      dir: dataDir,
      required: options.requiredBytes,
      describe: (need, have) =>
        `This install needs about ${need} free on ${volumeLabel(dataDir)} for Video Destroyer's data folder, but ` +
        `only ${have} is free; free up space or move the Video Destroyer folder to a larger drive.`,
    });
    if (tempDir && requiredTemp > 0) {
      checks.push({
        dir: tempDir,
        required: requiredTemp,
        describe: (need, have) =>
          `The system temp folder on ${volumeLabel(tempDir)} needs about ${need} free while packages ` +
          `download, but only ${have} is free; free up space on that drive.`,
      });
    }
  }

  const problems: PreflightProblem[] = [];
  for (const check of checks) {
    const free = await freeBytes(check.dir);
    if (free === null) continue; // Unknown is not a reason to refuse
    if (free < check.required) {
      problems.push({
        kind: 'disk-space',
        severity: 'blocking',
        message: check.describe(formatGB(check.required), formatGB(free)),
      });
    } else if (free < check.required + LOW_SPACE_MARGIN_BYTES) {
      problems.push({
        kind: 'disk-space',
        severity: 'warning',
        message:
          `This install will leave less than ${formatGB(LOW_SPACE_MARGIN_BYTES)} free on ` +
          `${volumeLabel(check.dir)}, which GPU engine builds and renders also need; consider freeing up space first.`,
      });
    }
  }
  return problems;
}

async function checkWritable(dataDir: string): Promise<PreflightProblem | null> {
  // A data folder that does not exist yet is created inside its parent, so
  // the parent is what has to accept writes.
  const target = await nearestExistingDir(dataDir);
  if (!target) return null;
  const probe = path.join(target, `.vapourkit-write-probe-${process.pid}-${Date.now()}`);
  try {
    const handle = await fs.promises.open(probe, 'wx');
    await handle.close();
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === 'EACCES' || code === 'EPERM' || code === 'EROFS') {
      return {
        kind: 'not-writable',
        severity: 'blocking',
        message:
          `Video Destroyer can't write to ${target}; move the Video Destroyer folder out of Program Files to a folder ` +
          `you own, such as C:\\VideoDestroyer or one in your user folder, and run it from there.`,
      };
    }
    // Anything else (a full disk, say) is left to the other checks rather
    // than guessed at here.
    logger.warn(`Write probe in ${target} failed:`, error);
    return null;
  }
  await fs.promises.unlink(probe).catch(error => {
    logger.warn(`Could not remove write probe ${probe}:`, error);
  });
  return null;
}

/**
 * True when `dir` is inside Program Files, Program Files (x86) or Windows.
 * Compared case-insensitively, as Windows does; the env vars cover installs
 * where those folders live somewhere other than C:.
 */
export function isProtectedLocation(dir: string, env: NodeJS.ProcessEnv = process.env): boolean {
  const roots = [
    env.ProgramFiles ?? 'C:\\Program Files',
    env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)',
    env.ProgramW6432,
    env.SystemRoot ?? env.windir ?? 'C:\\Windows',
  ].filter((root): root is string => Boolean(root));

  const target = path.win32.resolve(dir).toLowerCase();
  return roots.some(root => {
    const base = path.win32.resolve(root).toLowerCase().replace(/\\+$/, '');
    return target === base || target.startsWith(`${base}\\`);
  });
}

/** LongPathsEnabled from the registry; false when it cannot be read. */
function readLongPathsEnabled(): Promise<boolean> {
  return new Promise(resolve => {
    execFile(
      'reg',
      ['query', 'HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem', '/v', 'LongPathsEnabled'],
      { windowsHide: true, timeout: 5000 },
      (error, stdout) => resolve(!error && /LongPathsEnabled\s+REG_DWORD\s+0x0*1\b/i.test(stdout)),
    );
  });
}

function checkPathLength(dataDir: string, tempDir: string | undefined): PreflightProblem[] {
  const problems: PreflightProblem[] = [];
  const dataPath = path.win32.resolve(dataDir);
  if (dataPath.length > MAX_DATA_DIR_LENGTH) {
    problems.push({
      kind: 'path-length',
      severity: 'warning',
      message:
        `Video Destroyer's data folder path is ${dataPath.length} characters, too long for some files it installs ` +
        `(at most ${MAX_DATA_DIR_LENGTH} fits); move the Video Destroyer folder somewhere shorter, such as ` +
        `C:\\VideoDestroyer, or turn on Windows long path support.`,
    });
  }
  if (tempDir) {
    const tempPath = path.win32.resolve(tempDir);
    if (tempPath.length > MAX_TEMP_DIR_LENGTH) {
      problems.push({
        kind: 'path-length',
        severity: 'warning',
        message:
          `The system temp folder path is ${tempPath.length} characters, too long for the packages Video Destroyer ` +
          `installs through it (at most ${MAX_TEMP_DIR_LENGTH} fits); turn on Windows long path support or ` +
          `point TEMP at a shorter folder.`,
      });
    }
  }
  return problems;
}

/**
 * Everything worth knowing before an install starts, blocking problems
 * first. An empty list means go ahead. Checks that cannot get an answer
 * (free space unreadable, say) stay quiet rather than refuse.
 */
export async function runInstallPreflight(options: PreflightOptions): Promise<PreflightProblem[]> {
  const problems: PreflightProblem[] = [];
  const isWindows = process.platform === 'win32';

  const notWritable = await checkWritable(options.dataDir);
  if (notWritable) problems.push(notWritable);

  problems.push(...await checkDiskSpace(options));

  if (isWindows) {
    // The not-writable message already says to move out of Program Files
    if (!notWritable && isProtectedLocation(options.dataDir)) {
      problems.push({
        kind: 'protected-location',
        severity: 'warning',
        message:
          `Video Destroyer is running from a protected Windows folder (${options.dataDir}), where writes can be ` +
          `blocked or redirected; move the Video Destroyer folder to one you own, such as C:\\VideoDestroyer.`,
      });
    }

    const longPaths = options.longPathsEnabled ?? await readLongPathsEnabled();
    if (!longPaths) problems.push(...checkPathLength(options.dataDir, options.tempDir));
  }

  const rank = { blocking: 0, warning: 1 } as const;
  problems.sort((a, b) => rank[a.severity] - rank[b.severity]);
  for (const problem of problems) {
    logger.info(`Install preflight (${problem.severity}, ${problem.kind}): ${problem.message}`);
  }
  return problems;
}

// ---------------------------------------------------------------------------
// pip cache
// ---------------------------------------------------------------------------

/**
 * Bytes under `dir`, counting what can be read. Files vanish and folders
 * refuse listing while pip or an antivirus is at work; those are skipped
 * rather than failing the whole count. Links are not followed.
 */
async function directorySize(dir: string): Promise<number> {
  let total = 0;
  let entries: fs.Dirent[];
  try {
    entries = await fs.promises.readdir(dir, { withFileTypes: true });
  } catch {
    return 0;
  }
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      total += await directorySize(full);
    } else if (entry.isFile()) {
      try {
        total += (await fs.promises.lstat(full)).size;
      } catch {
        // Gone since the listing
      }
    }
  }
  return total;
}

/**
 * Deletes the pip cache when it is larger than maxBytes; returns bytes freed.
 * Never throws.
 *
 * The cache only saves re-downloading, and pip rebuilds it as it goes, so
 * the whole folder goes rather than the oldest entries: pip's own layout
 * gives no cheap way to tell which wheels a future install would reuse.
 * Not to be called while a pip install is running against the same cache.
 */
export async function prunePipCache(cacheDir: string, maxBytes: number): Promise<number> {
  try {
    const before = await directorySize(cacheDir);
    if (before <= maxBytes) return 0;

    logger.info(`pip cache at ${cacheDir} is ${formatGB(before)} (limit ${formatGB(maxBytes)}); removing it`);
    try {
      await fs.promises.rm(cacheDir, { recursive: true, force: true, maxRetries: 3 });
    } catch (error) {
      // A file held open by another process keeps the rest from going; what
      // did go is still counted below.
      logger.warn(`Could not fully remove pip cache ${cacheDir}:`, error);
    }
    const freed = Math.max(0, before - await directorySize(cacheDir));
    logger.info(`pip cache prune freed ${formatGB(freed)}`);
    return freed;
  } catch (error) {
    logger.warn(`pip cache prune failed for ${cacheDir}:`, error);
    return 0;
  }
}
