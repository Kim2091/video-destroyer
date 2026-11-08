import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';
import { app } from 'electron';
import { logger } from './logger';

interface GitHubRelease {
  tag_name: string;
  name: string;
  body: string;
  html_url: string;
  published_at: string;
  prerelease: boolean;
  draft: boolean;
}

export interface UpdateInfo {
  /** Whether a release was looked up at all; false when no repository is configured */
  checked: boolean;
  available: boolean;
  currentVersion: string;
  latestVersion: string;
  releaseUrl: string;
  changelog: string;
  publishedAt: string;
}

/**
 * The GitHub repository ("owner/name") whose releases the update check reads,
 * from the `updateRepository` field of package.json. Unset or malformed, the
 * app does not check for updates.
 */
export function parseUpdateRepository(value: unknown): string | null {
  return typeof value === 'string' && /^[\w.-]+\/[\w.-]+$/.test(value.trim()) ? value.trim() : null;
}

function updateRepository(): string | null {
  try {
    const manifest = JSON.parse(fs.readFileSync(path.join(app.getAppPath(), 'package.json'), 'utf-8'));
    return parseUpdateRepository(manifest.updateRepository);
  } catch {
    return null;
  }
}

function noUpdate(checked: boolean): UpdateInfo {
  const currentVersion = app.getVersion();
  return {
    checked,
    available: false,
    currentVersion,
    latestVersion: currentVersion,
    releaseUrl: '',
    changelog: '',
    publishedAt: ''
  };
}

export interface ParsedVersion {
  core: number[];
  prerelease: string | null;
}

/**
 * Parses "v0.16.1-nightly.2026-05-13" into { core: [0, 16, 1], prerelease: 'nightly.2026-05-13' }
 */
export function parseVersion(version: string): ParsedVersion {
  const clean = version.replace(/^v/, '');
  const dashIndex = clean.indexOf('-');
  const corePart = dashIndex === -1 ? clean : clean.slice(0, dashIndex);
  const prerelease = dashIndex === -1 ? null : clean.slice(dashIndex + 1);

  const core = corePart.split('.').map(part => {
    const num = parseInt(part, 10);
    return Number.isNaN(num) ? 0 : num;
  });

  return { core, prerelease };
}

/**
 * Returns true when the latest (stable) release is an actual upgrade from the
 * current version.
 *
 * Only the numeric version core is compared. Nightly builds carry a prerelease
 * suffix (e.g. 0.16.1-nightly.2026-05-13) that the old comparison mangled to
 * NaN→0, making the same-version stable release look newer — nightly users were
 * endlessly prompted to "update" to the stable build they were cut from. A
 * nightly is only offered an update once a stable release with a strictly newer
 * base version exists.
 */
export function isUpdateAvailable(currentVersion: string, latestVersion: string): boolean {
  const current = parseVersion(currentVersion);
  const latest = parseVersion(latestVersion);

  for (let i = 0; i < Math.max(current.core.length, latest.core.length); i++) {
    const currentNum = current.core[i] ?? 0;
    const latestNum = latest.core[i] ?? 0;

    if (latestNum > currentNum) return true;
    if (latestNum < currentNum) return false;
  }

  return false;
}

/**
 * Checks for updates by comparing current version with latest GitHub release
 */
export async function checkForUpdates(): Promise<UpdateInfo> {
  const repository = updateRepository();
  if (!repository) {
    logger.info('No update repository configured; not checking for updates');
    return noUpdate(false);
  }

  try {
    logger.info(`Checking ${repository} for updates...`);

    const currentVersion = app.getVersion();
    logger.info(`Current version: ${currentVersion}`);

    const response = await axios.get<GitHubRelease>(`https://api.github.com/repos/${repository}/releases/latest`, {
      headers: {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'video-destroyer-app'
      },
      timeout: 10000 // 10 second timeout
    });

    const release = response.data;

    // Skip draft and prerelease versions
    if (release.draft || release.prerelease) {
      logger.info('Latest release is draft or prerelease, skipping');
      return noUpdate(true);
    }

    const latestVersion = release.tag_name;
    logger.info(`Latest version: ${latestVersion}`);

    if (!isUpdateAvailable(currentVersion, latestVersion)) {
      logger.info('No updates available');
      return noUpdate(true);
    }

    logger.info('Update available!');
    return {
      checked: true,
      available: true,
      currentVersion,
      latestVersion,
      releaseUrl: release.html_url,
      changelog: release.body || 'No changelog available.',
      publishedAt: release.published_at
    };
  } catch (error) {
    logger.error('Error checking for updates:', error);
    return noUpdate(false);
  }
}

/**
 * The releases page of the configured repository, or null when there is none.
 */
export function getReleasesPageUrl(): string | null {
  const repository = updateRepository();
  return repository ? `https://github.com/${repository}/releases` : null;
}
