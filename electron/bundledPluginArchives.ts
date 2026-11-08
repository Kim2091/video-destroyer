/**
 * The bundled `plugins.7z` holds Windows VapourSynth plugin DLLs that have no
 * wheel. Linux obtains native plugins from platform-specific PyPI wheels.
 */
export function shouldExtractBundledPluginArchives(
  platform: NodeJS.Platform = process.platform
): boolean {
  return platform === 'win32';
}
