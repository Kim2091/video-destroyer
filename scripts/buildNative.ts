/**
 * Builds the analogue video VapourSynth plugin (native/vs-ntsc, over the
 * native/ntsc-simulator submodule) and puts it in include/vk_dataset/plugins
 * with its licence files, where the analogue video nodes load it from and the
 * app's packaging picks it up. Needs cargo on the PATH.
 */

import { execFileSync } from 'child_process';
import { copyFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';

const root = join(__dirname, '..');
const crate = join(root, 'native', 'vs-ntsc');
const simulator = join(root, 'native', 'ntsc-simulator');
const targetDir = join(crate, 'build');
const pluginsDir = join(root, 'include', 'vk_dataset', 'plugins');
const library = process.platform === 'win32' ? 'vsntsc.dll' : 'libvsntsc.so';

if (!existsSync(join(simulator, 'Cargo.toml'))) {
  execFileSync('git', ['submodule', 'update', '--init', 'native/ntsc-simulator'], { cwd: root, stdio: 'inherit' });
}
execFileSync('cargo', ['build', '--release', '--locked', '--target-dir', targetDir], { cwd: crate, stdio: 'inherit' });

mkdirSync(pluginsDir, { recursive: true });
copyFileSync(join(targetDir, 'release', library), join(pluginsDir, library));
copyFileSync(join(simulator, 'LICENSE'), join(pluginsDir, 'ntsc-simulator-LICENSE.txt'));
copyFileSync(join(simulator, 'NOTICE'), join(pluginsDir, 'ntsc-simulator-NOTICE.txt'));
copyFileSync(join(simulator, 'THIRD_PARTY_LICENSES.md'), join(pluginsDir, 'ntsc-simulator-THIRD_PARTY_LICENSES.md'));
copyFileSync(join(crate, 'THIRD_PARTY_NOTICES.md'), join(pluginsDir, 'vsntsc-THIRD_PARTY_NOTICES.md'));
console.log(`Built ${library} into ${pluginsDir}`);
