import { memo, useEffect, useState } from 'react';
import { Sparkles, ExternalLink } from 'lucide-react';
import { ModalShell } from './ModalShell';
import { Logo } from './Logo';

/** The projects the app is built on or ports from, in the order shown. */
const CREDITS: { name: string; note: string; url?: string }[] = [
  { name: 'Vapourkit', note: 'the application this one is built on', url: 'https://github.com/Kim2091/vapourkit' },
  { name: 'video-destroyer', note: 'the degradations and the default preset', url: 'https://github.com/Kim2091/video-destroyer' },
  { name: 'VSR-Dataset-Creator', note: 'by Bendel: pair mode, HR processing and renaming' },
  { name: 'VapourSynth', note: 'the frame server every job runs in', url: 'https://www.vapoursynth.com' },
  { name: 'vs-mlrt', note: 'ONNX inference for the model nodes', url: 'https://github.com/AmusementClub/vs-mlrt' },
  { name: 'vs_align', note: 'by pifroggi: temporal and spatial alignment', url: 'https://github.com/pifroggi/vs_align' },
  { name: 'vs-jetpack', note: 'Fast Line Darken, Fine Dehalo and the VapourSynth tooling', url: 'https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack' },
];

interface AboutModalProps {
  show: boolean;
  onClose: () => void;
}

export const AboutModal = memo<AboutModalProps>(({ show, onClose }) => {
  const [version, setVersion] = useState<string>('');

  useEffect(() => {
    const fetchVersion = async () => {
      try {
        const result = await window.electronAPI.getVersion();
        setVersion(result.version);
      } catch (error) {
        console.error('Failed to fetch version:', error);
      }
    };
    
    if (show) {
      fetchVersion();
    }
  }, [show]);

  if (!show) return null;

  const openExternal = (url: string): void => {
    window.electronAPI.openExternal(url);
  };

  return (
    <ModalShell
      title="About"
      icon={Sparkles}
      onClose={onClose}
      closeLabel="Close about"
      className="max-w-2xl w-full max-h-[90vh] overflow-hidden flex flex-col"
    >
      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="text-center px-6 pt-6 pb-5">
          {/* The one surface where the brand mark keeps its full colours (no monochrome prop) */}
          <Logo className="w-12 h-12 mx-auto mb-3" />
          <h3 className="font-display text-[15px] font-semibold uppercase tracking-[0.16em] text-ink-100 mb-1">
            Video Destroyer
          </h3>
          {version && (
            <p className="text-ink-400 text-[11.5px] font-mono tabular-nums mb-0.5">
              v{version}
            </p>
          )}
          <p className="text-ink-400 text-[11.5px]">
            Builds AI training datasets from video. Made by Kim2091
          </p>

          {/* Credits */}
          <div className="mt-5 pt-4 border-t border-ink-900">
            <div className="text-center space-y-1.5">
              <p className="text-[12px] font-medium text-ink-300 italic">In loving memory of my Mom</p>
              <p className="text-[11px] text-ink-500 leading-relaxed max-w-md mx-auto">
                Thank you for your love, support, and encouragement in everything I do
              </p>
              <p className="text-[11px] text-ink-500 pt-1">Rest in peace</p>
            </div>
          </div>
        </div>

        {/* Credits ledger */}
        <section className="border-t border-ink-800">
          <p className="px-4 pt-2.5 pb-1 text-[10px] font-display font-semibold uppercase tracking-[0.09em] text-ink-500">Credits</p>
          {CREDITS.map(credit => (
            <button
              key={credit.name}
              type="button"
              disabled={!credit.url}
              onClick={() => credit.url && openExternal(credit.url)}
              className="w-full min-h-9 py-1.5 flex items-center gap-2.5 px-4 border-b border-ink-900 text-left text-[12.5px] text-ink-300 enabled:hover:bg-ink-850 enabled:hover:text-ink-200 transition-colors group disabled:cursor-default"
            >
              <span className="flex-1 min-w-0">
                <span className="text-ink-200">{credit.name}</span>
                <span className="text-ink-500"> — {credit.note}</span>
              </span>
              {credit.url && <ExternalLink className="w-3.5 h-3.5 text-ink-600 group-hover:text-ink-400 transition-colors flex-shrink-0" />}
            </button>
          ))}
          <p className="px-4 py-2.5 text-[11px] text-ink-500">
            Licensed under the GNU GPL v3. The third-party notices for the dataset engine are in <span className="font-mono text-ink-400">include/vk_dataset/NOTICE.md</span>.
          </p>
        </section>
      </div>

      {/* Footer */}
      <div className="h-9 flex-shrink-0 flex items-center justify-end gap-2 px-4 border-t border-ink-800 bg-ink-900 text-[11px] text-ink-500">
        <kbd className="px-1.5 py-0.5 bg-ink-850 border border-ink-750 rounded text-[10px] font-mono text-ink-300">Esc</kbd>
        <span>to close</span>
      </div>
    </ModalShell>
  );
});
