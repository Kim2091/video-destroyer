// src/components/dataset/DatasetStartScreen.tsx — what a job starts from:
// the example workflows in order as cards with their name and a few words on
// what they make (the full description is the tooltip), an empty job, or a
// saved one. Shown on the canvas while no job is open, and by New.

import { memo } from 'react';
import { ArrowRight, FilePlus, FolderOpen } from 'lucide-react';
import { exampleTitle, type DatasetExample, type DatasetJob } from '../../utils/datasetGraph';

interface DatasetStartScreenProps {
  examples: DatasetExample[];
  onPick: (job: DatasetJob | null) => void;
  onOpen: () => void;
}

const SMALL_BTN = 'h-8 px-3 rounded-md inline-flex items-center gap-2 text-[12.5px] border border-ink-750 bg-ink-850 text-ink-300 hover:text-ink-100 hover:border-ink-700 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500';

export const DatasetStartScreen = memo(({ examples, onPick, onOpen }: DatasetStartScreenProps) => (
  <div className="w-full max-w-[880px] mx-auto">
    <h2 className="font-display text-[17px] font-semibold uppercase tracking-[0.12em] text-ink-100">New dataset job</h2>

    <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-2">
      {examples.map((example, i) => {
        const { number, title } = exampleTitle(example.name, i);
        const description = typeof example.job.description === 'string' ? example.job.description : undefined;
        const summary = typeof example.job.summary === 'string' ? example.job.summary : '';
        const first = i === 0;
        return (
          <button
            key={example.name}
            type="button"
            onClick={() => onPick(example.job)}
            title={description}
            className={`group text-left rounded-md border px-3 py-2.5 flex items-center gap-3 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 ${
              first ? 'border-accent-500/60 bg-accent-500/10 hover:bg-accent-500/15' : 'border-ink-750 bg-ink-900 hover:bg-ink-850 hover:border-ink-700'
            }`}
          >
            <span className={`w-7 h-7 flex-shrink-0 rounded-full grid place-items-center text-[12.5px] font-semibold tabular-nums ${
              first ? 'bg-accent-500 text-accent-ink' : 'bg-ink-800 text-ink-300'
            }`}>
              {number}
            </span>
            <span className="min-w-0 flex-1">
              <span className="flex items-center gap-2">
                <span className="text-[13px] font-semibold text-ink-100 truncate">{title}</span>
                {first && <span className="text-[10px] font-semibold uppercase tracking-[0.1em] text-accent-400 flex-shrink-0">Start here</span>}
                <span className="flex-1" />
                <span className="text-[11px] text-ink-500 flex-shrink-0 tabular-nums">{example.job.nodes.length} nodes</span>
              </span>
              {summary && <span className="mt-0.5 block text-[11.5px] text-ink-400 truncate">{summary}</span>}
            </span>
            <ArrowRight className="w-4 h-4 flex-shrink-0 text-ink-600 group-hover:text-ink-300 transition-colors" />
          </button>
        );
      })}
    </div>
    {examples.length === 0 && <p className="mt-4 text-[12px] text-ink-500">No examples are installed.</p>}

    <div className="mt-3 flex items-center gap-2">
      <button type="button" onClick={() => onPick(null)} className={SMALL_BTN}>
        <FilePlus className="w-4 h-4" />
        Empty job
      </button>
      <button type="button" onClick={onOpen} className={SMALL_BTN}>
        <FolderOpen className="w-4 h-4" />
        Open a saved job
      </button>
    </div>
  </div>
));

DatasetStartScreen.displayName = 'DatasetStartScreen';
