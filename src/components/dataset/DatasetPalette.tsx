// src/components/dataset/DatasetPalette.tsx — the Dataset workspace's left
// column: every node type by name, grouped under the stage of a job it
// belongs to, in stage order, then the groups of nodes that have no stage
// (the value nodes), with a search over labels and descriptions.
// A group folds away from its header; a node's description is its tooltip.
// A node is added by dragging it onto the canvas or by clicking it.

import { memo, useMemo, useState } from 'react';
import { ChevronRight, Search, X } from 'lucide-react';
import type { DatasetNodeSpec, DatasetStage } from '../../utils/datasetGraph';
import { paletteSections } from '../../utils/datasetStages';
import { nodeColor, StageTag } from './DatasetNodeCard';

export const DRAG_TYPE = 'application/x-vk-dataset-node';

interface DatasetPaletteProps {
  specs: DatasetNodeSpec[];
  stages: DatasetStage[];
  onAdd: (spec: DatasetNodeSpec) => void;
}

export const DatasetPalette = memo(({ specs, stages, onAdd }: DatasetPaletteProps) => {
  const [query, setQuery] = useState('');
  const [folded, setFolded] = useState<Set<string>>(() => new Set());
  const sections = useMemo(() => paletteSections(specs, stages, query), [specs, stages, query]);
  const toggle = (name: string) => setFolded(current => {
    const next = new Set(current);
    if (!next.delete(name)) next.add(name);
    return next;
  });

  return (
    <div className="h-full flex flex-col min-h-0">
      <div className="flex-shrink-0 p-2">
        <label className="relative block">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-ink-500 pointer-events-none" />
          <input
            type="search"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => { if (e.key === 'Escape') setQuery(''); }}
            placeholder="Search"
            aria-label="Search nodes"
            className="w-full h-7 pl-7 pr-7 bg-ink-850 border border-ink-750 rounded text-[12px] text-ink-200 placeholder-ink-500 focus:outline-none focus:border-accent-500 [&::-webkit-search-cancel-button]:hidden"
          />
          {query && (
            <button
              type="button"
              onClick={() => setQuery('')}
              aria-label="Clear the search"
              className="absolute right-1 top-1/2 -translate-y-1/2 w-5 h-5 rounded grid place-items-center text-ink-500 hover:text-ink-200"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </label>
      </div>

      <div className="flex-1 overflow-y-auto min-h-0 pb-2">
        {sections.length === 0 && (
          <div className="px-3 py-2 text-[12px] text-ink-500">No match</div>
        )}
        {sections.map(({ stage, specs: members }) => {
          // A search shows every match, so folding applies only without one.
          const open = query.trim() !== '' || !folded.has(stage.name);
          return (
            <section key={`${stage.stage}:${stage.name}`} className="mt-1">
              <button
                type="button"
                onClick={() => toggle(stage.name)}
                aria-expanded={open}
                title={stage.help || undefined}
                className="w-full h-7 px-2 flex items-center gap-1 text-left hover:bg-ink-900 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent-500"
              >
                <ChevronRight className={`w-3 h-3 text-ink-600 flex-shrink-0 transition-transform ${open ? 'rotate-90' : ''}`} />
                <StageTag stage={stage.stage || null} name={stage.name} color={stage.stage ? undefined : nodeColor(members[0])} />
              </button>
              {open && members.map(spec => (
                <button
                  key={spec.type}
                  type="button"
                  draggable
                  onDragStart={event => {
                    event.dataTransfer.setData(DRAG_TYPE, spec.type);
                    event.dataTransfer.effectAllowed = 'copy';
                  }}
                  onClick={() => onAdd(spec)}
                  title={spec.description}
                  className="w-full h-7 pl-7 pr-2 text-left text-[12.5px] text-ink-300 truncate hover:bg-ink-850 hover:text-ink-100 transition-colors cursor-grab active:cursor-grabbing focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent-500"
                >
                  {spec.label}
                </button>
              ))}
            </section>
          );
        })}
      </div>
    </div>
  );
});

DatasetPalette.displayName = 'DatasetPalette';
