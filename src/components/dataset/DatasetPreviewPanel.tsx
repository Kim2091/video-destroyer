// src/components/dataset/DatasetPreviewPanel.tsx — the Dataset workspace's
// preview: any node's output for one chunk of one video, as a run would
// compute it, with what each node drew for that chunk.
//
// The shown output follows the node selected on the canvas unless pinned.
// A second output can be compared side by side or with a wipe; sequences
// show their crops on the frame, kept ones solid, rejected ones dashed. A
// reroll previews other draws without touching the job's seed, until it is
// adopted.

import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, Check, ChevronLeft, ChevronRight, Dices, Loader2, Pin, PinOff, RefreshCw, RotateCcw, X } from 'lucide-react';
import { PrivacyVeil } from '../PrivacyVeil';
import { DatasetPreviewViewer } from './DatasetPreviewViewer';
import { useDatasetPreview } from '../../hooks/useDatasetPreview';
import { jobVariants, nodeLabel, type DatasetJob, type DatasetNodeSpec, type GraphNode } from '../../utils/datasetGraph';
import {
  cropsAt,
  existingPort,
  formatDrawn,
  portForNode,
  previewPorts,
  randomSeed,
  rangeLabel,
  samePort,
  type CompareMode,
  type CropOverlay,
  type PortRef,
  type ZoomMode,
} from '../../utils/datasetPreview';

interface DatasetPreviewPanelProps {
  job: DatasetJob;
  jobPath: string | null;
  nodes: GraphNode[];
  specs: Map<string, DatasetNodeSpec>;
  selectedNodeId: string | null;
  privacyMode: boolean;
  onAdoptSeed: (seed: number) => void;
  /** What every value node gave for the shown chunk, by node id, for their cards; null while nothing is shown. */
  onValues: (values: Map<string, { value?: unknown; error?: string }> | null) => void;
  onClose: () => void;
}

const BTN = 'h-[24px] px-1.5 rounded inline-flex items-center gap-1 text-[11.5px] border border-ink-750 bg-ink-850 text-ink-400 hover:text-ink-200 hover:border-ink-700 transition-colors flex-shrink-0 disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500';
const PRESSED = 'bg-accent-500/15 border-accent-500/40 text-accent-400 hover:text-accent-300';
const SELECT = 'h-[24px] bg-ink-850 border border-ink-750 rounded px-1.5 text-[11.5px] text-ink-200 focus:outline-none focus:border-accent-500 min-w-0 disabled:opacity-40';
const LABEL = 'font-display text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-500 flex-shrink-0';
const HEADING = 'px-3 h-7 flex items-center gap-2 font-display text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-500 border-b border-ink-800 bg-ink-900 sticky top-0';

const MIN_HEIGHT = 220;
const ZOOMS: { value: ZoomMode; label: string }[] = [{ value: 'fit', label: 'Fit' }, { value: 1, label: '100%' }, { value: 2, label: '200%' }];

const portKey = (ref: PortRef | null) => (ref ? `${ref.node}.${ref.port}` : '');
const parsePortKey = (key: string): PortRef | null => {
  const at = key.lastIndexOf('.');
  return at > 0 ? { node: key.slice(0, at), port: key.slice(at + 1) } : null;
};

const Segmented = <T extends string | number>({ options, value, onChange, title }: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
  title: string;
}) => (
  <div role="group" aria-label={title} className="flex flex-shrink-0">
    {options.map((option, i) => (
      <button
        key={String(option.value)}
        type="button"
        aria-pressed={value === option.value}
        onClick={() => onChange(option.value)}
        className={`${BTN} ${i > 0 ? '-ml-px rounded-l-none' : ''} ${i < options.length - 1 ? 'rounded-r-none' : ''} ${value === option.value ? PRESSED : ''}`}
      >
        {option.label}
      </button>
    ))}
  </div>
);

export const DatasetPreviewPanel = memo<DatasetPreviewPanelProps>(({
  job, jobPath, nodes, specs, selectedNodeId, privacyMode, onAdoptSeed, onValues, onClose,
}: DatasetPreviewPanelProps) => {
  const { selection, update, load, chunks, frame, retry } = useDatasetPreview({ enabled: true, job, jobPath });
  const [pinned, setPinned] = useState(false);
  const [mode, setMode] = useState<CompareMode>('side');
  const [smooth, setSmooth] = useState(false);
  const [zoom, setZoom] = useState<ZoomMode>('fit');
  const [height, setHeight] = useState(380);
  const [highlighted, setHighlighted] = useState<CropOverlay | null>(null);
  const resize = useRef<{ y: number; height: number } | null>(null);

  const ports = useMemo(() => previewPorts(nodes, specs), [nodes, specs]);
  const variants = useMemo(() => jobVariants(job), [job]);
  useEffect(() => {
    if (selection.variant >= variants) update({ variant: 0 });
  }, [selection.variant, variants, update]);
  const labels = useMemo(() => new Map(nodes.map(n => [n.id, nodeLabel(n.data, specs.get(n.data.type))])), [nodes, specs]);
  const portLabel = (ref: PortRef | null) => ports.find(p => samePort(p, ref))?.label ?? (ref ? `${ref.node}.${ref.port}` : '');

  // The shown output follows the canvas selection, and outputs that left the graph are dropped.
  useEffect(() => {
    const target = !pinned && selectedNodeId
      ? portForNode(selectedNodeId, selection.target, ports) ?? existingPort(selection.target, ports)
      : existingPort(selection.target, ports);
    const compare = existingPort(selection.compare, ports);
    if (!samePort(target, selection.target) || !samePort(compare, selection.compare)) update({ target, compare });
  }, [pinned, selectedNodeId, ports, selection.target, selection.compare, update]);

  const shown = frame.status === 'idle' ? null : frame.frame;
  const shownValues = shown?.values;
  useEffect(() => {
    onValues(shownValues ? new Map(Object.entries(shownValues)) : null);
  }, [shownValues, onValues]);
  useEffect(() => () => onValues(null), [onValues]);
  const view = shown?.view ?? null;
  const overlays = useMemo(
    () => (shown?.sequences && view ? cropsAt(shown.sequences, view, shown.n) : []),
    [shown, view],
  );
  const chunkList = chunks.status === 'ready' ? chunks.chunks : [];
  const videos = load.status === 'idle' ? [] : load.videos;
  const frames = shown?.frames ?? 0;
  const busy = load.status === 'loading' || chunks.status === 'loading' || frame.status === 'loading';
  const error = load.status === 'error' ? load.error : chunks.status === 'error' ? chunks.error : frame.status === 'error' ? frame.error : null;

  const setFrameIndex = useCallback((n: number) => update({ frame: Math.max(0, Math.min(n, Math.max(0, frames - 1))) }), [frames, update]);

  const onResizeDown = (event: React.PointerEvent) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    resize.current = { y: event.clientY, height };
  };
  const onResizeMove = (event: React.PointerEvent) => {
    const start = resize.current;
    if (!start) return;
    const parent = (event.currentTarget.parentElement?.parentElement?.clientHeight ?? 900) - 120;
    setHeight(Math.round(Math.min(Math.max(MIN_HEIGHT, parent), Math.max(MIN_HEIGHT, start.height + start.y - event.clientY))));
  };

  const jump = (item: { frames: Record<string, number[]> }) => {
    const first = view ? item.frames[view]?.[0] : undefined;
    if (first !== undefined) setFrameIndex(first);
  };

  const sequenceRow = (item: { frames: Record<string, number[]>; crops: Record<string, number[]>; info: Record<string, unknown> }, index: number, rejected: boolean) => {
    const numbers = view ? item.frames[view] ?? [] : [];
    const crop = view ? item.crops[view] : undefined;
    const overlay = overlays.find(o => o.rejected === rejected && o.index === index) ?? null;
    return (
      <li key={`${rejected ? 'r' : 'k'}${index}`}>
        <button
          type="button"
          onClick={() => jump(item)}
          onMouseEnter={() => setHighlighted(overlay)}
          onMouseLeave={() => setHighlighted(null)}
          disabled={numbers.length === 0}
          title={JSON.stringify(item.info)}
          className={`w-full px-3 py-1 text-left text-[11.5px] border-b border-ink-900 enabled:hover:bg-ink-850 ${overlay ? 'bg-ink-850/60' : ''}`}
        >
          <div className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-sm flex-shrink-0 border ${rejected ? 'border-bad-400 border-dashed' : 'border-accent-500 bg-accent-500/40'}`} />
            <span className="text-ink-200 tabular-nums">#{index + 1}</span>
            <span className="text-ink-500 tabular-nums truncate">
              {numbers.length ? `frames ${numbers[0]}–${numbers[numbers.length - 1]}` : `not in ${view}`}
              {crop ? ` · ${crop[2]}×${crop[3]} at ${crop[0]},${crop[1]}` : ''}
            </span>
          </div>
          {rejected && <div className="pl-3.5 text-bad-400 truncate">{String(item.info.rejected ?? 'rejected')}</div>}
        </button>
      </li>
    );
  };

  const settings = shown ? Object.entries(shown.settings) : [];
  const targetType = ports.find(p => samePort(p, selection.target))?.type;
  const compareType = ports.find(p => samePort(p, selection.compare))?.type;

  return (
    <div className="flex-shrink-0 flex flex-col bg-ink-900 border-t border-ink-800 min-h-0" style={{ height }}>
      <div
        role="separator"
        aria-orientation="horizontal"
        aria-label="Resize the preview"
        onPointerDown={onResizeDown}
        onPointerMove={onResizeMove}
        onPointerUp={() => { resize.current = null; }}
        className="h-1.5 -mt-1 flex-shrink-0 cursor-ns-resize hover:bg-accent-500/30 transition-colors relative z-10"
      />

      {/* What is shown */}
      <div className="h-9 flex-shrink-0 flex items-center gap-2 px-3 border-b border-ink-800 min-w-0">
        <span className="font-display text-[12px] font-semibold uppercase tracking-[0.14em] text-ink-200 flex-shrink-0">Preview</span>
        <select
          aria-label="Output to preview"
          value={portKey(selection.target)}
          onChange={e => { setPinned(true); update({ target: parsePortKey(e.target.value), view: null }); }}
          className={`${SELECT} max-w-[220px]`}
        >
          {!selection.target && <option value="">Pick an output</option>}
          {ports.map(p => <option key={portKey(p)} value={portKey(p)}>{p.label}</option>)}
        </select>
        <button
          type="button"
          onClick={() => setPinned(p => !p)}
          aria-pressed={pinned}
          title={pinned ? 'Pinned: the preview stays on this output. Click to follow the canvas selection' : 'Following the canvas selection. Click to pin this output'}
          className={`${BTN} ${pinned ? PRESSED : ''}`}
        >
          {pinned ? <Pin className="w-3 h-3" /> : <PinOff className="w-3 h-3" />}
        </button>
        {targetType === 'sequences' && shown?.clips && shown.clips.length > 1 && (
          <select aria-label="Clip to show" value={view ?? ''} onChange={e => update({ view: e.target.value })} className={SELECT}>
            {shown.clips.map(name => <option key={name} value={name}>{name}</option>)}
          </select>
        )}
        <span className="w-px h-4 bg-ink-800 flex-shrink-0" />
        <span className={LABEL}>Compare</span>
        <select
          aria-label="Output to compare with"
          value={portKey(selection.compare)}
          onChange={e => update({ compare: parsePortKey(e.target.value), compareView: null })}
          className={`${SELECT} max-w-[200px]`}
        >
          <option value="">None</option>
          {ports.filter(p => !samePort(p, selection.target)).map(p => <option key={portKey(p)} value={portKey(p)}>{p.label}</option>)}
        </select>
        {compareType === 'sequences' && shown?.compare?.clips && shown.compare.clips.length > 1 && (
          <select aria-label="Compare clip" value={shown.compare.view ?? ''} onChange={e => update({ compareView: e.target.value })} className={SELECT}>
            {shown.compare.clips.map(name => <option key={name} value={name}>{name}</option>)}
          </select>
        )}
        {selection.compare && (
          <>
            <Segmented title="Compare layout" value={mode} onChange={setMode} options={[{ value: 'side', label: 'Side' }, { value: 'wipe', label: 'Wipe' }]} />
            <button
              type="button"
              onClick={() => setSmooth(s => !s)}
              aria-pressed={smooth}
              title="How the compare picture is scaled to the shown one's size"
              className={`${BTN} ${smooth ? PRESSED : ''}`}
            >
              {smooth ? 'Smooth' : 'Nearest'}
            </button>
          </>
        )}
        <div className="flex-1" />
        {busy && <Loader2 className="w-3.5 h-3.5 animate-spin text-ink-400 flex-shrink-0" aria-label="Loading" />}
        {shown && !busy && (
          <span className="text-[11px] text-ink-500 tabular-nums flex-shrink-0" title={shown.cached ? 'From this chunk’s cache' : 'Computed for this chunk'}>
            {shown.width}×{shown.height} · {Math.round(shown.ms)} ms
          </span>
        )}
        <Segmented title="Zoom" value={zoom} onChange={setZoom} options={ZOOMS} />
        <button type="button" onClick={onClose} title="Close the preview" aria-label="Close the preview" className={`${BTN} px-1`}>
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="flex-1 flex min-h-0">
        <div className="flex-1 flex flex-col min-w-0 min-h-0 relative">
          {shown ? (
            <PrivacyVeil enabled={privacyMode} className="flex-1 flex min-h-0" label="Preview hidden — click to reveal">
              <DatasetPreviewViewer
                frame={shown}
                mainLabel={`${portLabel({ node: shown.node, port: shown.port })}${shown.view ? ` (${shown.view})` : ''}`}
                compareLabel={shown.compare ? `${portLabel({ node: shown.compare.node, port: shown.compare.port })}${shown.compare.view ? ` (${shown.compare.view})` : ''}` : null}
                mode={mode}
                smooth={smooth}
                zoom={zoom}
                overlays={overlays}
                highlighted={highlighted}
                dimmed={frame.status !== 'ready'}
              />
            </PrivacyVeil>
          ) : (
            <div className="flex-1 grid place-items-center px-6">
              <p className="text-[12px] text-ink-500 text-center max-w-[360px] leading-relaxed">
                {ports.length === 0
                  ? 'Nothing to preview yet: add a node with a clip or sequences output.'
                  : !selection.target
                    ? 'Select a node on the canvas, or pick an output above, to see it here.'
                    : load.status === 'loading' || busy ? 'Loading the job' : ''}
              </p>
            </div>
          )}
          {error && (
            <div className="absolute left-3 right-3 bottom-3 z-10 flex items-start gap-2 px-3 py-2 rounded-md border border-bad-500/40 bg-ink-950/95 text-[12px] text-ink-200">
              <AlertCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0 text-bad-400" />
              <span className="flex-1 min-w-0 break-words">{error}</span>
              <button type="button" onClick={retry} className={BTN}>
                <RefreshCw className="w-3 h-3" />
                Retry
              </button>
            </div>
          )}
        </div>

        {/* What the chunk drew, and its sequences */}
        <div className="w-[280px] flex-shrink-0 overflow-y-auto border-l border-ink-800 bg-ink-950">
          <div className={HEADING}>Drawn for this chunk</div>
          {!shown && <div className="px-3 py-2 text-[11.5px] text-ink-600">Nothing shown yet.</div>}
          {shown && settings.length === 0 && shown.skipped.length === 0 && (
            <div className="px-3 py-2 text-[11.5px] text-ink-500">Nothing upstream draws a setting.</div>
          )}
          {settings.map(([id, drawn]) => (
            <div key={id} className="px-3 py-1.5 border-b border-ink-900">
              <div className="text-[11.5px] text-ink-200 truncate" title={id}>{labels.get(id) ?? id}</div>
              {drawn.params && (
                <dl className="mt-0.5 grid grid-cols-[auto_1fr] gap-x-2 text-[11px]">
                  {Object.entries(drawn.params).map(([key, value]) => (
                    <div key={key} className="contents">
                      <dt className="text-ink-500">{key.replace(/_/g, ' ')}</dt>
                      <dd className="text-ink-300 text-right tabular-nums break-all">{formatDrawn(value)}</dd>
                    </div>
                  ))}
                </dl>
              )}
            </div>
          ))}
          {shown && shown.skipped.length > 0 && (
            <div className="px-3 py-1.5 border-b border-ink-900 text-[11px] text-ink-500">
              Not applied on this chunk: {shown.skipped.map(id => labels.get(id) ?? id).join(', ')}
            </div>
          )}

          {shown?.sequences && (
            <>
              <div className={HEADING}>
                Sequences
                <span className="ml-auto normal-case tracking-normal font-normal text-[11px]">
                  {shown.sequences.items.length} kept · {shown.sequences.rejected.length} rejected
                </span>
              </div>
              <ul>
                {shown.sequences.items.map((item, i) => sequenceRow(item, i, false))}
                {shown.sequences.rejected.map((item, i) => sequenceRow(item, i, true))}
              </ul>
            </>
          )}

          {shown && shown.messages.length > 0 && (
            <>
              <div className={HEADING}>Messages</div>
              {shown.messages.map((message, i) => (
                <div key={i} className="px-3 py-1 text-[11px] text-ink-400 border-b border-ink-900">{message}</div>
              ))}
            </>
          )}
        </div>
      </div>

      {/* Where in the job */}
      <div className="h-10 flex-shrink-0 flex items-center gap-2 px-3 border-t border-ink-800 min-w-0">
        {videos.length > 1 && (
          <select
            aria-label="Video"
            value={selection.video}
            onChange={e => update({ video: Number(e.target.value), chunk: 0, frame: 0 })}
            className={`${SELECT} max-w-[160px]`}
          >
            {videos.map(v => <option key={v.index} value={v.index}>{privacyMode ? `Video ${v.index + 1}` : v.name}</option>)}
          </select>
        )}
        <button type="button" onClick={() => update({ chunk: selection.chunk - 1 })} disabled={selection.chunk <= 0} title="Previous chunk" aria-label="Previous chunk" className={`${BTN} px-1`}>
          <ChevronLeft className="w-3.5 h-3.5" />
        </button>
        <select
          aria-label="Chunk"
          value={selection.chunk}
          onChange={e => update({ chunk: Number(e.target.value) })}
          disabled={chunkList.length === 0}
          className={`${SELECT} w-[170px]`}
        >
          {chunkList.length === 0 && <option value={selection.chunk}>Chunk</option>}
          {chunkList.map((range, i) => (
            <option key={i} value={i}>{chunks.status === 'ready' && !chunks.chunked ? 'Whole video' : `Chunk ${i + 1} · ${rangeLabel(range)}`}</option>
          ))}
        </select>
        <button type="button" onClick={() => update({ chunk: selection.chunk + 1 })} disabled={selection.chunk >= chunkList.length - 1} title="Next chunk" aria-label="Next chunk" className={`${BTN} px-1`}>
          <ChevronRight className="w-3.5 h-3.5" />
        </button>
        {variants > 1 && (
          <select
            aria-label="Variant"
            title="A writer with variants writes each sequence several times, each with its own LR draws"
            value={Math.min(selection.variant, variants - 1)}
            onChange={e => update({ variant: Number(e.target.value) })}
            className={SELECT}
          >
            {Array.from({ length: variants }, (_, i) => <option key={i} value={i}>Variant {i + 1}</option>)}
          </select>
        )}
        <span className="w-px h-4 bg-ink-800 flex-shrink-0" />
        <button type="button" onClick={() => setFrameIndex(selection.frame - 1)} disabled={!shown || selection.frame <= 0} title="Previous frame" aria-label="Previous frame" className={`${BTN} px-1`}>
          <ChevronLeft className="w-3.5 h-3.5" />
        </button>
        <input
          type="range"
          aria-label="Frame in the chunk"
          min={0}
          max={Math.max(0, frames - 1)}
          value={Math.min(selection.frame, Math.max(0, frames - 1))}
          disabled={!shown || frames <= 1}
          onChange={e => setFrameIndex(Number(e.target.value))}
          className="flex-1 min-w-[80px] accent-[rgb(var(--accent-500))]"
        />
        <button type="button" onClick={() => setFrameIndex(selection.frame + 1)} disabled={!shown || selection.frame >= frames - 1} title="Next frame" aria-label="Next frame" className={`${BTN} px-1`}>
          <ChevronRight className="w-3.5 h-3.5" />
        </button>
        <span className="text-[11px] text-ink-400 tabular-nums flex-shrink-0 w-[150px] truncate" title="Frame within the chunk, and in the video">
          {shown ? `${shown.n + 1} / ${frames}${shown.range && (shown.scope === 'chunk' || shown.kind === 'clip') ? ` · video ${shown.range[0] + shown.n}` : ''}` : '—'}
        </span>
        <span className="w-px h-4 bg-ink-800 flex-shrink-0" />
        <button type="button" onClick={() => update({ seed: randomSeed() })} title="Preview other random draws for this chunk; the job's seed stays as it is" className={BTN}>
          <Dices className="w-3.5 h-3.5" />
          Reroll
        </button>
        {selection.seed !== null && (
          <>
            <span className="text-[11px] text-warn-400 tabular-nums flex-shrink-0" title="The preview draws from this seed instead of the job's">
              Seed {selection.seed}
            </span>
            <button type="button" onClick={() => { onAdoptSeed(selection.seed!); update({ seed: null }); }} title="Make this the job's seed" className={BTN}>
              <Check className="w-3.5 h-3.5" />
              Use
            </button>
            <button type="button" onClick={() => update({ seed: null })} title="Back to the job's seed" aria-label="Back to the job's seed" className={`${BTN} px-1`}>
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </>
        )}
      </div>
    </div>
  );
});

DatasetPreviewPanel.displayName = 'DatasetPreviewPanel';
