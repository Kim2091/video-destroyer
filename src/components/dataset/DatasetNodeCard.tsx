// src/components/dataset/DatasetNodeCard.tsx — one node on the dataset canvas.
//
// Drawn entirely from the node's declaration: a header band tinted with its
// stage's colour that holds its name, an optional one-line note, one row per
// output port (right edge) and then per input port (left edge), and its
// settings, one compact row each, edited in place. The header also holds the
// on/off switch, the chance and issues badges, a "?" popover with what the
// node does, and a menu for rename, note and delete. A port's colour is its
// type's, and only ports of one type connect (datasetGraph.canConnect). While
// the chain heat view is on, the header and border take the node's place
// along its chain instead, with its step number. Every card is CARD_WIDTH
// wide so columns line up. A value node's card is tinted with the value
// colour, and while the preview runs its output shows the value it gives
// for the previewed chunk.

import { createContext, memo, useCallback, useContext, useState, type ReactNode } from 'react';
import { Handle, Position, type NodeProps, type Node } from '@xyflow/react';
import { AlertTriangle, CircleHelp, Dices, MoreHorizontal, Pencil, Power, StickyNote, Trash2 } from 'lucide-react';
import { DatasetNodeSettings } from './DatasetNodeSettings';
import { DatasetPopover } from './DatasetPopover';
import { nodeLabel, type DatasetNodeData, type DatasetNodeSpec, type DatasetParamSpec, type DatasetPortInfo, type GraphIssue } from '../../utils/datasetGraph';
import { CARD_WIDTH } from '../../utils/datasetLayout';
import { isValueSpec, VALUE_PORT } from '../../utils/datasetLinks';
import { formatDrawn } from '../../utils/datasetPreview';
import { heatColor, stageColor } from '../../utils/datasetStages';
import { inputShares, nodeChance } from '../../utils/datasetSummary';

/** The value node that drives a setting through a link. */
export interface SettingLink {
  source: string;
  label: string;
}

/** What a value node gave for the previewed chunk. */
export interface PreviewValue {
  value?: unknown;
  error?: string;
}

/** How far along its chain each node is, while the chain heat view is on. */
export interface ChainHeat {
  depths: Map<string, number>;
  deepest: number;
}

export interface DatasetCanvasContext {
  specs: Map<string, DatasetNodeSpec>;
  /** Ids of nodes something is wrong with. */
  flagged: Set<string>;
  /** Ids of nodes with a stage-order hint. */
  hinted: Set<string>;
  /** Node id -> its inputs that have an edge. */
  connected: Map<string, Set<string>>;
  /** Node id -> its problems. */
  issues: Map<string, GraphIssue[]>;
  /** The chain heat view, or null when it is off. */
  heat: ChainHeat | null;
  /** Node id -> setting name -> the value node linked into it. */
  links: Map<string, Map<string, SettingLink>>;
  /** Value node id -> the setting its value goes into, followed through value nodes that pass it on. */
  targets: Map<string, DatasetParamSpec>;
  /** Value node id -> what it gave for the previewed chunk, while the preview shows one. */
  values: Map<string, PreviewValue> | null;
  onToggleEnabled: (id: string) => void;
  onLabelChange: (id: string, label: string) => void;
  onNoteChange: (id: string, note: string) => void;
  onParamChange: (id: string, name: string, value: unknown) => void;
  onUnlink: (id: string, name: string) => void;
  onRemove: (id: string) => void;
}

export const DatasetCanvas = createContext<DatasetCanvasContext>({
  specs: new Map(),
  flagged: new Set(),
  hinted: new Set(),
  connected: new Map(),
  issues: new Map(),
  heat: null,
  links: new Map(),
  targets: new Map(),
  values: null,
  onToggleEnabled: () => {},
  onLabelChange: () => {},
  onNoteChange: () => {},
  onParamChange: () => {},
  onUnlink: () => {},
  onRemove: () => {},
});

/** Port colours: pale or neutral, so a wire never reads as a stage colour. */
const PORT_COLORS: Record<string, string> = {
  clip: '#e4e7ec',
  sequences: '#f2d54a',
  static_runs: '#a8e06a',
  mask: '#8f96a1',
  signal: '#8fcfff',
  [VALUE_PORT]: '#c4b0ff',
};

/** Plain names for the port types, for port tooltips; a type without one shows as itself. */
const PORT_NAMES: Record<string, string> = {
  clip: 'video', sequences: 'sequences', static_runs: 'still shots', mask: 'a mask', signal: 'an analogue signal',
  [VALUE_PORT]: 'a value, into any setting\'s dot',
};

/** The colour edges and ports of a type are drawn in. */
export const portColor = (type: string | undefined) => (type && PORT_COLORS[type]) || 'rgb(var(--ink-400))';

/** A node type's own colour: its stage's, or the value colour for a value node. */
export const nodeColor = (spec: DatasetNodeSpec | undefined) =>
  (isValueSpec(spec) ? portColor(VALUE_PORT) : stageColor(spec?.stage));

/** A colour mixed into the card body, for a tinted header band. */
const tint = (color: string, amount: number) => `color-mix(in srgb, ${color} ${amount}%, rgb(var(--ink-800)))`;

/** A stage as a small coloured number and name, such as "1 Load"; a group with no stage gives its own colour. */
export const StageTag = ({ stage, name, color }: { stage?: number | null; name?: string; color?: string }) => (
  <span className="inline-flex items-center gap-1.5 text-[11px] font-medium whitespace-nowrap" style={{ color: color ?? stageColor(stage) }}>
    <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: 'currentColor' }} aria-hidden="true" />
    {stage ? `${stage} ` : ''}{name || 'Other'}
  </span>
);

const ICON_BTN = 'nodrag w-5 h-5 rounded grid place-items-center flex-shrink-0 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500';
const IDLE = 'text-white/45 hover:text-white hover:bg-black/20';
const ACTIVE = 'text-white bg-black/25';

/** What the node does and what each port carries, for the "?" popover. */
const NodeHelp = ({ spec }: { spec: DatasetNodeSpec }) => {
  const ports = [
    ...Object.keys(spec.inputs).map(name => ({ side: 'In', name, info: spec.inputInfo?.[name], optional: spec.optionalInputs.includes(name) })),
    ...Object.keys(spec.outputs).map(name => ({ side: 'Out', name, info: spec.outputInfo?.[name], optional: false })),
  ];
  return (
    <div className="w-[272px] max-h-[320px] overflow-y-auto px-3 py-2.5 text-[11.5px] leading-relaxed text-ink-400 space-y-2">
      <p className="text-ink-300">{spec.description}</p>
      {spec.help && <p>{spec.help}</p>}
      {ports.length > 0 && (
        <ul className="space-y-0.5">
          {ports.map(port => (
            <li key={`${port.side}:${port.name}`} className="flex gap-2">
              <span className="w-7 flex-shrink-0 text-ink-600">{port.side}</span>
              <span>
                <span className="text-ink-300">{port.info?.label || port.name}</span>
                {port.optional && ' (optional)'}
                {port.info?.help && <>: {port.info.help}</>}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

const MenuItem = ({ icon, label, onClick, danger }: { icon: ReactNode; label: string; onClick: () => void; danger?: boolean }) => (
  <button
    type="button"
    role="menuitem"
    onClick={onClick}
    className={`w-36 h-7 px-2.5 flex items-center gap-2 text-[12px] text-left text-ink-300 hover:bg-ink-800 ${danger ? 'hover:text-bad-400' : 'hover:text-ink-100'}`}
  >
    {icon}
    {label}
  </button>
);

const Port = ({ name, type, side, info, optional, share, readout }: {
  name: string;
  type: string;
  side: 'in' | 'out';
  info?: DatasetPortInfo;
  optional?: boolean;
  /** How often this input is picked, for a node that picks one input per chunk. */
  share?: number;
  /** What a value output gave for the previewed chunk, shown in place of its name. */
  readout?: PreviewValue;
}) => {
  const label = info?.label || name;
  const odds = share === undefined ? '' : `\nPicked on about ${Math.round(share * 100)}% of chunks`;
  const now = readout === undefined ? '' : readout.error ? `\nThis chunk: ${readout.error}` : `\nThis chunk: ${formatDrawn(readout.value)}`;
  const tip = `${label}${optional ? ' (optional)' : ''}${info?.help ? `: ${info.help}` : ''}\nCarries ${PORT_NAMES[type] ?? type}${odds}${now}`;
  return (
    <div title={tip} className={`relative h-5 flex items-center text-[11px] min-w-0 ${side === 'in' ? 'pl-2.5 justify-start' : 'pr-2.5 justify-end'}`}>
      <Handle
        type={side === 'in' ? 'target' : 'source'}
        position={side === 'in' ? Position.Left : Position.Right}
        id={name}
        title={tip}
        style={{ background: portColor(type), borderColor: 'rgb(var(--ink-950))', borderWidth: 2, width: 11, height: 11 }}
      />
      {readout === undefined ? (
        <span className={`truncate ${optional ? 'text-ink-500 italic' : 'text-ink-300'}`}>{label}</span>
      ) : (
        <span className={`truncate tabular-nums font-medium ${readout.error ? 'text-bad-400' : 'text-ink-100'}`}>
          {readout.error ? 'Error' : `= ${formatDrawn(readout.value)}`}
        </span>
      )}
      {share !== undefined && <span className="ml-1 text-[10.5px] tabular-nums text-chain-300 flex-shrink-0">{Math.round(share * 100)}%</span>}
    </div>
  );
};

export const DatasetNodeCard = memo(({ id, data, selected }: NodeProps<Node<DatasetNodeData>>) => {
  const { specs, flagged, hinted, connected, issues, heat, links, targets, values, onToggleEnabled, onLabelChange, onNoteChange, onParamChange, onUnlink, onRemove } = useContext(DatasetCanvas);
  const [panel, setPanel] = useState<'help' | 'menu' | null>(null);
  const [renaming, setRenaming] = useState(false);
  const [editingNote, setEditingNote] = useState(false);
  const closePanel = useCallback(() => setPanel(null), []);
  const spec = specs.get(data.type);
  const inputs = Object.entries(spec?.inputs ?? {});
  const outputs = Object.entries(spec?.outputs ?? {});
  const chance = nodeChance(spec, data.params);
  const shares = spec ? inputShares(spec, data.params, connected.get(id) ?? new Set()) : null;
  const problems = issues.get(id) ?? [];
  const bad = flagged.has(id) || !spec;
  const off = !data.enabled;
  const name = nodeLabel(data, spec);
  const note = typeof data.extra.note === 'string' ? data.extra.note : '';

  const depth = heat?.depths.get(id);
  const heated = depth !== undefined ? heatColor(depth / Math.max(1, heat!.deepest)) : null;
  const band = heated ? tint(heated, 32) : tint(nodeColor(spec), 50);
  const readout = isValueSpec(spec) && values ? values.get(id) : undefined;

  const border = heated
    ? ''
    : selected
      ? 'border-accent-400 ring-2 ring-accent-500/40'
      : bad ? 'border-bad-500' : hinted.has(id) ? 'border-warn-500/70' : 'border-ink-700';

  return (
    <div
      style={{ width: CARD_WIDTH, ...(heated ? { borderColor: heated, boxShadow: `0 0 0 1px ${heated}` } : {}) }}
      className={`relative rounded-md border bg-ink-800 shadow-lg shadow-black/40 transition-[opacity,filter] ${border} ${off ? 'opacity-50 grayscale' : ''}`}
    >
      <div
        className="relative h-[26px] pl-1 pr-1 flex items-center gap-1 min-w-0 rounded-t-[5px]"
        style={{ background: band }}
        title={spec?.stageName ? (spec.stage ? `${spec.stage} ${spec.stageName}` : spec.stageName) : undefined}
      >
        <button
          type="button"
          onClick={(event) => { event.stopPropagation(); onToggleEnabled(id); }}
          title={off ? 'Turn on' : 'Turn off (its input passes through)'}
          aria-label={off ? 'Turn this node on' : 'Turn this node off'}
          aria-pressed={!off}
          className={`${ICON_BTN} ${off ? IDLE : 'text-white/80 hover:text-white hover:bg-black/20'}`}
        >
          <Power className="w-3 h-3" />
        </button>
        {depth !== undefined && (
          <span
            title={`Step ${depth + 1} along its chain`}
            className="min-w-[18px] h-[16px] px-1 rounded-full grid place-items-center text-[10.5px] font-bold tabular-nums text-black flex-shrink-0"
            style={{ background: heated! }}
          >
            {depth + 1}
          </span>
        )}
        {renaming ? (
          <input
            type="text"
            autoFocus
            value={data.label ?? ''}
            placeholder={nodeLabel({ ...data, label: undefined }, spec)}
            aria-label="Name"
            onChange={e => onLabelChange(id, e.target.value)}
            onBlur={() => setRenaming(false)}
            onKeyDown={e => { if (e.key === 'Enter' || e.key === 'Escape') setRenaming(false); }}
            className="nodrag nowheel flex-1 min-w-0 h-5 px-1 rounded bg-ink-900 border border-accent-500 text-[12px] font-semibold text-ink-100 placeholder-ink-500 focus:outline-none"
          />
        ) : (
          <span
            className="flex-1 min-w-0 pl-0.5 text-[12px] font-semibold leading-tight text-white truncate"
            title={spec ? spec.description : `This version has no node type ${data.type}.`}
          >
            {name}
          </span>
        )}
        {chance !== null && (
          <span
            title={`Runs on about ${Math.round(chance * 100)}% of chunks; the others pass through unchanged`}
            className="inline-flex items-center gap-0.5 h-4 px-1 rounded text-[10px] font-medium tabular-nums text-white/85 bg-black/25 flex-shrink-0"
          >
            <Dices className="w-2.5 h-2.5" />
            {Math.round(chance * 100)}%
          </span>
        )}
        {problems.length > 0 && (
          <span
            title={problems.map(p => p.message).join('\n')}
            className="inline-flex items-center gap-0.5 h-4 px-1 rounded text-[10px] font-semibold tabular-nums text-white bg-bad-600 flex-shrink-0"
          >
            <AlertTriangle className="w-2.5 h-2.5" />
            {problems.length}
          </span>
        )}
        {spec && (
          <button
            type="button"
            data-popover-button
            onClick={() => setPanel(p => (p === 'help' ? null : 'help'))}
            aria-expanded={panel === 'help'}
            aria-label="What this node does"
            title="What this node does"
            className={`${ICON_BTN} ${panel === 'help' ? ACTIVE : IDLE}`}
          >
            <CircleHelp className="w-3 h-3" />
          </button>
        )}
        <button
          type="button"
          data-popover-button
          onClick={() => setPanel(p => (p === 'menu' ? null : 'menu'))}
          aria-expanded={panel === 'menu'}
          aria-haspopup="menu"
          aria-label="More"
          title="Rename, note, delete"
          className={`${ICON_BTN} ${panel === 'menu' ? ACTIVE : IDLE}`}
        >
          <MoreHorizontal className="w-3.5 h-3.5" />
        </button>
        {panel === 'help' && spec && <DatasetPopover onClose={closePanel}><NodeHelp spec={spec} /></DatasetPopover>}
        {panel === 'menu' && (
          <DatasetPopover onClose={closePanel}>
            <div role="menu" className="py-1">
              <MenuItem icon={<Pencil className="w-3.5 h-3.5" />} label="Rename" onClick={() => { setPanel(null); setRenaming(true); }} />
              <MenuItem
                icon={<StickyNote className="w-3.5 h-3.5" />}
                label={note ? 'Edit note' : 'Add note'}
                onClick={() => { setPanel(null); setEditingNote(true); }}
              />
              <MenuItem icon={<Trash2 className="w-3.5 h-3.5" />} label="Delete" danger onClick={() => onRemove(id)} />
            </div>
          </DatasetPopover>
        )}
      </div>

      {editingNote ? (
        <div className="nodrag nowheel nopan cursor-auto px-2 pt-1.5">
          <textarea
            value={note}
            rows={3}
            autoFocus
            placeholder="Note"
            aria-label="Note"
            onChange={e => onNoteChange(id, e.target.value)}
            onBlur={() => setEditingNote(false)}
            onKeyDown={e => { if (e.key === 'Escape') setEditingNote(false); }}
            className="w-full bg-ink-900 border border-accent-500 rounded px-1.5 py-1 text-[11px] leading-snug text-ink-200 placeholder-ink-500 focus:outline-none resize-y"
          />
        </div>
      ) : note && (
        <p
          className="px-2.5 pt-1 text-[11px] leading-snug text-ink-400 italic truncate cursor-default"
          title={note}
          onDoubleClick={() => setEditingNote(true)}
        >
          {note}
        </p>
      )}

      {(outputs.length > 0 || inputs.length > 0) && (
        <div className="py-1">
          {outputs.map(([port, type]) => (
            <Port key={port} name={port} type={type} side="out" info={spec?.outputInfo?.[port]} readout={type === VALUE_PORT ? readout : undefined} />
          ))}
          {inputs.map(([port, type]) => (
            <Port key={port} name={port} type={type} side="in" info={spec?.inputInfo?.[port]} optional={spec?.optionalInputs.includes(port)} share={shares?.get(port)} />
          ))}
        </div>
      )}

      {!spec && <p className="px-2.5 pb-1.5 text-[11px] text-bad-400">This version has no node type {data.type}.</p>}
      {problems.map((issue, i) => (
        <p key={i} title={issue.message} className="px-2.5 pb-1 flex items-start gap-1 text-[10.5px] leading-snug text-bad-400">
          <AlertTriangle className="w-2.5 h-2.5 mt-[2px] flex-shrink-0" />
          <span className="line-clamp-2">{issue.message}</span>
        </p>
      ))}

      {spec && (
        <DatasetNodeSettings
          nodeId={id}
          spec={spec}
          params={data.params}
          links={links.get(id)}
          target={targets.get(id)}
          values={values}
          onParamChange={onParamChange}
          onUnlink={onUnlink}
        />
      )}
    </div>
  );
});

DatasetNodeCard.displayName = 'DatasetNodeCard';
