// src/components/dataset/DatasetNodeSettings.tsx — the settings inside a node
// card, one compact row each. The first few show; the rest fold behind a "more" toggle so a node with
// many settings does not dominate the canvas. The fold is per card and is not
// saved. A setting that shows another's preset follows that choice until edited.
//
// A setting that takes a link has a small value socket on the card's left
// edge, Blender style: it shows while the card is hovered or a value wire is
// being drawn, and stays while linked. A linked setting shows the value node
// it comes from (and its value for the previewed chunk) instead of its editor,
// and never folds away, so its wire always has a row to end on. On a value
// node, a setting that holds what the node gives is edited with the choices
// of the setting the node feeds.

import { memo, useState } from 'react';
import { Handle, Position } from '@xyflow/react';
import { ChevronDown, Link2, X } from 'lucide-react';
import { DatasetParamField, Row } from './DatasetParamField';
import { DatasetOptionsField } from './DatasetOptionsField';
import { portColor, type PreviewValue, type SettingLink } from './DatasetNodeCard';
import type { DatasetNodeSpec, DatasetParamSpec } from '../../utils/datasetGraph';
import { paramHandle, VALUE_PORT } from '../../utils/datasetLinks';
import { formatDrawn } from '../../utils/datasetPreview';
import { cardSettings } from '../../utils/datasetSummary';

/** How many settings a card shows before the rest fold away. */
export const VISIBLE_SETTINGS = 7;

interface DatasetNodeSettingsProps {
  nodeId: string;
  spec: DatasetNodeSpec;
  params: Record<string, unknown>;
  /** The settings driven by a link, by name. */
  links?: Map<string, SettingLink>;
  /** For a value node: the setting its value goes into. */
  target?: DatasetParamSpec;
  /** What each value node gave for the previewed chunk, while the preview shows one. */
  values: Map<string, PreviewValue> | null;
  onParamChange: (id: string, name: string, value: unknown) => void;
  onUnlink: (id: string, name: string) => void;
}

/**
 * A `gives` setting as the editor of the setting the node feeds: its
 * dropdown or number box, with the dice menu, as the engine draws it.
 */
function asTarget(param: DatasetParamSpec, target: DatasetParamSpec | undefined): DatasetParamSpec {
  if (!param.gives || param.kind !== 'value' || !target) return param;
  if (target.kind === 'choice') return { ...param, kind: 'choice', choices: target.choices, random: true };
  if (target.kind === 'float' || target.kind === 'int') {
    return { ...param, kind: target.kind, min: target.min, max: target.max, random: true };
  }
  if (target.kind === 'bool') return { ...param, kind: 'bool', random: false };
  return param;
}

/** A linked setting: the value node it comes from, its value for the previewed chunk, and an unlink button. */
const LinkedRow = ({ param, link, value, onUnlink }: {
  param: DatasetParamSpec;
  link: SettingLink;
  value?: PreviewValue;
  onUnlink: () => void;
}) => {
  const color = portColor(VALUE_PORT);
  const shown = value === undefined ? null : value.error ? 'error' : formatDrawn(value.value);
  return (
    <Row
      label={param.label}
      help={param.help || param.name}
      control={
        <>
          <span
            title={`Linked to ${link.label} (${link.source}): its value replaces this setting${shown ? `\nThis chunk: ${value?.error ?? shown}` : ''}`}
            className="flex-1 min-w-0 h-[18px] px-1.5 rounded-[3px] inline-flex items-center gap-1 text-[11px] text-ink-400 border border-dashed"
            style={{ borderColor: `color-mix(in srgb, ${color} 45%, transparent)`, background: `color-mix(in srgb, ${color} 8%, transparent)` }}
          >
            <Link2 className="w-3 h-3 flex-shrink-0" style={{ color }} aria-hidden="true" />
            <span className="truncate">{link.label}</span>
            {shown !== null && <span className={`ml-auto pl-1 flex-shrink-0 tabular-nums ${value?.error ? 'text-bad-400' : 'text-ink-200'}`}>{shown}</span>}
          </span>
          <button
            type="button"
            onClick={onUnlink}
            title="Unlink: the setting goes back to its own value"
            aria-label={`Unlink ${param.label}`}
            className="w-4 h-[18px] flex-shrink-0 rounded-[3px] grid place-items-center text-ink-500 hover:text-ink-200 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500"
          >
            <X className="w-3 h-3" />
          </button>
        </>
      }
    />
  );
};

export const DatasetNodeSettings = memo<DatasetNodeSettingsProps>(({ nodeId, spec, params, links, target, values, onParamChange, onUnlink }: DatasetNodeSettingsProps) => {
  const [expanded, setExpanded] = useState(false);
  const all = cardSettings(spec);
  const shown = expanded ? all : all.filter((param, i) => i < VISIBLE_SETTINGS || links?.has(param.name));
  const hidden = all.length - shown.length;

  const editor = (param: DatasetParamSpec) => {
    const value = params[param.name] ?? param.default;
    if (param.presetFrom) {
      // Unedited, it holds nothing and shows the preset of the choice it follows.
      const from = spec.params.find(p => p.name === param.presetFrom);
      const preset = param.presets?.[String(params[param.presetFrom] ?? from?.default)] ?? '';
      return (
        <DatasetParamField
          param={{ ...param, default: preset }}
          value={value || preset}
          onChange={next => onParamChange(nodeId, param.name, next === preset ? param.default : next)}
        />
      );
    }
    const change = (next: unknown) => {
      onParamChange(nodeId, param.name, next);
      // A setting that shows this one's preset goes back to the new preset.
      for (const follower of spec.params) {
        if (follower.presetFrom === param.name) onParamChange(nodeId, follower.name, follower.default);
      }
    };
    if (param.kind === 'options') return <DatasetOptionsField param={param} value={value} target={target} onChange={change} />;
    return <DatasetParamField param={asTarget(param, target)} value={value} onChange={change} />;
  };

  const field = (param: DatasetParamSpec) => {
    const link = links?.get(param.name);
    return (
      <div key={param.name} className="relative">
        {param.linkable && (
          <Handle
            type="target"
            position={Position.Left}
            id={paramHandle(param.name)}
            title={link ? `${param.label}: linked to ${link.label}` : `${param.label}: drop a value wire here to drive it`}
            className={`vk-param-socket ${link ? 'vk-linked' : ''}`}
            style={{ background: portColor(VALUE_PORT), borderColor: 'rgb(var(--ink-950))', borderWidth: 1.5, width: 8, height: 8, top: 11 }}
          />
        )}
        {link
          ? <LinkedRow param={param} link={link} value={values?.get(link.source)} onUnlink={() => onUnlink(nodeId, param.name)} />
          : editor(param)}
      </div>
    );
  };

  if (all.length === 0) return null;
  return (
    <div className="nodrag nowheel nopan cursor-auto border-t border-ink-750 py-1">
      {shown.map(field)}
      {(hidden > 0 || expanded) && all.length > VISIBLE_SETTINGS && (
        <button
          type="button"
          onClick={() => setExpanded(e => !e)}
          aria-expanded={expanded}
          className="mx-2.5 h-5 flex items-center gap-1 text-[11px] text-ink-500 hover:text-ink-200 transition-colors rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500"
        >
          <ChevronDown className={`w-3 h-3 transition-transform ${expanded ? 'rotate-180' : ''}`} />
          {expanded ? 'Fewer' : `${hidden} more`}
        </button>
      )}
    </div>
  );
});

DatasetNodeSettings.displayName = 'DatasetNodeSettings';
