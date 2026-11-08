// src/components/dataset/DatasetParamField.tsx — the control for one node
// setting, chosen by the declared kind, as one compact row: the label on the
// left and the value on the right. A setting marked random has a dice button
// at the row's end that picks a fixed value or one of the engine's samplers;
// a range shows as "low – high" in one field. Code and JSON show a one-line
// preview that opens into an editor under the row. The setting's help is the
// tooltip of its label, and a changed setting offers a reset on hover.

import { memo, useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { Check, ChevronDown, Dices, File, Folder, RotateCcw } from 'lucide-react';
import { DatasetPopover } from './DatasetPopover';
import type { DatasetParamSpec } from '../../utils/datasetGraph';
import {
  formatNumberList,
  isSampler,
  parseNumberList,
  samplerModeOf,
  samplerModes,
  samplerProblem,
  toSamplerMode,
  pickOf,
  withChoices,
  withWeights,
  type SamplerMode,
} from '../../utils/datasetSampler';

const BOX = 'h-[18px] rounded-[3px] bg-ink-950/60 border border-ink-700/70 text-[11px] text-ink-100 transition-colors';
const FIELD = `${BOX} w-full min-w-0 px-1.5 placeholder-ink-600 focus:outline-none focus:border-accent-500 hover:border-ink-600`;
const FIELD_BAD = 'border-bad-500/70 focus:border-bad-500';
const BARE = 'w-full min-w-0 h-full bg-transparent px-1 text-center text-[11px] text-ink-100 placeholder-ink-600 focus:outline-none';
const EDITOR = 'w-full bg-ink-950/60 border border-ink-700/70 rounded px-1.5 py-1 font-mono text-[11px] leading-snug text-ink-100 focus:outline-none focus:border-accent-500 resize-y';
const ROW_BTN = 'w-4 h-[18px] flex-shrink-0 rounded-[3px] grid place-items-center transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500';

const MODE_LABELS: Record<SamplerMode, string> = {
  fixed: 'Fixed',
  uniform: 'Range',
  int: 'Int range',
  choice: 'List',
};

const MODE_HELP: Record<SamplerMode, string> = {
  fixed: 'The same value on every chunk',
  uniform: 'Any number between low and high, drawn per chunk',
  int: 'A whole number between low and high, both included, drawn per chunk',
  choice: 'One of the listed values per chunk, optionally weighted',
};

// A dropdown's list is the options ticked in its pick.
const modeLabel = (mode: SamplerMode, param: DatasetParamSpec) =>
  param.kind === 'choice' && mode === 'choice' ? 'Random' : MODE_LABELS[mode];
const modeHelp = (mode: SamplerMode, param: DatasetParamSpec) =>
  param.kind === 'choice' && mode === 'choice' ? 'One of the ticked options per chunk, optionally weighted' : MODE_HELP[mode];

/**
 * A text box that holds what is typed until it reads as a value, so a
 * half-typed number or JSON object is never pushed into the job. The text is
 * replaced only when the value changes from elsewhere (an undo, a reset),
 * never by the box's own edit, which would reformat it under the cursor.
 */
function useDraft<T>(value: T, format: (value: T) => string, onChange: (value: T) => void) {
  const formatted = format(value);
  const [draft, setDraft] = useState(formatted);
  const [error, setError] = useState<string | null>(null);
  const emitted = useRef<string | null>(null);
  useEffect(() => {
    if (formatted === emitted.current) return;
    setDraft(formatted);
    setError(null);
  }, [formatted]);
  const emit = (next: T) => {
    emitted.current = format(next);
    setError(null);
    onChange(next);
  };
  return { draft, setDraft, error, setError, emit };
}

/**
 * A number box; `bare` draws it without a border, to sit inside a range
 * field. With `empty`, the box may be left empty, which is null and shows
 * `empty` as its placeholder.
 */
const NumberInput = ({ value, kind, min, max, onChange, ariaLabel, bare, empty }: {
  value: unknown;
  kind: string;
  min?: number;
  max?: number;
  onChange: (value: number | null) => void;
  ariaLabel: string;
  bare?: boolean;
  empty?: string;
}) => {
  const { draft, setDraft, error, setError, emit } = useDraft<unknown>(
    value, v => (typeof v === 'number' ? String(v) : ''), v => onChange(v as number | null));
  const commit = (text: string) => {
    setDraft(text);
    const n = Number(text);
    if (text.trim() === '' && empty !== undefined) return emit(null);
    if (text.trim() === '' || !Number.isFinite(n)) return setError('Enter a number.');
    if (kind === 'int' && !Number.isInteger(n)) return setError('Enter a whole number.');
    if (min !== undefined && n < min) return setError(`At least ${min}.`);
    if (max !== undefined && n > max) return setError(`At most ${max}.`);
    emit(n);
  };
  const look = bare ? `${BARE} ${error ? 'text-bad-400' : ''}` : `${FIELD} ${error ? FIELD_BAD : ''}`;
  return (
    <input
      type="text"
      inputMode={kind === 'int' ? 'numeric' : 'decimal'}
      aria-label={ariaLabel}
      aria-invalid={!!error}
      title={error ?? undefined}
      placeholder={empty}
      value={draft}
      onChange={e => commit(e.target.value)}
      className={`${look} tabular-nums`}
    />
  );
};

const ListInput = ({ values, kind, placeholder, onChange, ariaLabel }: {
  values: unknown;
  kind: string;
  placeholder: string;
  onChange: (values: number[]) => void;
  ariaLabel: string;
}) => {
  const { draft, setDraft, error, setError, emit } = useDraft<unknown>(values, formatNumberList, v => onChange(v as number[]));
  return (
    <input
      type="text"
      aria-label={ariaLabel}
      aria-invalid={!!error}
      title={error ?? placeholder}
      placeholder={placeholder}
      value={draft}
      onChange={e => {
        setDraft(e.target.value);
        const parsed = parseNumberList(e.target.value, kind);
        if ('error' in parsed) return setError(parsed.error);
        emit(parsed.values);
      }}
      className={`${FIELD} tabular-nums ${error ? FIELD_BAD : ''}`}
    />
  );
};

const JsonInput = ({ value, onChange, ariaLabel }: { value: unknown; onChange: (value: unknown) => void; ariaLabel: string }) => {
  const { draft, setDraft, error, setError, emit } = useDraft(value, v => JSON.stringify(v, null, 2) ?? '', onChange);
  return (
    <>
      <textarea
        aria-label={ariaLabel}
        aria-invalid={!!error}
        spellCheck={false}
        autoFocus
        value={draft}
        rows={Math.min(14, Math.max(3, draft.split('\n').length))}
        onChange={e => {
          setDraft(e.target.value);
          let parsed: unknown;
          try {
            parsed = JSON.parse(e.target.value);
          } catch (err) {
            setError(err instanceof Error ? err.message : 'Not valid JSON.');
            return;
          }
          emit(parsed);
        }}
        className={`${EDITOR} ${error ? FIELD_BAD : ''}`}
      />
      {error && <p className="mt-0.5 text-[10.5px] text-bad-400">{error}</p>}
    </>
  );
};

/** A path as its file name (the full path while editing and on hover), with file and folder pickers. */
const PathInput = ({ value, label, onChange }: { value: unknown; label: string; onChange: (value: string) => void }) => {
  const [editing, setEditing] = useState(false);
  const path = typeof value === 'string' ? value : '';
  const name = path.split(/[\\/]/).filter(Boolean).pop() ?? path;
  const pick = async (mode: 'file' | 'folder') => {
    const picked = await window.electronAPI.datasetSelectPath(mode);
    if (picked) onChange(picked);
  };
  return (
    <>
      <input
        type="text"
        aria-label={label}
        title={path || undefined}
        placeholder="None"
        value={editing ? path : name}
        onFocus={() => setEditing(true)}
        onBlur={() => setEditing(false)}
        onChange={e => onChange(e.target.value)}
        spellCheck={false}
        className={FIELD}
      />
      <button type="button" onClick={() => void pick('file')} title="Choose a file" aria-label="Choose a file" className={`${ROW_BTN} text-ink-500 hover:text-ink-200`}>
        <File className="w-3 h-3" />
      </button>
      <button type="button" onClick={() => void pick('folder')} title="Choose a folder" aria-label="Choose a folder" className={`${ROW_BTN} text-ink-500 hover:text-ink-200`}>
        <Folder className="w-3 h-3" />
      </button>
    </>
  );
};

/** A one-line preview that opens an editor under the row, for code and JSON. */
const Preview = ({ text, open, onToggle, label }: { text: string; open: boolean; onToggle: () => void; label: string }) => (
  <button
    type="button"
    onClick={onToggle}
    aria-expanded={open}
    aria-label={`${open ? 'Close' : 'Edit'} ${label}`}
    title={open ? 'Close the editor' : 'Edit'}
    className={`${FIELD} flex items-center gap-1 text-left font-mono text-ink-300 hover:text-ink-100`}
  >
    <span className="flex-1 min-w-0 truncate">{text || '—'}</span>
    <ChevronDown className={`w-3 h-3 flex-shrink-0 transition-transform ${open ? 'rotate-180' : ''}`} />
  </button>
);

/** The dice button at a random setting's row end, and its menu of modes. */
const ModeMenu = ({ param, value, onChange }: { param: DatasetParamSpec; value: unknown; onChange: (value: unknown) => void }) => {
  const [open, setOpen] = useState(false);
  const close = useCallback(() => setOpen(false), []);
  const mode = samplerModeOf(value);
  return (
    <div className="relative flex-shrink-0">
      <button
        type="button"
        data-popover-button
        onClick={() => setOpen(o => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`${param.label}: ${modeLabel(mode, param)}. Change how it is drawn`}
        title={`${modeLabel(mode, param)}: ${modeHelp(mode, param).toLowerCase()}. Click to change`}
        className={`${ROW_BTN} ${mode === 'fixed' ? 'text-ink-600 hover:text-ink-300' : 'text-chain-300 hover:text-chain-200'}`}
      >
        <Dices className="w-3 h-3" />
      </button>
      {open && (
        <DatasetPopover onClose={close} className="right-0 top-full mt-1">
          <div role="menu" aria-label={`${param.label} mode`} className="py-1">
            {samplerModes(param).map(m => (
              <button
                key={m}
                type="button"
                role="menuitemradio"
                aria-checked={mode === m}
                title={modeHelp(m, param)}
                onClick={() => { onChange(toSamplerMode(value, m, param)); setOpen(false); }}
                className="w-32 h-6 px-2 flex items-center gap-1.5 text-[11.5px] text-left text-ink-300 hover:bg-ink-800 hover:text-ink-100"
              >
                <Check className={`w-3 h-3 flex-shrink-0 ${mode === m ? 'text-accent-400' : 'invisible'}`} />
                {modeLabel(m, param)}
              </button>
            ))}
          </div>
        </DatasetPopover>
      )}
    </div>
  );
};

/**
 * A dropdown drawn at random: the ticked options in one field, which opens a
 * list of every option with a tick and a weight. The last ticked option
 * cannot be unticked, as a pick needs one.
 */
export const ChoicePicker = ({ param, sampler, onChange }: {
  param: DatasetParamSpec;
  sampler: Record<string, unknown>;
  onChange: (value: unknown) => void;
}) => {
  const [open, setOpen] = useState(false);
  const close = useCallback(() => setOpen(false), []);
  const options = param.choices ?? [];
  const picked = Array.isArray(sampler.choice) ? sampler.choice : [];
  const weights = Array.isArray(sampler.weights) ? (sampler.weights as number[]) : null;
  const weightOf = (option: unknown) => {
    const i = picked.indexOf(option);
    return i < 0 ? 1 : (weights?.[i] ?? 1);
  };
  const total = picked.reduce((sum: number, option) => sum + weightOf(option), 0);
  const share = (option: unknown) => (total > 0 ? `${Math.round((weightOf(option) / total) * 100)}%` : '–');
  const emit = (ticked: unknown[], weightFor: (option: unknown) => number) => {
    const ordered = options.filter(o => ticked.includes(o));
    onChange(pickOf(ordered, ordered.map(weightFor)));
  };
  const toggle = (option: unknown) => {
    if (picked.includes(option)) {
      if (picked.length > 1) emit(picked.filter(o => o !== option), weightOf);
    } else {
      emit([...picked, option], weightOf);
    }
  };
  const summary = picked.map(String).join(', ');
  return (
    <div className="relative flex-1 min-w-0">
      <button
        type="button"
        data-popover-button
        onClick={() => setOpen(o => !o)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={`${param.label}: one of ${summary}`}
        title={picked.map(o => `${String(o)} ${share(o)}`).join('\n')}
        className={`${FIELD} flex items-center gap-1 text-left`}
      >
        <span className="flex-1 min-w-0 truncate">{summary}</span>
        <ChevronDown className="w-3 h-3 flex-shrink-0 text-ink-500" />
      </button>
      {open && (
        <DatasetPopover onClose={close} className="right-0 top-full mt-1">
          <div role="group" aria-label={`${param.label} options`} className="py-1 w-52">
            {options.map(option => {
              const on = picked.includes(option);
              return (
                <div key={String(option)} className="h-6 px-2 flex items-center gap-1.5 text-[11.5px]">
                  <label className={`flex-1 min-w-0 flex items-center gap-1.5 cursor-pointer ${on ? 'text-ink-100' : 'text-ink-500'}`}>
                    <input
                      type="checkbox"
                      checked={on}
                      disabled={on && picked.length === 1}
                      onChange={() => toggle(option)}
                      className="w-3 h-3 accent-[rgb(var(--accent-500))]"
                    />
                    <span className="truncate">{String(option)}</span>
                  </label>
                  {on && (
                    <>
                      <div className="w-10 flex-shrink-0" title="Weight: how often this option is picked against the others">
                        <NumberInput
                          value={weightOf(option)}
                          kind="float"
                          min={0}
                          onChange={w => emit(picked, o => (o === option ? (w ?? 0) : weightOf(o)))}
                          ariaLabel={`${String(option)} weight`}
                        />
                      </div>
                      <span className="w-8 flex-shrink-0 text-right tabular-nums text-ink-500">{share(option)}</span>
                    </>
                  )}
                </div>
              );
            })}
          </div>
        </DatasetPopover>
      )}
    </div>
  );
};

/** A label and its control on one line; `children` sit under the row. */
export const Row = ({ label, help, dim, control, reset, children }: {
  label: string;
  help?: string;
  dim?: boolean;
  control: ReactNode;
  reset?: ReactNode;
  children?: ReactNode;
}) => (
  <div className="group px-2.5">
    <div className="h-[22px] flex items-center gap-1.5">
      <div className="w-[42%] flex-shrink-0 min-w-0 flex items-center gap-0.5">
        <span className={`min-w-0 truncate text-[11px] ${dim ? 'pl-2 text-ink-500' : 'text-ink-400'} ${help ? 'cursor-help' : ''}`} title={help}>
          {label}
        </span>
        {reset}
      </div>
      <div className="flex-1 min-w-0 flex items-center gap-0.5">{control}</div>
    </div>
    {children}
  </div>
);

interface DatasetParamFieldProps {
  param: DatasetParamSpec;
  value: unknown;
  onChange: (value: unknown) => void;
}

export const DatasetParamField = memo(({ param, value, onChange }: DatasetParamFieldProps) => {
  const [editorOpen, setEditorOpen] = useState(false);
  const isDefault = JSON.stringify(value) === JSON.stringify(param.default);
  const label = param.label;

  const reset = !isDefault && (
    <button
      type="button"
      onClick={() => onChange(JSON.parse(JSON.stringify(param.default ?? null)))}
      title="Back to the default"
      aria-label={`Reset ${label} to its default`}
      className="w-4 h-4 flex-shrink-0 rounded grid place-items-center text-ink-500 hover:text-ink-200 opacity-0 group-hover:opacity-100 focus-visible:opacity-100 transition-opacity"
    >
      <RotateCcw className="w-2.5 h-2.5" />
    </button>
  );
  const row = (control: ReactNode, under?: ReactNode) => (
    <Row label={label} help={param.help || param.name} control={control} reset={reset}>{under}</Row>
  );

  if (param.random && (param.kind === 'float' || param.kind === 'int')) {
    const mode = samplerModeOf(value);
    const sampler = isSampler(value) ? value : null;
    const problem = sampler ? samplerProblem(sampler, param) : null;
    const range = sampler && (mode === 'uniform' || mode === 'int') ? (sampler[mode] as unknown[]) : null;
    const rangeKind = mode === 'int' ? 'int' : 'float';
    let control: ReactNode;
    if (range) {
      control = (
        <div className={`${BOX} flex-1 min-w-0 flex items-center focus-within:border-accent-500 hover:border-ink-600`} title={MODE_HELP[mode]}>
          <NumberInput bare value={range[0]} kind={rangeKind} onChange={low => onChange({ [mode]: [low, range[1]] })} ariaLabel={`${label} low`} />
          <span className="text-ink-500 flex-shrink-0" aria-hidden="true">–</span>
          <NumberInput bare value={range[1]} kind={rangeKind} onChange={high => onChange({ [mode]: [range[0], high] })} ariaLabel={`${label} high`} />
        </div>
      );
    } else if (sampler && mode === 'choice') {
      control = (
        <ListInput
          values={sampler.choice}
          kind={param.kind}
          placeholder="Options, e.g. 0.5, 0.75, 1"
          onChange={choices => onChange(withChoices(sampler, choices))}
          ariaLabel={`${label} options`}
        />
      );
    } else {
      // A setting whose default is empty may be left empty: the node then uses its own value.
      const empty = param.default === null ? 'Auto' : undefined;
      control = <NumberInput value={value} kind={param.kind} min={param.min} max={param.max} onChange={onChange} ariaLabel={label} empty={empty} />;
    }
    return (
      <>
        {row(
          <>
            {control}
            <ModeMenu param={param} value={value} onChange={onChange} />
          </>,
          problem && <p className="pb-1 text-[10.5px] leading-snug text-bad-400">{problem.charAt(0).toUpperCase() + problem.slice(1)}</p>,
        )}
        {sampler && mode === 'choice' && (
          <Row
            label="Weights"
            dim
            help="How often each option is picked, one weight per option; empty gives equal odds"
            control={
              <>
                <ListInput
                  values={sampler.weights ?? []}
                  kind="float"
                  placeholder="Equal"
                  onChange={weights => onChange(withWeights(sampler, weights))}
                  ariaLabel={`${label} weights`}
                />
                <span className="w-4 flex-shrink-0" aria-hidden="true" />
              </>
            }
          />
        )}
      </>
    );
  }

  if (param.kind === 'float' || param.kind === 'int') {
    return row(<NumberInput value={value} kind={param.kind} min={param.min} max={param.max} onChange={onChange} ariaLabel={label} />);
  }

  if (param.kind === 'bool') {
    return row(
      <label className="h-[18px] inline-flex items-center gap-1.5 text-[11px] text-ink-100 cursor-pointer">
        <input type="checkbox" aria-label={label} checked={value === true} onChange={e => onChange(e.target.checked)} className="w-3 h-3 accent-[rgb(var(--accent-500))]" />
        {value === true ? 'On' : 'Off'}
      </label>,
    );
  }

  if (param.kind === 'choice') {
    const choices = param.choices ?? [];
    const dice = <ModeMenu param={param} value={value} onChange={onChange} />;
    if (isSampler(value)) {
      const problem = samplerProblem(value, param);
      return row(
        <>
          <ChoicePicker param={param} sampler={value} onChange={onChange} />
          {dice}
        </>,
        problem && <p className="pb-1 text-[10.5px] leading-snug text-bad-400">{problem.charAt(0).toUpperCase() + problem.slice(1)}</p>,
      );
    }
    const index = choices.findIndex(c => JSON.stringify(c) === JSON.stringify(value));
    return row(
      <>
      <div className="relative flex-1 min-w-0">
        <select
          aria-label={label}
          title={String(value ?? '')}
          value={index}
          onChange={e => onChange(choices[Number(e.target.value)])}
          className={`${FIELD} appearance-none pr-4 cursor-pointer`}
        >
          {index < 0 && <option value={-1}>{String(value ?? '')}</option>}
          {choices.map((choice, i) => <option key={i} value={i}>{String(choice)}</option>)}
        </select>
        <ChevronDown className="pointer-events-none absolute right-1 top-1/2 -translate-y-1/2 w-3 h-3 text-ink-500" aria-hidden="true" />
      </div>
      {dice}
      </>,
    );
  }

  if (param.kind === 'path') {
    return row(<PathInput value={value} label={label} onChange={onChange} />);
  }

  if (param.kind === 'weights') {
    return row(
      <ListInput values={value} kind="float" placeholder="One per input, e.g. 3, 1" onChange={onChange} ariaLabel={label} />,
    );
  }

  if (param.kind === 'json') {
    return row(
      <Preview text={JSON.stringify(value) ?? ''} open={editorOpen} onToggle={() => setEditorOpen(o => !o)} label={label} />,
      editorOpen && <div className="pb-1"><JsonInput value={value} onChange={onChange} ariaLabel={label} /></div>,
    );
  }

  if (param.kind === 'code') {
    const code = typeof value === 'string' ? value : '';
    const first = code.split('\n').map(line => line.trim()).find(line => line && !line.startsWith('#')) ?? '';
    return row(
      <Preview text={first} open={editorOpen} onToggle={() => setEditorOpen(o => !o)} label={label} />,
      editorOpen && (
        <div className="pb-1">
          <textarea
            aria-label={label}
            autoFocus
            value={code}
            onChange={e => onChange(e.target.value)}
            rows={12}
            spellCheck={false}
            wrap="off"
            className={EDITOR}
          />
        </div>
      ),
    );
  }

  const textValue = typeof value === 'string' ? value : String(value ?? '');
  return row(
    <input type="text" aria-label={label} title={textValue} value={textValue} onChange={e => onChange(e.target.value)} className={FIELD} />,
  );
});

DatasetParamField.displayName = 'DatasetParamField';
