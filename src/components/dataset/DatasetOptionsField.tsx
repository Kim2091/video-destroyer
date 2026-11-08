// src/components/dataset/DatasetOptionsField.tsx — the options a value node
// picks from (kind "options"). Wired into a dropdown, they are ticked from
// that dropdown's choices with a weight each, in the same picker a random
// dropdown uses; otherwise they are typed as a list, with an optional
// weights list under it.

import { useState } from 'react';
import { ChoicePicker, Row } from './DatasetParamField';
import type { DatasetParamSpec } from '../../utils/datasetGraph';
import { formatOptionList, parseOptionList, readOptions, writeOptions } from '../../utils/datasetOptions';

const FIELD = 'h-[18px] w-full min-w-0 px-1.5 rounded-[3px] bg-ink-950/60 border border-ink-700/70 text-[11px] text-ink-100 placeholder-ink-600 transition-colors focus:outline-none focus:border-accent-500 hover:border-ink-600';

/**
 * A text box for a list that keeps what is typed and reports each edit. It
 * takes the list's text only when that changed from elsewhere (an undo, a
 * tick), never because of its own edit, which would reformat it while typing.
 */
const ListBox = ({ text, placeholder, label, onText }: { text: string; placeholder: string; label: string; onText: (text: string) => void }) => {
  const [draft, setDraft] = useState(text);
  const [seen, setSeen] = useState(text);
  if (text !== seen) {
    setSeen(text);
    if (formatOptionList(parseOptionList(draft)) !== text) setDraft(text);
  }
  return (
    <input
      type="text"
      aria-label={label}
      title={placeholder}
      placeholder={placeholder}
      value={draft}
      onChange={e => { setDraft(e.target.value); onText(e.target.value); }}
      className={FIELD}
    />
  );
};

export const DatasetOptionsField = ({ param, value, target, onChange }: {
  param: DatasetParamSpec;
  value: unknown;
  /** The setting the node's value goes into, whose choices are ticked when it is a dropdown. */
  target?: DatasetParamSpec;
  onChange: (value: unknown) => void;
}) => {
  const options = readOptions(value);
  const help = param.help || param.name;

  if (target?.kind === 'choice' && target.choices?.length) {
    return (
      <Row
        label={param.label}
        help={help}
        control={
          options.choice.length
            ? <ChoicePicker param={{ ...target, label: param.label }} sampler={writeOptions(options)} onChange={onChange} />
            : (
              <button
                type="button"
                onClick={() => onChange(writeOptions({ choice: [...target.choices!], weights: null }))}
                title={`Pick from ${target.label}'s options`}
                className={`${FIELD} text-left text-ink-400 hover:text-ink-100`}
              >
                Pick from {target.choices.length} options
              </button>
            )
        }
      />
    );
  }

  return (
    <>
      <Row
        label={param.label}
        help={help}
        control={
          <ListBox
            text={formatOptionList(options.choice)}
            placeholder="Options, e.g. a, b, 0.5"
            label={param.label}
            onText={text => {
              const choice = parseOptionList(text);
              onChange(writeOptions({ choice, weights: options.weights?.length === choice.length ? options.weights : null }));
            }}
          />
        }
      />
      {options.choice.length > 1 && (
        <Row
          label="Weights"
          dim
          help="How often each option is picked, one weight per option; empty gives equal odds"
          control={
            <ListBox
              text={options.weights ? options.weights.join(', ') : ''}
              placeholder="Equal"
              label={`${param.label} weights`}
              onText={text => {
                const weights = parseOptionList(text);
                if (!weights.length) onChange(writeOptions({ choice: options.choice, weights: null }));
                else if (weights.length === options.choice.length && weights.every(w => typeof w === 'number' && w >= 0)) {
                  onChange(writeOptions({ choice: options.choice, weights: weights as number[] }));
                }
              }}
            />
          }
        />
      )}
    </>
  );
};
