import { describe, expect, it } from 'vitest';
import type { DatasetParamSpec } from './datasetGraph';
import {
  describeValue,
  formatNumberList,
  isSampler,
  parseNumberList,
  pickOf,
  samplerModeOf,
  samplerModes,
  samplerProblem,
  toSamplerMode,
  withChoices,
  withWeights,
} from './datasetSampler';

const floatParam: DatasetParamSpec = { name: 'scale', kind: 'float', default: 0.5, label: 'Scale', help: '', random: true };
const intParam: DatasetParamSpec = { name: 'radius', kind: 'int', default: 2, label: 'Radius', help: '', random: true, min: 1 };

describe('sampler modes', () => {
  it('tells a sampler from a value, as params.py does', () => {
    expect(isSampler({ uniform: [0, 1] })).toBe(true);
    expect(isSampler({ choice: [1], weights: [1] })).toBe(true);
    expect(isSampler({ h264: {} })).toBe(false);
    expect(isSampler([1, 2])).toBe(false);
    expect(samplerModeOf(0.5)).toBe('fixed');
    expect(samplerModeOf({ int: [1, 3] })).toBe('int');
    expect(samplerModeOf({ choice: [1] })).toBe('choice');
  });

  it('offers int params no float range', () => {
    expect(samplerModes(floatParam)).toEqual(['fixed', 'uniform', 'int', 'choice']);
    expect(samplerModes(intParam)).toEqual(['fixed', 'int', 'choice']);
  });

  it('keeps the numbers when switching mode', () => {
    expect(toSamplerMode(0.5, 'uniform', floatParam)).toEqual({ uniform: [0.5, 0.5] });
    expect(toSamplerMode({ uniform: [0.25, 0.75] }, 'fixed', floatParam)).toBe(0.25);
    expect(toSamplerMode({ uniform: [0.4, 2.6] }, 'int', floatParam)).toEqual({ int: [0, 3] });
    expect(toSamplerMode({ int: [1, 4] }, 'choice', intParam)).toEqual({ choice: [1, 4] });
    expect(toSamplerMode({ choice: [3, 1, 2] }, 'int', intParam)).toEqual({ int: [1, 3] });
    expect(toSamplerMode({ choice: [2.6] }, 'fixed', intParam)).toBe(3);
  });

  it('falls back to the default, then the minimum, when there is no number to keep', () => {
    expect(toSamplerMode('', 'uniform', floatParam)).toEqual({ uniform: [0.5, 0.5] });
    expect(toSamplerMode(null, 'choice', { ...intParam, default: null })).toEqual({ choice: [1] });
  });

  it('leaves a value already in the mode alone', () => {
    const value = { choice: [1, 2], weights: [3, 1] };
    expect(toSamplerMode(value, 'choice', intParam)).toBe(value);
  });
});

describe('choice lists', () => {
  it('read commas or spaces, and int params take whole numbers', () => {
    expect(parseNumberList('1, 2.5  3', 'float')).toEqual({ values: [1, 2.5, 3] });
    expect(parseNumberList('', 'float')).toEqual({ values: [] });
    expect(parseNumberList('1, x', 'float')).toEqual({ error: '"x" is not a number.' });
    expect(parseNumberList('1.5', 'int')).toEqual({ error: '"1.5" is not a whole number.' });
    expect(formatNumberList([1, 2.5])).toBe('1, 2.5');
  });

  it('keep weights only while there is one per option', () => {
    const value = { choice: [1, 2], weights: [3, 1] };
    expect(withChoices(value, [4, 5])).toEqual({ choice: [4, 5], weights: [3, 1] });
    expect(withChoices(value, [4, 5, 6])).toEqual({ choice: [4, 5, 6] });
    expect(withWeights(value, [])).toEqual({ choice: [1, 2] });
    expect(withWeights({ choice: [1, 2] }, [1, 9])).toEqual({ choice: [1, 2], weights: [1, 9] });
  });
});

describe('what the engine would refuse', () => {
  it('passes good samplers', () => {
    expect(samplerProblem({ uniform: [0, 1] }, floatParam)).toBeNull();
    expect(samplerProblem({ int: [1, 1] }, intParam)).toBeNull();
    expect(samplerProblem({ choice: [1, 2], weights: [0, 2] }, intParam)).toBeNull();
  });

  it('names what is wrong', () => {
    expect(samplerProblem({ uniform: [1] }, floatParam)).toBe('a range needs a low and a high number.');
    expect(samplerProblem({ uniform: [2, 1] }, floatParam)).toBe('the low end of the range is above the high end.');
    expect(samplerProblem({ int: [1.5, 3] }, intParam)).toBe('an int range needs whole numbers.');
    expect(samplerProblem({ choice: [] }, intParam)).toBe('a list needs at least one option.');
    expect(samplerProblem({ choice: [1, 2], weights: [1] }, intParam)).toBe('a list needs one weight per option.');
    expect(samplerProblem({ choice: [1, 2], weights: [0, 0] }, intParam)).toBe('at least one weight must be above 0.');
    expect(samplerProblem({ choice: [1], weights: [-1] }, intParam)).toBe('weights must be numbers of 0 or more.');
  });
});

describe('node card summaries', () => {
  it('say a value or sampler in a few characters', () => {
    expect(describeValue(0.123456)).toBe('0.123');
    expect(describeValue({ uniform: [0.75, 1] })).toBe('0.75–1');
    expect(describeValue({ int: [2, 6] })).toBe('int 2–6');
    expect(describeValue({ choice: [1, 2, 3] })).toBe('one of 3');
    expect(describeValue('')).toBe('—');
    expect(describeValue(['a', 'b'])).toBe('2 items');
    expect(describeValue({ h264: {} })).toBe('custom');
  });
});

describe('dropdown picks', () => {
  const dropdown: DatasetParamSpec = {
    name: 'profile', kind: 'choice', default: 'b', label: 'Profile', help: '', random: true, choices: ['a', 'b', 'c'],
  };

  it('only switches between a fixed option and a pick', () => {
    expect(samplerModes(dropdown)).toEqual(['fixed', 'choice']);
    expect(toSamplerMode('b', 'choice', dropdown)).toEqual({ choice: ['a', 'b', 'c'] });
    expect(toSamplerMode({ choice: ['c', 'a'] }, 'fixed', dropdown)).toBe('c');
  });

  it('keeps weights only when they differ', () => {
    expect(pickOf(['a', 'c'], [2, 2])).toEqual({ choice: ['a', 'c'] });
    expect(pickOf(['a', 'c'], [1, 3])).toEqual({ choice: ['a', 'c'], weights: [1, 3] });
  });

  it('refuses options the dropdown does not have', () => {
    expect(samplerProblem({ choice: ['a', 'z'] }, dropdown)).toBe('z is not one of the options.');
    expect(samplerProblem({ choice: ['a', 'c'], weights: [1, 0] }, dropdown)).toBeNull();
  });
});
