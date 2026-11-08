import { describe, expect, it } from 'vitest';
import { specMap, type DatasetParamSpec } from './datasetGraph';
import { NODE_SPECS } from './datasetSpecs.fixture';
import { cardSettings, formatSetting, nodeChance, summarizeSettings } from './datasetSummary';

const specs = specMap(NODE_SPECS);
const param = (name: string, kind = 'float'): DatasetParamSpec =>
  ({ name, kind, default: null, label: name, help: '', random: true });
const texts = (type: string, params: Record<string, unknown> = {}, privacy = false) =>
  summarizeSettings(specs.get(type)!, params, 3, privacy).map(s => `${s.label}: ${s.text}`);

describe('setting summaries', () => {
  it('show random settings as ranges and choices', () => {
    expect(formatSetting(param('sigma'), { uniform: [0.7, 2] })).toBe('0.7–2');
    expect(formatSetting(param('steps', 'int'), { int: [3, 3] })).toBe('3');
    expect(formatSetting(param('scale'), { choice: [0.5, 0.25] })).toBe('0.5 / 0.25');
    expect(formatSetting(param('scale'), { choice: [0.5, 0.25], weights: [3, 1] })).toBe('0.5 (75%) / 0.25 (25%)');
  });

  it('show lists, ranges and weighted tables in a few words', () => {
    expect(formatSetting(param('kernels', 'json'), ['bicubic', 'area', 'lanczos'])).toBe('bicubic / area / lanczos');
    expect(formatSetting(param('kernels', 'json'), ['a', 'b', 'c', 'd', 'e', 'f'])).toBe('a / b / c / d +2');
    expect(formatSetting(param('opacity_range', 'json'), [0.05, 0.15])).toBe('0.05–0.15');
    expect(formatSetting(param('codecs', 'json'), {
      h264: { probability: 0.7, quality_range: [22, 32] },
      h265: { probability: 0.3 },
      vp9: { probability: 0, quality_range: [1, 2] },
    })).toBe('h264 22–32 (70%) · h265 (30%)');
    expect(formatSetting(param('folders', 'json'), { hr: 'HR', lr: 'LR' })).toBe('hr HR · lr LR');
    expect(formatSetting(param('on', 'bool'), true)).toBe('on');
  });

  it('name a node\'s file, or say it is not set, and hide it in privacy mode', () => {
    expect(texts('load_video')).toEqual(['Path: not set']);
    expect(summarizeSettings(specs.get('load_video')!, {})[0].missing).toBe(true);
    expect(texts('load_video', { path: 'D:\\clips\\film.mkv' })).toEqual(['Path: film.mkv']);
    expect(texts('load_video', { path: 'D:\\clips\\film.mkv' }, true)).toEqual(['Path: set']);
  });

  it('say a code setting is edited instead of showing the code', () => {
    expect(formatSetting(param('code', 'code'), 'clip = core.std.Invert(clip)')).toBe('edited');
    expect(formatSetting(param('code', 'code'), '')).toBeNull();
  });

  it('prefer changed and random settings, shown in declaration order, without the probability', () => {
    expect(texts('resize', { scale: 0.25 })).toEqual([
      'Scale: 0.25', 'Kernels: bicubic / bilinear / area', 'Down up scale: 0.75–1',
    ]);
    expect(texts('codec')).toEqual(['Codecs: h264 23–33 (60%) · mpeg2 6000–10000 (40%)']);
    expect(texts('write_dataset').length).toBe(3);
    // An optional folder left empty is not worth a line.
    expect(texts('write_dataset').some(t => t.startsWith('Bad directory'))).toBe(false);
  });

  it('give how often a node runs, only when it may be skipped', () => {
    expect(nodeChance(specs.get('blur'), {})).toBeNull();
    expect(nodeChance(specs.get('blur'), { probability: 0.3 })).toBe(0.3);
    expect(nodeChance(specs.get('make_sequences'), { probability: 0.3 })).toBeNull();
  });
});

describe('card settings', () => {
  it('put files and how often first, then keep the declared order', () => {
    const spec = specs.get('resize')!;
    const names = cardSettings(spec).map(p => p.name);
    expect(names[0]).toBe('probability');
    expect(names.slice(1)).toEqual(spec.params.map(p => p.name).filter(n => n !== 'probability'));
    expect(cardSettings(specs.get('load_video')!)[0].kind).toBe('path');
  });
});
