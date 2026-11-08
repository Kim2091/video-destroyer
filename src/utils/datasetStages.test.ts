import { describe, expect, it } from 'vitest';
import { jobToGraph, specMap, type DatasetStage } from './datasetGraph';
import { NODE_SPECS } from './datasetSpecs.fixture';
import { heatColor, matchesSearch, paletteSections, stageColor, stageHints, stageList, stagesOf } from './datasetStages';

const specs = specMap(NODE_SPECS);
const STAGES: DatasetStage[] = ['Load', 'Prepare', 'Split', 'Degrade', 'Clean up', 'Pick', 'Filter', 'Save']
  .map((name, i) => ({ stage: i + 1, name, help: `${name} help` }));

describe('stages', () => {
  it('give each stage its own colour, and a node its own and its also stages', () => {
    expect(new Set(STAGES.map(s => stageColor(s.stage))).size).toBe(8);
    expect(stageColor(undefined)).toContain('--ink-');
    expect(stagesOf(specs.get('resize'))).toEqual([4, 2]);
    expect(stagesOf(undefined)).toEqual([]);
  });

  it('colour a chain from green at its start to red at its end', () => {
    expect(heatColor(0)).toBe('hsl(130 80% 52%)');
    expect(heatColor(0.5)).toBe('hsl(65 80% 52%)');
    expect(heatColor(1)).toBe('hsl(0 80% 52%)');
    expect(heatColor(2)).toBe(heatColor(1));
  });

  it('come from the node types when the engine list is missing', () => {
    const named = stageList([], NODE_SPECS);
    expect(named.map(s => s.name)).toEqual(['Load', 'Prepare', 'Split', 'Degrade', 'Clean up', 'Pick', 'Filter', 'Save']);
  });
});

describe('palette', () => {
  it('groups node types by stage, in stage order, with the stage sentence', () => {
    const sections = paletteSections(NODE_SPECS, STAGES);
    expect(sections.map(s => s.stage.stage)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(sections[0].stage.help).toBe('Load help');
    expect(sections[0].specs.map(s => s.type).sort()).toEqual(['load_dataset', 'load_video']);
  });

  it('searches labels, descriptions and stage names, every word', () => {
    expect(matchesSearch(specs.get('resize')!, 'downscale')).toBe(true);
    expect(matchesSearch(specs.get('resize')!, 'SHRINKS filter')).toBe(true);
    expect(matchesSearch(specs.get('resize')!, 'shrinks tape')).toBe(false);
    const found = paletteSections(NODE_SPECS, STAGES, 'compress');
    expect(found.flatMap(s => s.specs.map(n => n.type))).toContain('codec');
    expect(paletteSections(NODE_SPECS, STAGES, 'degrade').map(s => s.stage.name)).toEqual(['Degrade']);
    expect(paletteSections(NODE_SPECS, STAGES, 'zzz')).toEqual([]);
  });

  it('puts node types without a stage under Other', () => {
    const odd = { ...NODE_SPECS[0], type: 'odd', label: 'Odd', stage: null };
    expect(paletteSections([odd], STAGES).map(s => s.stage.name)).toEqual(['Other']);
  });
});

describe('stage hints', () => {
  const job = (edges: [string, string][]) => jobToGraph({
    nodes: [
      { id: 'src', type: 'load_video' },
      { id: 'chunks', type: 'chunk' },
      { id: 'seq', type: 'make_sequences' },
      { id: 'blur', type: 'blur' },
      { id: 'down', type: 'resize' },
      { id: 'crop', type: 'crop_rescale' },
      { id: 'out', type: 'write_dataset' },
    ],
    edges: edges.map(([from, to]) => ({ from, to })),
  });

  it('say nothing for a job in stage order', () => {
    expect(stageHints(job([['src.clip', 'chunks.clip'], ['chunks.clip', 'blur.clip'], ['blur.clip', 'seq.lr']]), specs)).toEqual([]);
  });

  it('count a node\'s also stages: Downscale may prepare a source before it is split', () => {
    expect(stageHints(job([['src.clip', 'down.clip'], ['down.clip', 'chunks.clip']]), specs)).toEqual([]);
  });

  it('point at an edge that runs back through the stages, and skip edges that join no ports', () => {
    const hints = stageHints(job([['chunks.clip', 'blur.clip'], ['out.x', 'down.clip'], ['blur.clip', 'crop.clip']]), specs);
    expect(hints).toHaveLength(1);
    expect(hints[0]).toMatchObject({ nodeId: 'crop', edgeId: 'blur.clip->crop.clip' });
    expect(hints[0].message).toBe('Crop and Rescale is a Prepare step but comes after Blur (Degrade). '
      + 'Jobs usually work best in stage order: try moving Crop and Rescale before Blur.');
  });
});
