// src/components/dataset/DatasetWorkspace.tsx — builds paired training
// datasets from video as a graph of nodes, run by the vk_dataset engine.
//
// Left, the node palette, grouped by the stages of a job; middle, the
// canvas, which reads left to right and shades the nodes that run once per
// chunk. Each node card holds its own settings; the explanations live in
// tooltips and behind a card's "?": an edge that
// runs back through the stages is dashed and says why on hover, and the
// problems that stop a run open from their count in the bottom bar. The
// edges of the selected nodes stay bright while the others dim. An edge that
// runs back left, where a long job wraps onto its next band, turns at right
// angles like a carriage return, and so does one that drops to a lower row
// more than a column on. Holding Alt colours the chain from
// green at its start to red at its end. A value wire runs from a value node
// into the socket of one setting and is drawn in the value colour, outside
// the chain heat; while one is drawn, every setting's socket shows. The
// strip on top holds the job file and its seed, the bar at the bottom runs it. With no job open the
// canvas shows the example workflows to start from. Nothing here names a
// node type: the palette, the ports and the settings all come from the
// engine's declarations (datasetGraph.ts).

import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Background,
  BackgroundVariant,
  BaseEdge,
  Controls,
  MiniMap,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  ViewportPortal,
  getBezierPath,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Connection,
  type Edge,
  type EdgeProps,
  type Node,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  AlertCircle, AlertTriangle, CheckCircle, Dices, FilePlus, FolderOpen, LayoutGrid, Loader2, Maximize, Play, RefreshCw, Save, ScanEye, Terminal, XCircle,
} from 'lucide-react';
import { ModalShell } from '../ModalShell';
import { DatasetCanvas, DatasetNodeCard, nodeColor, portColor, type ChainHeat, type PreviewValue, type SettingLink } from './DatasetNodeCard';
import { DatasetPalette, DRAG_TYPE } from './DatasetPalette';
import { DatasetPreviewPanel } from './DatasetPreviewPanel';
import { DatasetStartScreen } from './DatasetStartScreen';
import { useDatasetRun } from '../../hooks/useDatasetRun';
import { ask } from '../../utils/dialogs';
import { notify } from '../../utils/notifications';
import {
  canConnect,
  connectEdge,
  connectedInputs,
  createNode,
  emptyGraph,
  graphToJob,
  jobHasLayout,
  jobToGraph,
  nodeLabel,
  specMap,
  validateGraph,
  type DatasetExample,
  type DatasetJob,
  type DatasetNodeData,
  type GraphIssue,
  type DatasetNodeSpec,
  type DatasetStage,
  type GraphState,
} from '../../utils/datasetGraph';
import { autoConnect, autoLayout, boundsOf, CARD_HEIGHT, CARD_WIDTH, chainDepth, chunkScope, COLUMN, placeNewNode } from '../../utils/datasetLayout';
import { isLink, linkedParam, paramHandle, VALUE_PORT, valueTargets } from '../../utils/datasetLinks';
import { heatColor, stageHints, stageList } from '../../utils/datasetStages';

type CanvasNode = Node<DatasetNodeData>;

const NODE_TYPES = { dataset: DatasetNodeCard };

interface RoutedEdgeData extends Record<string, unknown> {
  /** The stage-order hint, shown as the tooltip. */
  hint?: string;
  /** For an edge that turns at right angles: the height its horizontal run sits at. */
  turnY?: number;
  /** The chain heat colours at its two ends, drawn as a gradient. */
  heat?: [string, string];
}

/** How far a turning edge runs out of its output, and into its input, before it turns. */
const TURN_OFFSET = 24;
const TURN_RADIUS = 8;

/**
 * A path that leaves the output to the right, runs down (or up) the gap
 * after the source column to turnY, across to the gap before the target
 * column, and into the input from the left, with rounded corners.
 */
function turningPath(sx: number, sy: number, tx: number, ty: number, turnY: number): string {
  const points = [[sx, sy], [sx + TURN_OFFSET, sy], [sx + TURN_OFFSET, turnY], [tx - TURN_OFFSET, turnY], [tx - TURN_OFFSET, ty], [tx, ty]]
    .filter((p, i, all) => i === 0 || p[0] !== all[i - 1][0] || p[1] !== all[i - 1][1]);
  let path = `M ${points[0][0]} ${points[0][1]}`;
  for (let i = 1; i < points.length - 1; i++) {
    const [px, py] = points[i - 1];
    const [x, y] = points[i];
    const [nx, ny] = points[i + 1];
    const r = Math.min(TURN_RADIUS, Math.hypot(x - px, y - py) / 2, Math.hypot(nx - x, ny - y) / 2);
    const into = Math.hypot(x - px, y - py) || 1;
    const out = Math.hypot(nx - x, ny - y) || 1;
    path += ` L ${x - ((x - px) / into) * r} ${y - ((y - py) / into) * r}`;
    path += ` Q ${x} ${y} ${x + ((nx - x) / out) * r} ${y + ((ny - y) / out) * r}`;
  }
  const last = points[points.length - 1];
  return `${path} L ${last[0]} ${last[1]}`;
}

/**
 * Every edge: a curve, or a path that turns at right angles and crosses at
 * turnY. A hinted edge carries its hint as the
 * tooltip; in the chain heat view the stroke is a gradient between the
 * colours of its ends.
 */
const RoutedEdge = ({ id, sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition, style, data }: EdgeProps<Edge<RoutedEdgeData>>) => {
  const path = data?.turnY !== undefined
    ? turningPath(sourceX, sourceY, targetX, targetY, data.turnY)
    : getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition })[0];
  const gradient = data?.heat ? `vk-heat-${id.replace(/[^\w-]/g, '_')}` : null;
  return (
    <g>
      {data?.hint && <title>{data.hint}</title>}
      {gradient && (
        <defs>
          <linearGradient id={gradient} gradientUnits="userSpaceOnUse" x1={sourceX} y1={sourceY} x2={targetX} y2={targetY}>
            <stop offset="0" stopColor={data!.heat![0]} />
            <stop offset="1" stopColor={data!.heat![1]} />
          </linearGradient>
        </defs>
      )}
      <BaseEdge path={path} style={gradient ? { ...style, stroke: `url(#${gradient})` } : style} interactionWidth={18} />
    </g>
  );
};

const EDGE_TYPES = { routed: RoutedEdge };

const BTN = 'h-[26px] px-2 rounded-md inline-flex items-center gap-1.5 text-[12px] border border-ink-750 bg-ink-850 text-ink-400 hover:text-ink-200 hover:border-ink-700 transition-colors flex-shrink-0 disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500';
const ACTION = 'h-8 px-3 rounded-md inline-flex items-center gap-2 text-[12.5px] font-semibold border transition-colors flex-shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-500 disabled:cursor-not-allowed';
const CANVAS_BTN = `${BTN} bg-ink-900/90 backdrop-blur-sm shadow-md shadow-black/30`;

/** The smallest zoom a job opens at, so its cards stay readable. */
const READABLE_ZOOM = 0.8;

type SpecsState =
  | { status: 'loading' }
  | { status: 'ready'; list: DatasetNodeSpec[] }
  | { status: 'error'; error: string };

const fileName = (filePath: string) => filePath.split(/[\\/]/).pop() ?? filePath;
const text = (value: unknown) => (typeof value === 'string' ? value.trim() : '');

const DatasetEditor = ({ privacyMode }: { privacyMode: boolean }) => {
  const flow = useReactFlow<CanvasNode, Edge>();
  const canvasRef = useRef<HTMLDivElement>(null);
  const [specsState, setSpecsState] = useState<SpecsState>({ status: 'loading' });
  const [stages, setStages] = useState<DatasetStage[]>([]);
  const [examples, setExamples] = useState<DatasetExample[]>([]);
  const [nodes, setNodes, onNodesChange] = useNodesState<CanvasNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [meta, setMeta] = useState<Pick<GraphState, 'seed' | 'settings' | 'extra'>>(() => emptyGraph());
  const [jobPath, setJobPath] = useState<string | null>(null);
  const [jobOpen, setJobOpen] = useState(false);
  const [showStart, setShowStart] = useState(false);
  const [savedSnapshot, setSavedSnapshot] = useState(() => JSON.stringify(graphToJob(emptyGraph())));
  const [showLog, setShowLog] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [showProblems, setShowProblems] = useState(false);
  const [previewValues, setPreviewValues] = useState<Map<string, PreviewValue> | null>(null);
  // Whether a value wire is being drawn, which shows every setting's socket.
  const [linking, setLinking] = useState(false);
  // A job with no saved positions is arranged once its cards have been measured.
  const arrangePending = useRef(false);
  const runner = useDatasetRun();
  const busy = runner.status !== 'idle';

  const loadSpecs = useCallback(async () => {
    setSpecsState({ status: 'loading' });
    const outcome = await window.electronAPI.datasetGetNodes();
    setSpecsState(outcome.success ? { status: 'ready', list: outcome.value } : { status: 'error', error: outcome.error });
  }, []);

  useEffect(() => {
    void loadSpecs();
    void window.electronAPI.datasetGetStages()
      .then(outcome => setStages(outcome.success ? outcome.value : []))
      .catch(() => setStages([]));
    void window.electronAPI.datasetGetPresets().then(setExamples).catch(() => setExamples([]));
  }, [loadSpecs]);

  const specList = useMemo(() => (specsState.status === 'ready' ? specsState.list : []), [specsState]);
  const specs = useMemo(() => specMap(specList), [specList]);
  const stageRows = useMemo(() => stageList(stages, specList), [stages, specList]);

  const state: GraphState = useMemo(() => ({ ...meta, nodes, edges }), [meta, nodes, edges]);
  const job = useMemo(() => graphToJob(state), [state]);
  const snapshot = useMemo(() => JSON.stringify(job), [job]);
  const dirty = snapshot !== savedSnapshot;
  const issues = useMemo(
    () => (specsState.status === 'ready' ? validateGraph(state, specs, jobPath !== null) : []),
    [specsState.status, state, specs, jobPath],
  );
  const hints = useMemo(() => (specsState.status === 'ready' ? stageHints(state, specs) : []), [specsState.status, state, specs]);
  const flagged = useMemo(() => new Set(issues.flatMap(i => (i.nodeId ? [i.nodeId] : []))), [issues]);
  const hinted = useMemo(() => new Set(hints.map(h => h.nodeId)), [hints]);
  const connected = useMemo(() => connectedInputs(edges), [edges]);
  const links = useMemo(() => {
    const labels = new Map(nodes.map(n => [n.id, nodeLabel(n.data, specs.get(n.data.type))]));
    const byNode = new Map<string, Map<string, SettingLink>>();
    for (const edge of edges) {
      const setting = linkedParam(edge.targetHandle);
      if (setting === null) continue;
      const own = byNode.get(edge.target) ?? new Map<string, SettingLink>();
      own.set(setting, { source: edge.source, label: labels.get(edge.source) ?? edge.source });
      byNode.set(edge.target, own);
    }
    return byNode;
  }, [nodes, edges, specs]);
  const targets = useMemo(() => valueTargets(nodes, edges, specs), [nodes, edges, specs]);
  const hintByEdge = useMemo(() => new Map(hints.map(h => [h.edgeId, h.message])), [hints]);
  const perChunk = useMemo(() => chunkScope(nodes, edges, specs), [nodes, edges, specs]);
  const selected = nodes.filter(n => n.selected);
  const selectedNode = selected.length === 1 ? selected[0] : null;
  const jobName = text(meta.extra.name);
  const jobDescription = text(meta.extra.description);
  const shownIssues = jobOpen && nodes.length > 0 ? issues : [];
  const jobTip = [jobName, jobDescription].filter(Boolean).join('\n');
  const issuesByNode = useMemo(() => {
    const byNode = new Map<string, GraphIssue[]>();
    for (const issue of shownIssues) {
      if (issue.nodeId) byNode.set(issue.nodeId, [...(byNode.get(issue.nodeId) ?? []), issue]);
    }
    return byNode;
  }, [shownIssues]);

  const heights = useMemo(() => new Map(nodes.map(n => [n.id, n.measured?.height])), [nodes]);
  const heightOf = useCallback((id: string) => heights.get(id), [heights]);
  const region = useMemo(() => boundsOf(nodes, perChunk, heightOf, 28, 30), [nodes, perChunk, heightOf]);

  const updateNode = useCallback((id: string, update: (data: DatasetNodeData) => DatasetNodeData) => {
    setNodes(current => current.map(n => (n.id === id ? { ...n, data: update(n.data) } : n)));
  }, [setNodes]);

  const removeNode = useCallback((id: string) => {
    setNodes(current => current.filter(n => n.id !== id));
    setEdges(current => current.filter(e => e.source !== id && e.target !== id));
  }, [setNodes, setEdges]);

  // While Alt is held, the chain heat view shows how far along its chain
  // each node and edge is, from green at the start to red at the end, and
  // the edges move in the direction the video flows.
  const [showFlow, setShowFlow] = useState(false);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Alt') {
        event.preventDefault();
        setShowFlow(event.type === 'keydown');
      }
    };
    const onBlur = () => setShowFlow(false);
    window.addEventListener('keydown', onKey);
    window.addEventListener('keyup', onKey);
    window.addEventListener('blur', onBlur);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('keyup', onKey);
      window.removeEventListener('blur', onBlur);
    };
  }, []);
  const heat: ChainHeat | null = useMemo(() => {
    if (!showFlow) return null;
    const depths = chainDepth(nodes, edges, specs);
    return { depths, deepest: Math.max(1, ...depths.values()) };
  }, [showFlow, nodes, edges, specs]);

  const canvasContext = useMemo(() => ({
    specs,
    flagged,
    hinted,
    connected,
    issues: issuesByNode,
    heat,
    links,
    targets,
    values: showPreview ? previewValues : null,
    onToggleEnabled: (id: string) => updateNode(id, data => ({ ...data, enabled: !data.enabled })),
    onLabelChange: (id: string, label: string) => updateNode(id, data => {
      const { label: _old, ...rest } = data;
      void _old;
      return label ? { ...rest, label } : rest;
    }),
    onNoteChange: (id: string, note: string) => updateNode(id, data => {
      const { note: _old, ...extra } = data.extra;
      void _old;
      return { ...data, extra: note ? { ...extra, note } : extra };
    }),
    onParamChange: (id: string, name: string, value: unknown) => updateNode(id, data => ({ ...data, params: { ...data.params, [name]: value } })),
    onUnlink: (id: string, name: string) => setEdges(current => current.filter(e => !(e.target === id && e.targetHandle === paramHandle(name)))),
    onRemove: removeNode,
  }), [specs, flagged, hinted, connected, issuesByNode, heat, links, targets, showPreview, previewValues, updateNode, setEdges, removeNode]);

  const fitAll = useCallback(() => {
    requestAnimationFrame(() => { void flow.fitView({ padding: 0.15, maxZoom: 1, duration: 200 }); });
  }, [flow]);

  /**
   * Shows the whole job when it fits at a zoom its cards can be read at;
   * otherwise shows its start, left aligned at that zoom, for reading to the right.
   */
  const showJob = useCallback(() => {
    requestAnimationFrame(() => {
      void flow.fitView({ padding: 0.15, maxZoom: 1 }).then(() => {
        const rect = canvasRef.current?.getBoundingClientRect();
        const shown = flow.getNodes();
        if (!rect || shown.length === 0 || flow.getZoom() >= READABLE_ZOOM) return;
        const bounds = flow.getNodesBounds(shown);
        const zoom = READABLE_ZOOM;
        const top = Math.max(64, (rect.height - bounds.height * zoom) / 2);
        void flow.setViewport({ x: 48 - bounds.x * zoom, y: top - bounds.y * zoom, zoom });
      });
    });
  }, [flow]);

  /**
   * Lays the graph out left to right by dependency, HR path above LR path,
   * using the cards' measured heights. `asOpened` counts the layout as part
   * of the job as it was opened rather than as an unsaved edit.
   */
  const arrange = useCallback((asOpened = false) => {
    const layout = autoLayout(nodes, edges, { specs, heightOf });
    const placed = nodes.map(n => ({ ...n, position: layout.get(n.id) ?? n.position }));
    setNodes(placed);
    if (asOpened) setSavedSnapshot(JSON.stringify(graphToJob({ ...meta, nodes: placed, edges })));
    showJob();
  }, [nodes, edges, specs, heightOf, meta, setNodes, showJob]);

  useEffect(() => {
    if (!arrangePending.current || nodes.length === 0 || !nodes.every(n => n.measured?.height)) return;
    arrangePending.current = false;
    arrange(true);
  }, [nodes, arrange]);

  /**
   * Puts a job on the canvas; `path` is the file it came from, or null for a
   * new job. An example is arranged to the cards as they are drawn now; a
   * saved job keeps its layout, or is arranged when it has none.
   */
  const loadJob = useCallback((source: DatasetJob, path: string | null) => {
    const graph = jobToGraph(source);
    setMeta({ seed: graph.seed, settings: graph.settings, extra: graph.extra });
    setNodes(graph.nodes as CanvasNode[]);
    setEdges(graph.edges as Edge[]);
    setJobPath(path);
    setJobOpen(true);
    setSavedSnapshot(JSON.stringify(graphToJob(graph)));
    arrangePending.current = graph.nodes.length > 0 && (path === null || !jobHasLayout(source));
    showJob();
  }, [showJob, setNodes, setEdges]);

  const confirmDiscard = useCallback(async () => !dirty || ask({
    title: 'Discard changes',
    message: 'This job has changes that are not saved. Discard them?',
    confirmLabel: 'Discard',
    tone: 'danger',
  }), [dirty]);

  const handleNew = useCallback(async (example: DatasetJob | null) => {
    if (!await confirmDiscard()) return;
    setShowStart(false);
    loadJob(example ?? graphToJob(emptyGraph()), null);
  }, [confirmDiscard, loadJob]);

  const handleOpen = useCallback(async () => {
    if (!await confirmDiscard()) return;
    const opened = await window.electronAPI.datasetOpenJob();
    if (!opened) return;
    if (!opened.success) {
      notify.error('Could not open the job', opened.error);
      return;
    }
    setShowStart(false);
    loadJob(opened.job, opened.filePath);
  }, [confirmDiscard, loadJob]);

  const handleSave = useCallback(async (saveAs: boolean) => {
    const saved = await window.electronAPI.datasetSaveJob({ job, filePath: saveAs ? null : jobPath });
    if (!saved) return;
    if (!saved.success) {
      notify.error('Could not save the job', saved.error);
      return;
    }
    setJobPath(saved.filePath);
    setJobOpen(true);
    setSavedSnapshot(snapshot);
  }, [job, jobPath, snapshot]);

  /** Pans to a card when it is not fully in view. */
  const reveal = useCallback((at: { x: number; y: number }) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const topLeft = flow.flowToScreenPosition(at);
    const bottomRight = flow.flowToScreenPosition({ x: at.x + CARD_WIDTH, y: at.y + CARD_HEIGHT });
    if (topLeft.x >= rect.left && topLeft.y >= rect.top && bottomRight.x <= rect.right && bottomRight.y <= rect.bottom) return;
    void flow.setCenter(at.x + CARD_WIDTH / 2, at.y + CARD_HEIGHT / 2, { zoom: flow.getZoom(), duration: 250 });
  }, [flow]);

  /**
   * Adds a node: where it was dropped, or right of the selected node and
   * joined to it when there is only one way to join them, or with nothing
   * selected, after the nodes of earlier stages.
   */
  const addNode = useCallback((spec: DatasetNodeSpec, position?: { x: number; y: number }) => {
    const from = position ? null : selectedNode;
    const at = position ?? placeNewNode(spec, nodes, specs, from?.id ?? null, heightOf);
    const node = createNode(spec, at, nodes.map(n => n.id)) as CanvasNode;
    const join = from ? autoConnect(spec, from, edges, specs) : null;
    setNodes(current => [...current.map(n => ({ ...n, selected: false })), { ...node, selected: true }]);
    if (from && join) {
      setEdges(current => connectEdge(current, { source: from.id, target: node.id, ...join }) as Edge[]);
    }
    setJobOpen(true);
    if (!position) reveal(at);
  }, [selectedNode, nodes, edges, specs, heightOf, setNodes, setEdges, reveal]);

  const onDrop = useCallback((event: React.DragEvent) => {
    const type = event.dataTransfer.getData(DRAG_TYPE);
    const spec = specs.get(type);
    if (!spec) return;
    event.preventDefault();
    const at = flow.screenToFlowPosition({ x: event.clientX, y: event.clientY });
    addNode(spec, { x: at.x - CARD_WIDTH / 2, y: at.y - 20 });
  }, [addNode, flow, specs]);

  const isValidConnection = useCallback(
    (connection: Connection | Edge) => canConnect(connection, nodes, edges, specs),
    [nodes, edges, specs],
  );
  const onConnect = useCallback(
    (connection: Connection) => setEdges(current => connectEdge(current, connection) as Edge[]),
    [setEdges],
  );
  const onConnectStart = useCallback((_event: unknown, from: { nodeId: string | null; handleId: string | null; handleType: 'source' | 'target' | null }) => {
    const node = nodes.find(n => n.id === from.nodeId);
    const out = from.handleType === 'source' && node ? specs.get(node.data.type)?.outputs[from.handleId ?? ''] : undefined;
    setLinking(out === VALUE_PORT || (from.handleType === 'target' && linkedParam(from.handleId) !== null));
  }, [nodes, specs]);
  const onConnectEnd = useCallback(() => setLinking(false), []);


  const styledEdges = useMemo(() => {
    const byId = new Map(nodes.map(n => [n.id, n]));
    const anySelected = nodes.some(n => n.selected);
    const heatAt = (id: string) => heatColor((heat?.depths.get(id) ?? 0) / (heat?.deepest ?? 1));
    return edges.map(edge => {
      const source = byId.get(edge.source);
      const target = byId.get(edge.target);
      const type = source ? specs.get(source.data.type)?.outputs[edge.sourceHandle ?? ''] : undefined;
      const lit = edge.selected || !!source?.selected || !!target?.selected;
      if (isLink(edge)) {
        // A value wire is a plain curve in the value colour, and stays out of the chain heat.
        return {
          ...edge,
          type: 'routed',
          data: {},
          style: {
            stroke: portColor(VALUE_PORT),
            strokeWidth: lit ? 2.5 : 1.5,
            ...((heat || (anySelected && !lit)) ? { strokeOpacity: 0.25 } : {}),
          },
        };
      }
      const hint = hintByEdge.get(edge.id);
      const backwards = hint !== undefined;
      // An input left of the output it reads from, or in a lower row more than
      // a column on: the edge turns at right angles, crossing above the target card.
      // A wrap crosses a little higher than a drop, so the two never share a line.
      const wraps = !!source && !!target && target.position.x < source.position.x + CARD_WIDTH;
      const drops = !!source && !!target && !wraps
        && target.position.y > source.position.y + (source.measured?.height ?? CARD_HEIGHT)
        && target.position.x - source.position.x > COLUMN + 1;
      const data: RoutedEdgeData = {
        ...(backwards ? { hint } : {}),
        ...(wraps || drops ? { turnY: target!.position.y - (wraps ? 26 : 14) } : {}),
        ...(heat ? { heat: [heatAt(edge.source), heatAt(edge.target)] as [string, string] } : {}),
      };
      return {
        ...edge,
        type: 'routed',
        data,
        animated: heat !== null,
        style: {
          stroke: backwards ? '#e0b341' : portColor(type),
          strokeWidth: heat ? 3 : lit ? 3 : 2,
          ...(!heat && anySelected && !lit ? { strokeOpacity: 0.2 } : {}),
          ...(backwards ? { strokeDasharray: '6 4' } : {}),
        },
      };
    });
  }, [edges, nodes, specs, hintByEdge, heat]);

  const selectNode = useCallback((id: string) => {
    setNodes(current => current.map(n => ({ ...n, selected: n.id === id })));
    const node = flow.getNode(id);
    if (node) void flow.setCenter(node.position.x + CARD_WIDTH / 2, node.position.y + 80, { zoom: flow.getZoom(), duration: 200 });
  }, [flow, setNodes]);

  const handleRun = useCallback(async () => {
    if (issues.length > 0) {
      setShowProblems(true);
      notify.warning('The job cannot run yet', issues[0].message + (issues.length > 1 ? ` (and ${issues.length - 1} more)` : ''));
      return;
    }
    setShowLog(true);
    await runner.run(job, jobPath);
  }, [issues, job, jobPath, runner]);

  const handleCheck = useCallback(async () => {
    if (issues.length > 0) {
      setShowProblems(true);
      runner.append(`Check failed: ${issues[0].message}`, 'bad');
      return;
    }
    await runner.check(job, jobPath);
  }, [issues, job, jobPath, runner]);

  const percent = runner.progress === null ? null : Math.round(runner.progress * 100);
  const status = runner.status === 'running'
    ? { label: percent === null ? 'Running' : `Running ${percent}%`, tone: 'border-accent-500/45 bg-accent-500/12 text-accent-400' }
    : runner.status === 'stopping'
      ? { label: 'Stopping', tone: 'border-warn-500/40 bg-warn-500/12 text-warn-400' }
      : runner.status === 'checking'
        ? { label: 'Checking', tone: 'border-ink-750 bg-ink-850 text-ink-300' }
        : !jobOpen || nodes.length === 0
          ? { label: jobOpen ? 'Empty job' : 'No job open', tone: 'border-ink-750 bg-ink-850 text-ink-500' }
          : issues.length > 0
            ? { label: `${issues.length} problem${issues.length === 1 ? '' : 's'}`, tone: 'border-bad-500/40 bg-bad-500/10 text-bad-400 hover:bg-bad-500/15' }
            : { label: 'Ready', tone: 'border-ink-750 bg-ink-850 text-ink-400' };
  const problemsShown = showProblems && shownIssues.length > 0 && runner.status === 'idle';

  const ready = specsState.status === 'ready';
  const startShown = ready && !jobOpen;

  return (
    <div className="flex-1 flex flex-col min-w-0 min-h-0">
      {/* Job strip */}
      <div className="h-11 flex-shrink-0 flex items-center gap-3 px-4 bg-ink-900 border-b border-ink-800">
        <span className="font-display text-[13.5px] font-semibold uppercase tracking-[0.16em] text-ink-200 select-none whitespace-nowrap">
          Dataset
        </span>
        <span className="w-px h-5 bg-ink-800 flex-shrink-0" />
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <button type="button" onClick={() => setShowStart(true)} disabled={busy} title="Start a new job from an example or empty" className={BTN}>
            <FilePlus className="w-3.5 h-3.5" />
            New
          </button>
          <button type="button" onClick={() => void handleOpen()} disabled={busy} title="Open a .vkdataset job" className={BTN}>
            <FolderOpen className="w-3.5 h-3.5" />
            Open
          </button>
          <button type="button" onClick={() => void handleSave(false)} disabled={!jobOpen} title={jobPath ? `Save to ${jobPath}` : 'Save the job'} className={BTN}>
            <Save className="w-3.5 h-3.5" />
            Save
          </button>
          <button type="button" onClick={() => void handleSave(true)} disabled={!jobOpen} title="Save the job to a new file" className={BTN}>
            Save as
          </button>
        </div>
        <span className="text-[12px] text-ink-400 truncate min-w-0" title={privacyMode ? undefined : [jobPath, jobTip].filter(Boolean).join('\n\n') || undefined}>
          {!jobOpen ? 'No job open' : jobPath ? (privacyMode ? 'Saved job' : fileName(jobPath)) : 'Unsaved job'}{dirty ? ' •' : ''}
        </span>
        <div className="flex-1" />
        <label className="flex items-center gap-1.5 flex-shrink-0" title="Every random setting is drawn from this seed; the same seed gives the same dataset">
          <span className="font-display text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-500">Seed</span>
          <input
            type="number"
            step={1}
            value={meta.seed}
            onChange={e => {
              const seed = Number(e.target.value);
              if (Number.isInteger(seed)) setMeta(m => ({ ...m, seed }));
            }}
            className="w-[110px] h-[26px] bg-ink-850 border border-ink-750 rounded px-2 text-[12.5px] tabular-nums text-ink-200 focus:outline-none focus:border-accent-500"
          />
        </label>
        <button
          type="button"
          onClick={() => setMeta(m => ({ ...m, seed: Math.floor(Math.random() * 2 ** 31) }))}
          title="Pick a new random seed"
          aria-label="Pick a new random seed"
          className={`${BTN} px-1.5`}
        >
          <Dices className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="flex-1 flex min-h-0">
        {/* Palette */}
        <div className="w-[196px] flex-shrink-0 bg-ink-950 border-r border-ink-800 min-h-0">
          {specsState.status === 'loading' && (
            <div className="px-3 py-3 flex items-center gap-2 text-[12px] text-ink-400">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Loading nodes
            </div>
          )}
          {specsState.status === 'error' && (
            <div className="px-3 py-3 space-y-2">
              <div className="flex items-start gap-2 text-[12px] text-bad-400">
                <AlertCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                <span>The dataset engine did not load: {specsState.error}</span>
              </div>
              <button type="button" onClick={() => void loadSpecs()} className={BTN}>
                <RefreshCw className="w-3.5 h-3.5" />
                Try again
              </button>
            </div>
          )}
          {ready && (
            <DatasetPalette
              specs={specList}
              stages={stageRows}
              onAdd={spec => addNode(spec)}
            />
          )}
        </div>

        {/* Canvas, and the preview under it */}
        <div className="flex-1 min-w-0 flex flex-col">
        <div
          ref={canvasRef}
          className="flex-1 min-w-0 min-h-0 relative"
          onDragOver={event => { event.preventDefault(); event.dataTransfer.dropEffect = 'copy'; }}
          onDrop={onDrop}
        >
          <DatasetCanvas.Provider value={canvasContext}>
            <ReactFlow
              className={`vk-dataset-flow ${linking ? 'vk-linking' : ''}`}
              colorMode="dark"
              nodes={nodes}
              edges={styledEdges}
              nodeTypes={NODE_TYPES}
              edgeTypes={EDGE_TYPES}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onConnectStart={onConnectStart}
              onConnectEnd={onConnectEnd}
              isValidConnection={isValidConnection}
              deleteKeyCode={['Backspace', 'Delete']}
              minZoom={0.15}
              maxZoom={2}
              proOptions={{ hideAttribution: true }}
            >
              <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
              {region && (
                <ViewportPortal>
                  <div
                    className="vk-chunk-region absolute rounded-xl border border-dashed"
                    style={{ transform: `translate(${region.x}px, ${region.y}px)`, width: region.width, height: region.height }}
                  >
                    <span
                      className="vk-chunk-label inline-block mx-3 mt-2 text-[11px] font-medium cursor-help"
                      title="These nodes run once per chunk, and draw their random settings again for every chunk."
                    >
                      Per chunk
                    </span>
                  </div>
                </ViewportPortal>
              )}
              <Controls showInteractive={false} />
              {nodes.length > 0 && (
                <MiniMap
                  pannable
                  zoomable
                  nodeColor={node => nodeColor(specs.get((node.data as DatasetNodeData).type))}
                  nodeStrokeWidth={0}
                  nodeBorderRadius={6}
                  ariaLabel="Overview of the job"
                  style={{ width: 132, height: 84 }}
                />
              )}
              {nodes.length > 0 && (
                <Panel position="top-right" className="!m-2 flex items-center gap-1.5">
                  <span
                    title="Hold Alt to colour the chain by step, from green at its start to red at its end"
                    className="h-[26px] px-1.5 inline-flex items-center gap-1 text-[11px] text-ink-500 cursor-help"
                  >
                    <kbd className="px-1 rounded border border-ink-700 bg-ink-850 font-sans text-[10px] text-ink-400">Alt</kbd>
                    steps
                  </span>
                  <button type="button" onClick={() => arrange()} title="Lay out left to right by stage (hold Alt to see the steps)" className={CANVAS_BTN}>
                    <LayoutGrid className="w-3.5 h-3.5" />
                    Arrange
                  </button>
                  <button type="button" onClick={fitAll} title="Fit the whole job in view" className={CANVAS_BTN}>
                    <Maximize className="w-3.5 h-3.5" />
                    Fit
                  </button>
                </Panel>
              )}
            </ReactFlow>
          </DatasetCanvas.Provider>
          {jobOpen && nodes.length === 0 && ready && (
            <div className="absolute inset-0 grid place-items-center pointer-events-none">
              <p className="text-[12.5px] text-ink-500">Add nodes from the left, starting with 1 Load</p>
            </div>
          )}
          {startShown && (
            <div className="absolute inset-0 overflow-y-auto bg-ink-950 px-6 py-12">
              <DatasetStartScreen examples={examples} onPick={example => void handleNew(example)} onOpen={() => void handleOpen()} />
            </div>
          )}
        </div>
        {showPreview && ready && (
          <DatasetPreviewPanel
            job={job}
            jobPath={jobPath}
            nodes={nodes}
            specs={specs}
            selectedNodeId={selectedNode?.id ?? null}
            privacyMode={privacyMode}
            onAdoptSeed={seed => setMeta(m => ({ ...m, seed }))}
            onValues={setPreviewValues}
            onClose={() => setShowPreview(false)}
          />
        )}
        </div>
      </div>

      {showStart && (
        <ModalShell
          title="New job"
          icon={FilePlus}
          onClose={() => setShowStart(false)}
          closeOnBackdrop
          className="w-[min(960px,100%)] max-h-full flex flex-col"
        >
          <div className="flex-1 overflow-y-auto p-5">
            <DatasetStartScreen examples={examples} onPick={example => void handleNew(example)} onOpen={() => void handleOpen()} />
          </div>
        </ModalShell>
      )}

      {/* Run log */}
      {showLog && (
        <div className="h-[180px] flex-shrink-0 flex flex-col bg-ink-900 border-t border-ink-800">
          <div className="h-7 flex-shrink-0 flex items-center gap-2 px-3 border-b border-ink-800">
            <span className="font-display text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-500">Log</span>
            <div className="flex-1" />
            <button type="button" onClick={runner.clearLog} className="text-[11px] text-ink-500 hover:text-ink-200">Clear</button>
          </div>
          {privacyMode
            ? <div className="flex-1 px-3 py-1.5 text-[12px] text-ink-500">Details hidden by privacy mode</div>
            : <LogView lines={runner.log} />}
        </div>
      )}

      {/* Run bar */}
      <div className="h-[52px] flex-shrink-0 relative flex items-center gap-3 px-4 bg-ink-900 border-t border-ink-800">
        {runner.status === 'running' && percent !== null && (
          <div className="absolute top-0 left-0 h-[2px] bg-accent-500 transition-all duration-300" style={{ width: `${percent}%` }} />
        )}
        {runner.status === 'idle' && shownIssues.length > 0 ? (
          <button
            type="button"
            onClick={() => setShowProblems(shown => !shown)}
            aria-expanded={problemsShown}
            className={`inline-flex items-center gap-1.5 h-[26px] px-2.5 rounded-md border text-[12px] font-medium flex-shrink-0 transition-colors ${status.tone}`}
          >
            <AlertTriangle className="w-3.5 h-3.5" />
            {status.label}
          </button>
        ) : (
          <span className={`inline-flex items-center gap-2 h-[26px] px-2.5 rounded-md border text-[12px] font-medium flex-shrink-0 ${status.tone}`}>
            {(runner.status === 'running' || runner.status === 'checking' || runner.status === 'stopping') && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
            {status.label}
          </span>
        )}
        {problemsShown && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setShowProblems(false)} aria-hidden="true" />
            <ul className="absolute left-4 bottom-[calc(100%-6px)] z-50 w-[360px] max-h-[320px] overflow-y-auto py-1 rounded-md border border-ink-750 bg-ink-900 shadow-xl shadow-black/40">
              {shownIssues.map((issue, i) => (
                <li key={i}>
                  <button
                    type="button"
                    disabled={!issue.nodeId}
                    onClick={() => { setShowProblems(false); if (issue.nodeId) selectNode(issue.nodeId); }}
                    className="w-full px-3 py-1.5 flex items-start gap-2 text-left text-[12px] leading-snug text-ink-300 enabled:hover:bg-ink-850 transition-colors"
                  >
                    <AlertTriangle className="w-3.5 h-3.5 mt-[1px] flex-shrink-0 text-bad-400" />
                    {issue.message}
                  </button>
                </li>
              ))}
            </ul>
          </>
        )}
        {!privacyMode && <span className="text-[12px] text-ink-400 truncate min-w-0" title={runner.message}>{runner.message}</span>}
        <div className="flex-1" />
        <button
          type="button"
          onClick={() => setShowPreview(s => !s)}
          aria-pressed={showPreview}
          disabled={specsState.status !== 'ready'}
          title="Show a node's output for one chunk"
          className={`${ACTION} disabled:opacity-40 ${showPreview
            ? 'bg-accent-500/15 border-accent-500/40 text-accent-400'
            : 'bg-ink-850 border-ink-750 text-ink-300 hover:bg-ink-800 hover:border-ink-700'}`}
        >
          <ScanEye className="w-4 h-4" />
          Preview
        </button>
        <button
          type="button"
          onClick={() => setShowLog(s => !s)}
          aria-pressed={showLog}
          className={`${ACTION} ${showLog
            ? 'bg-accent-500/15 border-accent-500/40 text-accent-400'
            : 'bg-ink-850 border-ink-750 text-ink-300 hover:bg-ink-800 hover:border-ink-700'}`}
        >
          <Terminal className="w-4 h-4" />
          Log
        </button>
        <button
          type="button"
          onClick={() => void handleCheck()}
          disabled={busy || specsState.status !== 'ready'}
          title="Check whether the job would run"
          className={`${ACTION} bg-ink-850 border-ink-750 text-ink-300 hover:bg-ink-800 hover:border-ink-700 disabled:opacity-40`}
        >
          <CheckCircle className="w-4 h-4" />
          Check
        </button>
        <button
          type="button"
          onClick={() => void (runner.status === 'running' ? runner.cancel() : handleRun())}
          disabled={runner.status === 'checking' || runner.status === 'stopping' || specsState.status !== 'ready' || nodes.length === 0}
          className={runner.status === 'running'
            ? `${ACTION} bg-bad-600 border-bad-600 text-white hover:bg-bad-500`
            : `${ACTION} bg-accent-500 border-accent-500 text-accent-ink hover:bg-accent-400 disabled:bg-ink-800 disabled:border-ink-750 disabled:text-ink-600`}
        >
          {runner.status === 'running' ? <><XCircle className="w-4 h-4" />Stop</> : <><Play className="w-4 h-4" />Run</>}
        </button>
      </div>
    </div>
  );
};

/** The run log, kept scrolled to the newest line unless scrolled up. */
const LogView = ({ lines }: { lines: { id: number; text: string; tone: 'info' | 'ok' | 'bad' }[] }) => {
  const ref = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (el && pinned.current) el.scrollTop = el.scrollHeight;
  }, [lines]);
  return (
    <div
      ref={ref}
      onScroll={e => {
        const el = e.currentTarget;
        pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 8;
      }}
      className="flex-1 overflow-y-auto px-3 py-1.5 font-mono text-[11.5px] leading-relaxed select-text"
    >
      {lines.length === 0 && <div className="text-ink-600">Nothing yet.</div>}
      {lines.map(line => (
        <div key={line.id} className={line.tone === 'bad' ? 'text-bad-400' : line.tone === 'ok' ? 'text-ok-400' : 'text-ink-300'}>
          {line.text}
        </div>
      ))}
    </div>
  );
};

export const DatasetWorkspace = memo(({ privacyMode }: { privacyMode: boolean }) => (
  <ReactFlowProvider>
    <DatasetEditor privacyMode={privacyMode} />
  </ReactFlowProvider>
));

DatasetWorkspace.displayName = 'DatasetWorkspace';
