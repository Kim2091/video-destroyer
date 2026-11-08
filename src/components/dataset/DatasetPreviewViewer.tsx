// src/components/dataset/DatasetPreviewViewer.tsx — draws a preview frame,
// optionally beside or under a compare picture, with sequence crops on top.
//
// The compare picture is scaled to the frame's size (nearest by default, so
// an LR picture shows its pixels). Fit scales the picture to the box; 100%
// and 200% are picture pixels per screen pixel, and a picture larger than
// the box is panned by dragging. In wipe mode the handle moves the split:
// the frame is left of it, the compare picture right.

import { memo, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { DatasetPreviewFrame } from '../../../electron/datasetPreviewProtocol';
import {
  clampPan,
  compositeSize,
  rgbToRgba,
  zoomScale,
  type CompareMode,
  type CropOverlay,
  type Size,
  type ZoomMode,
} from '../../utils/datasetPreview';

interface DatasetPreviewViewerProps {
  frame: DatasetPreviewFrame;
  mainLabel: string;
  compareLabel: string | null;
  mode: CompareMode;
  smooth: boolean;
  zoom: ZoomMode;
  overlays: CropOverlay[];
  highlighted: CropOverlay | null;
  dimmed: boolean;
}

function pictureCanvas(data: Uint8Array, width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  canvas.getContext('2d')!.putImageData(new ImageData(rgbToRgba(data, width, height), width, height), 0, 0);
  return canvas;
}

const TAG = 'absolute top-2 px-1.5 py-0.5 rounded bg-black/70 text-[11px] text-ink-200 pointer-events-none max-w-[45%] truncate';

export const DatasetPreviewViewer = memo<DatasetPreviewViewerProps>(({
  frame, mainLabel, compareLabel, mode, smooth, zoom, overlays, highlighted, dimmed,
}: DatasetPreviewViewerProps) => {
  const boxRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [box, setBox] = useState<Size>({ width: 0, height: 0 });
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [wipe, setWipe] = useState(0.5);
  const drag = useRef<{ x: number; y: number; pan: { x: number; y: number } } | null>(null);
  const pictures = useRef<{ main: HTMLCanvasElement; compare: HTMLCanvasElement | null; source: DatasetPreviewFrame } | null>(null);

  const main: Size = { width: frame.width, height: frame.height };
  const hasCompare = Boolean(frame.compare && frame.compareData);
  const size = compositeSize(main, hasCompare, mode);
  const scale = zoomScale(zoom, size, box);
  const scaled = { width: size.width * scale, height: size.height * scale };
  const shownPan = clampPan(pan, scaled, box);

  useLayoutEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      setBox({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => { setPan({ x: 0, y: 0 }); }, [zoom]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (pictures.current?.source !== frame) {
      pictures.current = {
        main: pictureCanvas(frame.data, frame.width, frame.height),
        compare: frame.compare && frame.compareData
          ? pictureCanvas(frame.compareData, frame.compare.width, frame.compare.height)
          : null,
        source: frame,
      };
    }
    const { main: mainPicture, compare } = pictures.current;
    canvas.width = size.width;
    canvas.height = size.height;
    const context = canvas.getContext('2d')!;
    context.imageSmoothingEnabled = smooth;
    context.drawImage(mainPicture, 0, 0);
    if (!compare) return;
    if (mode === 'side') {
      context.drawImage(compare, frame.width, 0, frame.width, frame.height);
    } else {
      const split = Math.round(frame.width * wipe);
      context.save();
      context.beginPath();
      context.rect(split, 0, frame.width - split, frame.height);
      context.clip();
      context.drawImage(compare, 0, 0, frame.width, frame.height);
      context.restore();
    }
  }, [frame, mode, smooth, wipe, size.width, size.height]);

  const onPointerDown = useCallback((event: React.PointerEvent) => {
    if (event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { x: event.clientX, y: event.clientY, pan: shownPan };
  }, [shownPan]);

  const onPointerMove = useCallback((event: React.PointerEvent) => {
    const start = drag.current;
    if (!start) return;
    setPan(clampPan({ x: start.pan.x + event.clientX - start.x, y: start.pan.y + event.clientY - start.y }, scaled, box));
  }, [scaled, box]);

  const onWipeDown = useCallback((event: React.PointerEvent) => {
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
  }, []);

  const onWipeMove = useCallback((event: React.PointerEvent) => {
    if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return;
    setWipe(Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)));
  }, []);

  const pannable = scaled.width > box.width + 1 || scaled.height > box.height + 1;
  const placed = {
    width: scaled.width,
    height: scaled.height,
    left: (box.width - scaled.width) / 2 + shownPan.x,
    top: (box.height - scaled.height) / 2 + shownPan.y,
  };

  return (
    <div
      ref={boxRef}
      className={`relative flex-1 min-w-0 min-h-0 overflow-hidden bg-ink-950 select-none ${pannable ? 'cursor-grab active:cursor-grabbing' : ''}`}
      onPointerDown={pannable ? onPointerDown : undefined}
      onPointerMove={pannable ? onPointerMove : undefined}
      onPointerUp={() => { drag.current = null; }}
    >
      <div className={`absolute transition-opacity ${dimmed ? 'opacity-60' : ''}`} style={placed}>
        <canvas
          ref={canvasRef}
          className="block w-full h-full"
          style={{ imageRendering: scale >= 1 && !smooth ? 'pixelated' : 'auto' }}
        />
        {overlays.length > 0 && (
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox={`0 0 ${size.width} ${size.height}`}
            preserveAspectRatio="none"
          >
            {overlays.map(overlay => {
              const [x, y, w, h] = overlay.rect;
              const active = highlighted && highlighted.rejected === overlay.rejected && highlighted.index === overlay.index;
              return (
                <rect
                  key={`${overlay.rejected ? 'r' : 'k'}${overlay.index}`}
                  x={x + 0.5} y={y + 0.5} width={Math.max(0, w - 1)} height={Math.max(0, h - 1)}
                  style={{
                    fill: active ? (overlay.rejected ? 'rgb(239 95 95 / 0.18)' : 'rgb(var(--accent-500) / 0.18)') : 'none',
                    stroke: overlay.rejected ? 'rgb(239 95 95)' : 'rgb(var(--accent-500))',
                  }}
                  strokeWidth={active ? 3 : 1.5}
                  strokeDasharray={overlay.rejected ? '6 4' : undefined}
                  vectorEffect="non-scaling-stroke"
                />
              );
            })}
          </svg>
        )}
        {hasCompare && mode === 'wipe' && (
          <div
            role="slider"
            aria-label="Wipe position"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(wipe * 100)}
            tabIndex={0}
            onKeyDown={event => {
              if (event.key === 'ArrowLeft') setWipe(w => Math.max(0, w - 0.05));
              if (event.key === 'ArrowRight') setWipe(w => Math.min(1, w + 0.05));
            }}
            onPointerDown={onWipeDown}
            onPointerMove={onWipeMove}
            className="absolute top-0 bottom-0 w-3 -ml-1.5 cursor-ew-resize flex justify-center focus-visible:outline-none group"
            style={{ left: `${wipe * 100}%` }}
          >
            <span className="w-0.5 h-full bg-accent-400 group-focus-visible:bg-accent-200 shadow-[0_0_4px_rgba(0,0,0,0.8)]" />
          </div>
        )}
      </div>
      <span className={`${TAG} left-2`} title={mainLabel}>{mainLabel}</span>
      {hasCompare && compareLabel && <span className={`${TAG} right-2`} title={compareLabel}>{compareLabel}</span>}
    </div>
  );
});

DatasetPreviewViewer.displayName = 'DatasetPreviewViewer';
