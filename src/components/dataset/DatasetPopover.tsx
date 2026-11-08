// src/components/dataset/DatasetPopover.tsx — a small floating panel under a
// button on a node card, such as a node's "?" help or a setting's sampler
// menu. It closes on an outside click or Escape; a click on its own button
// (marked data-popover-button, in the same parent) is left to that button,
// so it can toggle the panel.

import { useEffect, useRef, type ReactNode } from 'react';

export const DatasetPopover = ({ onClose, children, className = 'right-1 top-full mt-1' }: {
  onClose: () => void;
  children: ReactNode;
  /** Where the panel sits against its positioned parent. */
  className?: string;
}) => {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const away = (event: MouseEvent) => {
      const target = event.target as Element;
      const own = target.closest?.('[data-popover-button]');
      if (own && ref.current?.parentElement?.contains(own)) return;
      if (ref.current && !ref.current.contains(target)) onClose();
    };
    const key = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    document.addEventListener('mousedown', away);
    document.addEventListener('keydown', key);
    return () => {
      document.removeEventListener('mousedown', away);
      document.removeEventListener('keydown', key);
    };
  }, [onClose]);
  return (
    <div ref={ref} className={`nodrag nowheel nopan cursor-auto absolute z-50 rounded-md border border-ink-700 bg-ink-900 shadow-lg shadow-black/50 ${className}`}>
      {children}
    </div>
  );
};
