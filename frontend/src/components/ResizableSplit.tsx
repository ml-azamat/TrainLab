import * as React from "react";
import { cn } from "@/lib/utils";

/**
 * Two panes with a draggable divider between them.
 *
 * Owned here rather than pulled from a panel library, for the reason the primitives file
 * gives: the behaviour is a few pointer events, and the layout it has to cooperate with
 * (independent scroll containers, a single-column fallback below `lg`) is this app's, not
 * a library's.
 *
 * The position is a fraction of the container width, clamped so neither pane can be
 * dragged under its minimum — the right rails genuinely break below theirs, since the
 * augmentation preview and the warning rows have intrinsic widths. It is remembered per
 * `storageKey`, because a layout you have to re-drag every visit is worse than a fixed one.
 */

/** Where a drag at `x` puts the divider, as a fraction of the container. */
export function splitFraction(
  x: number, rect: { left: number; width: number },
  minLeft: number, minRight: number,
): number {
  const { left, width } = rect;
  if (width <= 0) return 0.5;
  // A container too narrow for both minimums has no valid position; splitting the
  // difference keeps the divider inside the box instead of inverting the panes.
  if (minLeft + minRight >= width) return minLeft / (minLeft + minRight);
  const frac = (x - left) / width;
  return Math.min(Math.max(frac, minLeft / width), 1 - minRight / width);
}

/**
 * The strip the two `fr` tracks actually divide: the container minus its padding and the
 * handle sitting between them.
 *
 * Measuring the padded border box instead is off by exactly those pixels, so a pane
 * clamped to its "minimum" comes out a little under it — enough to break the layout the
 * minimum exists to protect.
 */
function trackRect(container: HTMLElement, handle: Element): { left: number; width: number } {
  const box = container.getBoundingClientRect();
  const cs = getComputedStyle(container);
  const padLeft = parseFloat(cs.paddingLeft) || 0;
  const padRight = parseFloat(cs.paddingRight) || 0;
  return {
    left: box.left + padLeft,
    width: box.width - padLeft - padRight - handle.getBoundingClientRect().width,
  };
}

/** A stored position, or `fallback` when there is nothing usable to restore. */
export function storedFraction(raw: string | null, fallback: number): number {
  const v = Number(raw);
  return Number.isFinite(v) && v > 0.05 && v < 0.95 ? v : fallback;
}

const KEY_STEP = 0.02;

export function ResizableSplit({
  storageKey, initial, minLeft = 320, minRight = 340, className, children,
}: {
  storageKey: string;
  /** Left pane's share of the width, 0..1, before anything is dragged. */
  initial: number;
  minLeft?: number;
  minRight?: number;
  className?: string;
  children: [React.ReactNode, React.ReactNode];
}) {
  const ref = React.useRef<HTMLDivElement>(null);
  const [frac, setFrac] = React.useState(() => {
    try {
      return storedFraction(localStorage.getItem(`split:${storageKey}`), initial);
    } catch {
      return initial;            // storage disabled (private mode, embedded webview)
    }
  });
  const [dragging, setDragging] = React.useState(false);

  // Both the drag flag and the position are read inside pointer handlers that can fire
  // before React re-renders — a `pointermove` arriving in the same tick as its
  // `pointerdown` would see `dragging === false` and be dropped, losing the first (and
  // for a flick, only) movement. State drives the styling; refs drive the behaviour.
  const draggingRef = React.useRef(false);
  const fracRef = React.useRef(frac);

  const persist = React.useCallback(() => {
    try { localStorage.setItem(`split:${storageKey}`, String(fracRef.current)); }
    catch { /* storage disabled — the layout just does not survive a reload */ }
  }, [storageKey]);

  const setFraction = (v: number) => {
    fracRef.current = v;
    setFrac(v);
  };

  const handleRef = React.useRef<HTMLDivElement>(null);

  const move = (clientX: number) => {
    const el = ref.current, handle = handleRef.current;
    if (!el || !handle) return;
    setFraction(splitFraction(clientX, trackRect(el, handle), minLeft, minRight));
  };

  // Dragging over the log pane or a canvas must not lose the pointer, and the text under
  // the cursor must not select — both are what make a hand-rolled splitter feel broken.
  React.useEffect(() => {
    if (!dragging) return;
    const prev = document.body.style.userSelect;
    const prevCursor = document.body.style.cursor;
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
    return () => {
      document.body.style.userSelect = prev;
      document.body.style.cursor = prevCursor;
    };
  }, [dragging]);

  const nudge = (delta: number) => {
    const el = ref.current, handle = handleRef.current;
    if (!el || !handle) return;
    const rect = trackRect(el, handle);
    move(rect.left + (fracRef.current + delta) * rect.width);
  };

  return (
    <div
      ref={ref}
      // The variable is only read at `lg` and above, where the third (handle) column
      // exists. Below it the single-column class wins and the handle is display:none, so
      // a dragged width cannot leak into the stacked layout.
      className={cn("grid grid-cols-1 gap-3 lg:gap-0 lg:grid-cols-[var(--split-cols)]", className)}
      style={{ ["--split-cols" as any]: `minmax(0,${frac}fr) auto minmax(0,${1 - frac}fr)` }}
    >
      {children[0]}
      <div
        ref={handleRef}
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize panes"
        aria-valuenow={Math.round(frac * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        tabIndex={0}
        onPointerDown={(e) => {
          // Capture keeps the drag alive when the cursor leaves the 12px handle, which is
          // immediately. It throws for a pointer the browser does not consider active, so
          // a failure here must not take the drag down with it.
          try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* not fatal */ }
          draggingRef.current = true;
          setDragging(true);
        }}
        onPointerMove={(e) => { if (draggingRef.current) move(e.clientX); }}
        onPointerUp={(e) => {
          try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* see above */ }
          draggingRef.current = false;
          setDragging(false);
          persist();
        }}
        onPointerCancel={() => { draggingRef.current = false; setDragging(false); persist(); }}
        onDoubleClick={() => { setFraction(initial); persist(); }}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") { nudge(-KEY_STEP); e.preventDefault(); }
          else if (e.key === "ArrowRight") { nudge(KEY_STEP); e.preventDefault(); }
          else if (e.key === "Home") { setFraction(initial); persist(); e.preventDefault(); }
        }}
        onBlur={() => persist()}
        title="Drag to resize · double-click to reset"
        className={cn(
          "group hidden lg:flex w-3 shrink-0 cursor-col-resize items-center justify-center",
          "outline-none touch-none select-none",
        )}
      >
        <div className={cn(
          "h-full w-px rounded-full transition-colors",
          dragging ? "bg-[var(--color-accent)]" : "bg-[var(--color-border)]",
          "group-hover:bg-[var(--color-accent)] group-focus-visible:bg-[var(--color-accent)]",
        )} />
      </div>
      {children[1]}
    </div>
  );
}
