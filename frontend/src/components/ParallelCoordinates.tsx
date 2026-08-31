import * as React from "react";
import type { Json } from "@/lib/api";
import { cn, shortLabel } from "@/lib/utils";

/**
 * Parallel-coordinates plot of hyperparameters against the primary metric.
 *
 * Hand-rolled SVG rather than a charting library: brushing has to drive selection in the
 * run table and the diff view, and owning the scales keeps that coupling simple.
 * Numeric axes optionally use a log scale (auto-detected server-side for lr / wd).
 */

type Axis = {
  key: string;
  type: "number" | "category" | "metric";
  min?: number;
  max?: number;
  log?: boolean;
  categories?: string[];
};
type Line = { run_id: string; name: string; metric: number; values: Record<string, string> };
type Brush = { lo: number; hi: number };

// Generous horizontal margins: the first and last axis labels are centred on their axis,
// so a narrow margin clips them against the panel edge.
const M = { top: 34, right: 64, bottom: 34, left: 64 };
const H = 300;

/** Fractional height of a category on its axis. Shared by lines and tick labels so the
 *  two cannot disagree — a single category used to draw its line at 0 and its label at
 *  0.5, because the guard for it was unreachable behind a `Math.max(1, …)`. */
export function categoryPos(categories: string[], index: number): number {
  if (categories.length <= 1) return 0.5;
  return index / (categories.length - 1);
}

function pos(axis: Axis, raw: string | number | undefined): number | null {
  if (raw == null || raw === "") return null;
  if (axis.type === "category") {
    const cats = axis.categories ?? [];
    const i = cats.indexOf(String(raw));
    if (i < 0) return null;
    return categoryPos(cats, i);
  }
  const v = Number(raw);
  if (!Number.isFinite(v)) return null;
  const lo = axis.min ?? 0, hi = axis.max ?? 1;
  if (hi === lo) return 0.5;
  if (axis.log && lo > 0 && v > 0) {
    return (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo));
  }
  // Clamp: a value outside [lo,hi] (or a non-positive value on an axis labelled "log")
  // would otherwise be drawn outside the plot area entirely.
  return Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
}

function colorFor(t: number): string {
  // Low metric -> muted blue, high metric -> green. Perceptually monotonic enough
  // to read rank without a legend.
  const c = Math.max(0, Math.min(1, t));
  const h = 220 + (145 - 220) * c;
  const l = 42 + 18 * c;
  return `hsl(${h} 70% ${l}%)`;
}

export function ParallelCoordinates({
  data, selected, onSelect, height = H, higherIsBetter = true,
}: {
  data: Json | null;
  selected: string[];
  onSelect: (runIds: string[]) => void;
  height?: number;
  higherIsBetter?: boolean;
}) {
  const axes: Axis[] = data?.axes ?? [];
  const lines: Line[] = data?.lines ?? [];
  const [width, setWidth] = React.useState(720);
  const [brushes, setBrushes] = React.useState<Record<string, Brush>>({});
  const [hover, setHover] = React.useState<string | null>(null);
  const ref = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  React.useEffect(() => setBrushes({}), [data?.metric, axes.length]);

  const innerW = Math.max(120, width - M.left - M.right);
  const step = axes.length > 1 ? innerW / (axes.length - 1) : 0;
  const rotateLabels = axes.length > 1 && step < 55;
  // Rotated labels need vertical room; grow the header rather than overlapping the plot.
  const labelPad = rotateLabels ? 46 : 0;
  const topPad = M.top + labelPad;
  const svgHeight = height + labelPad;
  const innerH = height - M.top - M.bottom;
  // A single axis is centred rather than pinned to the left margin.
  const x = (i: number) => (axes.length > 1 ? M.left + i * step : M.left + innerW / 2);
  const y = (t: number) => topPad + (1 - t) * innerH;

  const metricAxis = axes.find((a) => a.type === "metric");
  const mLo = metricAxis?.min ?? 0, mHi = metricAxis?.max ?? 1;

  const passes = React.useCallback((l: Line) => {
    for (const [key, b] of Object.entries(brushes)) {
      const axis = axes.find((a) => a.key === key);
      if (!axis) continue;
      const raw = axis.type === "metric" ? l.metric : l.values[key];
      const p = pos(axis, raw as any);
      if (p == null || p < b.lo || p > b.hi) return false;
    }
    return true;
  }, [brushes, axes]);

  const visible = lines.filter(passes);

  // Clearing the last brush must clear the selection too. Early-returning on an empty
  // brush set left the previous filtered selection stuck in the table and the diff view,
  // so "double-click to clear" cleared the visual brush and nothing else.
  const brushed = React.useRef(false);
  React.useEffect(() => {
    const active = Object.keys(brushes).length > 0;
    if (active) {
      brushed.current = true;
      onSelect(visible.map((l) => l.run_id));
    } else if (brushed.current) {
      brushed.current = false;
      onSelect([]);
    }
  }, [brushes]); // eslint-disable-line react-hooks/exhaustive-deps

  // A missing value skips its axis, so the command letter has to be decided AFTER
  // filtering: emitting "M" only for index 0 meant a run missing the first axis produced
  // a path starting with "L", which SVG rejects outright — the whole line silently
  // disappeared from the plot rather than showing a gap.
  const path = (l: Line) =>
    axes
      .map((a, i) => {
        const raw = a.type === "metric" ? l.metric : l.values[a.key];
        const p = pos(a, raw as any);
        return p == null ? null : `${x(i)},${y(p)}`;
      })
      .filter((pt): pt is string => pt !== null)
      .map((pt, i) => `${i === 0 ? "M" : "L"}${pt}`)
      .join(" ");

  if (axes.length === 0 || lines.length === 0) {
    return (
      <div ref={ref} className="flex h-[200px] items-center justify-center text-[11px] text-[var(--color-muted)]">
        Not enough runs yet — launch two runs that differ in one parameter and they'll
        appear here.
      </div>
    );
  }

  return (
    <div ref={ref} className="w-full">
      <svg width={width} height={svgHeight} className="overflow-visible select-none">
        {axes.map((a, i) => {
          const b = brushes[a.key];
          return (
            <g key={a.key}>
              <line x1={x(i)} y1={topPad} x2={x(i)} y2={topPad + innerH}
                    stroke="var(--color-border)" strokeWidth={1} />

              {/* Brush target: drag vertically on an axis to filter. */}
              <rect
                x={x(i) - 9} y={topPad} width={18} height={innerH}
                fill="transparent" style={{ cursor: "ns-resize" }}
                onMouseDown={(e) => {
                  const svg = (e.target as SVGRectElement).ownerSVGElement!;
                  const start = e.clientY - svg.getBoundingClientRect().top;
                  const move = (ev: MouseEvent) => {
                    const cur = ev.clientY - svg.getBoundingClientRect().top;
                    const t1 = 1 - (Math.min(start, cur) - topPad) / innerH;
                    const t0 = 1 - (Math.max(start, cur) - topPad) / innerH;
                    setBrushes((prev) => ({
                      ...prev,
                      [a.key]: { lo: Math.max(0, t0), hi: Math.min(1, t1) },
                    }));
                  };
                  const up = () => {
                    window.removeEventListener("mousemove", move);
                    window.removeEventListener("mouseup", up);
                  };
                  window.addEventListener("mousemove", move);
                  window.addEventListener("mouseup", up);
                }}
                onDoubleClick={() => setBrushes((p) => {
                  const n = { ...p }; delete n[a.key]; return n;
                })}
              />

              {b && (
                <rect
                  x={x(i) - 7} y={y(b.hi)} width={14} height={Math.max(2, y(b.lo) - y(b.hi))}
                  fill="var(--color-accent)" fillOpacity={0.18}
                  stroke="var(--color-accent)" strokeOpacity={0.5} rx={2}
                  style={{ pointerEvents: "none" }}
                />
              )}

              {/* Below ~55px of axis spacing horizontal labels overlap into an
                  unreadable smear (24 varying params across a 1100px panel gave each
                  axis ~46px). Rotating keeps every label legible at any axis count. */}
              <text
                x={x(i)} y={topPad - (rotateLabels ? 8 : 12)}
                textAnchor={rotateLabels ? "start" : "middle"}
                transform={rotateLabels ? `rotate(-55 ${x(i)} ${topPad - 8})` : undefined}
                className="fill-[var(--color-fg)] text-[9px]"
                style={{ fontWeight: a.type === "metric" ? 700 : 400 }}
              >
                <title>{a.key}</title>
                {shortLabel(a.key)}
              </text>
              {a.type !== "category" && (
                <>
                  <text x={x(i)} y={topPad - 3} textAnchor="middle"
                        className="fill-[var(--color-muted)] text-[8px] tnum">
                    {fmtAxis(a.max)}
                  </text>
                  <text x={x(i)} y={topPad + innerH + 10} textAnchor="middle"
                        className="fill-[var(--color-muted)] text-[8px] tnum">
                    {fmtAxis(a.min)}
                  </text>
                </>
              )}
              {a.type === "category" && (a.categories ?? []).map((c, ci) => (
                <text key={c} x={x(i) + 6} y={y(categoryPos(a.categories!, ci)) + 3}
                      className="fill-[var(--color-muted)] text-[8px]">
                  {c.length > 14 ? c.slice(0, 13) + "…" : c}
                </text>
              ))}
              {a.log && (
                <text x={x(i)} y={topPad + innerH + 20} textAnchor="middle"
                      className="fill-[var(--color-muted)] text-[7px]">log</text>
              )}
            </g>
          );
        })}

        {lines.map((l) => {
          const on = passes(l);
          const isSel = selected.includes(l.run_id);
          const isHover = hover === l.run_id;
          // Colour encodes quality, not magnitude: on a lower-is-better metric the scale
          // must invert, or the worst runs are the green ones.
          const frac = mHi === mLo ? 1 : (l.metric - mLo) / (mHi - mLo);
          const t = higherIsBetter ? frac : 1 - frac;
          return (
            <path
              key={l.run_id} d={path(l)} fill="none"
              stroke={on ? colorFor(t) : "var(--color-border)"}
              strokeWidth={isHover || isSel ? 2.5 : 1.4}
              strokeOpacity={on ? (isHover || isSel ? 1 : 0.75) : 0.15}
              style={{ cursor: "pointer", transition: "stroke-width 100ms" }}
              onMouseEnter={() => setHover(l.run_id)}
              onMouseLeave={() => setHover(null)}
              onClick={() => onSelect(
                selected.includes(l.run_id)
                  ? selected.filter((r) => r !== l.run_id)
                  : [...selected, l.run_id].slice(-2),
              )}
            >
              <title>{`${l.name}\n${data?.metric} = ${l.metric.toFixed(4)}`}</title>
            </path>
          );
        })}
      </svg>

      <div className="flex items-center justify-between px-1 pt-1 text-[10px] text-[var(--color-muted)]">
        <span>
          {visible.length} of {lines.length} runs
          {Object.keys(brushes).length > 0 && (
            <button className="ml-2 text-[var(--color-accent)] hover:underline"
                    onClick={() => { setBrushes({}); onSelect([]); }}>
              clear filters
            </button>
          )}
        </span>
        <span className={cn(hover ? "text-[var(--color-fg)]" : "")}>
          {hover ? lines.find((l) => l.run_id === hover)?.name
                 : "drag an axis to filter · double-click to clear · click a line to select"}
        </span>
      </div>
    </div>
  );
}

function fmtAxis(v?: number): string {
  if (v == null) return "";
  if (v !== 0 && (Math.abs(v) < 0.001 || Math.abs(v) >= 1e5)) return v.toExponential(0);
  return String(+v.toFixed(3));
}
