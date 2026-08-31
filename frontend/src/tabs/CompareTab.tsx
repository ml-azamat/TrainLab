import * as React from "react";
import { ArrowRight, Copy, GitCompare, RefreshCw } from "lucide-react";
import { api, type Json, type TrackedRun } from "@/lib/api";
import { cn, fmtDuration, fmtNum, metricHigherIsBetter, shortLabel } from "@/lib/utils";
import { ParallelCoordinates } from "@/components/ParallelCoordinates";
import { Badge, Button, Panel, Select, Tooltip } from "@/components/ui/primitives";

/**
 * The reason the app exists: what helped, what didn't.
 *
 * Three linked views over one filtered run set — a table that hides every parameter
 * identical across the selection, a parallel-coordinates plot you can brush, and a diff
 * that shows ONLY what changed between two runs alongside the metric delta.
 */
export function CompareTab({
  onClone, initialExperiment = "default", trackingUri,
}: {
  onClone: (config: Json, droppedFields?: string[]) => void;
  initialExperiment?: string;
  /** The tracker the form points at. Every read below follows it, so changing
   *  `tracking.tracking_uri` redirects these views and not just the next run. */
  trackingUri?: string;
}) {
  // Open on the experiment currently configured, not a hardcoded name — otherwise you
  // land on an empty table every time you use a non-default experiment.
  const [experiment, setExperiment] = React.useState(initialExperiment);
  const [experiments, setExperiments] = React.useState<string[]>([]);
  const [runs, setRuns] = React.useState<TrackedRun[]>([]);
  const [varying, setVarying] = React.useState<string[]>([]);
  const [metrics, setMetrics] = React.useState<string[]>([]);
  const [metric, setMetric] = React.useState("acc_at_1");
  const [selected, setSelected] = React.useState<string[]>([]);
  const [parallel, setParallel] = React.useState<Json | null>(null);
  const [diff, setDiff] = React.useState<Json | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [sortKey, setSortKey] = React.useState<string>("metric");
  const [sortDesc, setSortDesc] = React.useState(true);

  const load = React.useCallback(() => {
    setLoading(true); setError(null);
    api.trackerRuns(experiment, trackingUri)
      .then((r) => {
        setRuns(r.runs);
        setVarying(r.varying_params);
        setMetrics(r.metrics);
        if (r.metrics.length && !r.metrics.includes(metric)) setMetric(r.metrics[0]);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [experiment, trackingUri]); // eslint-disable-line react-hooks/exhaustive-deps

  React.useEffect(() => {
    api.experiments(trackingUri)
      .then((r) => setExperiments(r.experiments.map((e) => e.name)))
      .catch((e) => setError(e.message));
  }, [trackingUri]);

  React.useEffect(load, [load]);

  React.useEffect(() => {
    if (runs.length === 0) return;
    api.parallel(experiment, metric, undefined, trackingUri)
      .then(setParallel).catch(() => setParallel(null));
  }, [experiment, metric, runs.length, trackingUri]);

  React.useEffect(() => {
    if (selected.length === 2) {
      api.diff(selected[0], selected[1], trackingUri)
        .then(setDiff).catch(() => setDiff(null));
    } else setDiff(null);
  }, [selected, trackingUri]);

  const sorted = React.useMemo(() => {
    const arr = [...runs];
    // Missing values sort to the bottom whichever way the metric runs, instead of being
    // pinned to -Infinity and floating to the top of an ascending sort.
    const absent = metricHigherIsBetter(metric) ? -Infinity : Infinity;
    arr.sort((a, b) => {
      const va = sortKey === "metric" ? a.metrics[metric] ?? absent
        : sortKey === "name" ? a.name
        : sortKey === "duration" ? a.duration_s
        : a.params[sortKey] ?? "";
      const vb = sortKey === "metric" ? b.metrics[metric] ?? absent
        : sortKey === "name" ? b.name
        : sortKey === "duration" ? b.duration_s
        : b.params[sortKey] ?? "";
      const na = Number(va), nb = Number(vb);
      const cmp = (Number.isFinite(na) && Number.isFinite(nb))
        ? na - nb : String(va).localeCompare(String(vb));
      return sortDesc ? -cmp : cmp;
    });
    return arr;
  }, [runs, sortKey, sortDesc, metric]);

  // The leaderboard has to know which end of the scale is good. Marking the largest
  // value as best put the green ★ on the WORST run for val_loss, train_loss and ece —
  // all of which are offered in the metric picker.
  const higherIsBetter = metricHigherIsBetter(metric);
  const values = runs.map((r) => r.metrics[metric]).filter((v): v is number => v != null);
  const best = values.length
    ? (higherIsBetter ? Math.max(...values) : Math.min(...values))
    : null;

  const clone = async (runId: string) => {
    try {
      const { config, dropped_fields } = await api.clone(runId, trackingUri);
      onClone(config, dropped_fields);
    } catch (e: any) { setError(e.message); }
  };

  const toggle = (id: string) =>
    setSelected((s) => s.includes(id) ? s.filter((x) => x !== id) : [...s, id].slice(-2));

  if (error) {
    return (
      <Panel title="Compare">
        <div className="py-8 text-center text-[11px]">
          <div className="text-[var(--color-danger)]">{error}</div>
          <div className="mt-2 text-[var(--color-muted)]">
            Start the tracking server with <span className="mono">make up-local</span> (or{" "}
            <span className="mono">make up</span> for the docker-compose stack).
          </div>
          <Button className="mt-3" onClick={load}><RefreshCw className="h-3 w-3" /> Retry</Button>
        </div>
      </Panel>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-3 overflow-y-auto p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">Experiment</span>
        <Select
          className="w-44" value={experiment} onValueChange={setExperiment}
          // Keep the current name in the list even when the tracker has never seen it —
          // otherwise a not-yet-created experiment renders as a blank select.
          options={Array.from(new Set([...experiments, experiment])).map((e) => ({ value: e }))}
        />
        <span className="ml-2 text-[10px] uppercase tracking-wide text-[var(--color-muted)]">Metric</span>
        <Select
          className="w-40" value={metric} onValueChange={setMetric}
          options={(metrics.length ? metrics : ["acc_at_1"]).map((m) => ({ value: m }))}
        />
        <Badge>{runs.length} runs</Badge>
        <Badge variant="accent">{varying.length} params vary</Badge>
        <Button variant="ghost" size="sm" className="ml-auto" onClick={load}>
          <RefreshCw className={cn("h-3 w-3", loading && "spin")} /> Refresh
        </Button>
      </div>

      <Panel title="Hyperparameters vs metric">
        <ParallelCoordinates
          data={parallel} selected={selected} onSelect={setSelected}
          higherIsBetter={higherIsBetter}
        />
      </Panel>

      {selected.length === 2 && diff && <DiffView diff={diff} onClone={clone} />}
      {selected.length === 1 && (
        <div className="rounded-lg border border-dashed border-[var(--color-border)] px-3 py-2 text-[11px] text-[var(--color-muted)]">
          Select a second run to see exactly which parameters differ.
        </div>
      )}

      <Panel
        title="Runs"
        actions={
          <span className="text-[10px] text-[var(--color-muted)]">
            columns show only parameters that vary across these runs
          </span>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-[11px]">
            <thead>
              <tr className="border-b border-[var(--color-border)] text-[10px] text-[var(--color-muted)]">
                <th className="w-6 px-1 py-1"></th>
                <Th onClick={() => { setSortKey("name"); setSortDesc(!sortDesc); }}
                    active={sortKey === "name"} desc={sortDesc} align="left">run</Th>
                <Th onClick={() => { setSortKey("metric"); setSortDesc(!sortDesc); }}
                    active={sortKey === "metric"} desc={sortDesc}>{shortLabel(metric)}</Th>
                {varying.map((p) => (
                  <Th key={p} onClick={() => { setSortKey(p); setSortDesc(!sortDesc); }}
                      active={sortKey === p} desc={sortDesc}>
                    <Tooltip content={p}><span>{shortLabel(p)}</span></Tooltip>
                  </Th>
                ))}
                <Th onClick={() => { setSortKey("duration"); setSortDesc(!sortDesc); }}
                    active={sortKey === "duration"} desc={sortDesc}>time</Th>
                <th className="px-2 py-1"></th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => {
                const v = r.metrics[metric];
                const isBest = v != null && best != null && v === best;
                return (
                  <tr
                    key={r.run_id}
                    onClick={() => toggle(r.run_id)}
                    className={cn(
                      "cursor-pointer border-b border-[var(--color-border)]/40 last:border-0 hover:bg-[var(--color-panel2)]",
                      selected.includes(r.run_id) && "bg-[var(--color-accent)]/10",
                    )}
                  >
                    <td className="px-1 py-1">
                      <span className={cn(
                        "block h-1.5 w-1.5 rounded-full",
                        r.status === "FINISHED" ? "bg-[var(--color-ok)]"
                          : r.status === "FAILED" ? "bg-[var(--color-danger)]"
                          : "bg-[var(--color-muted)]",
                      )} />
                    </td>
                    <td className="max-w-[240px] truncate px-2 py-1 mono">{r.name}</td>
                    <td className={cn("px-2 py-1 text-right tnum font-semibold",
                      isBest && "text-[var(--color-ok)]")}>
                      {fmtNum(v)}{isBest && " ★"}
                    </td>
                    {varying.map((p) => (
                      <td key={p} className="px-2 py-1 text-right tnum text-[var(--color-muted)]">
                        {r.params[p] ?? "—"}
                      </td>
                    ))}
                    <td className="px-2 py-1 text-right tnum text-[var(--color-muted)]">
                      {fmtDuration(r.duration_s)}
                    </td>
                    <td className="px-2 py-1">
                      <Tooltip content="Load this run's config back into the form">
                        <Button
                          variant="ghost" size="sm"
                          onClick={(e) => { e.stopPropagation(); clone(r.run_id); }}
                        >
                          <Copy className="h-3 w-3" />
                        </Button>
                      </Tooltip>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {runs.length === 0 && (
            <div className="py-8 text-center text-[11px] text-[var(--color-muted)]">
              No runs in “{experiment}” yet.
            </div>
          )}
        </div>
      </Panel>
    </div>
  );
}

function Th({
  children, onClick, active, desc, align = "right",
}: {
  children: React.ReactNode; onClick: () => void; active: boolean;
  desc: boolean; align?: "left" | "right";
}) {
  return (
    <th
      onClick={onClick}
      className={cn("cursor-pointer whitespace-nowrap px-2 py-1 font-medium hover:text-[var(--color-fg)]",
        align === "left" ? "text-left" : "text-right",
        active && "text-[var(--color-fg)]")}
    >
      {children}{active && (desc ? " ↓" : " ↑")}
    </th>
  );
}

function DiffView({ diff, onClone }: { diff: Json; onClone: (id: string) => void }) {
  const headline = diff.metrics?.find((m: Json) =>
    m.key === "acc_at_1" && m.delta != null) ?? diff.metrics?.find((m: Json) => m.delta != null);

  return (
    <Panel
      title={<span className="flex items-center gap-1.5"><GitCompare className="h-3 w-3" /> Run diff</span>}
      actions={
        <div className="flex gap-1">
          <Button variant="ghost" size="sm" onClick={() => onClone(diff.a.run_id)}>Clone A</Button>
          <Button variant="ghost" size="sm" onClick={() => onClone(diff.b.run_id)}>Clone B</Button>
        </div>
      }
    >
      {headline && (
        <div className="mb-3 flex items-baseline gap-3 border-b border-[var(--color-border)] pb-2">
          <span className="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">
            Δ {shortLabel(headline.key)}
          </span>
          {/* Green means "B is better than A", which is not the same as "B is larger". */}
          <span className={cn("text-lg font-semibold tnum",
            (metricHigherIsBetter(headline.key) ? headline.delta > 0 : headline.delta < 0)
              ? "text-[var(--color-ok)]" : "text-[var(--color-danger)]")}>
            {headline.delta > 0 ? "+" : ""}{fmtNum(headline.delta)}
          </span>
          <span className="text-[11px] tnum text-[var(--color-muted)]">
            {fmtNum(headline.a)} <ArrowRight className="inline h-3 w-3" /> {fmtNum(headline.b)}
          </span>
        </div>
      )}

      <div className="mb-2 grid grid-cols-2 gap-3 text-[11px]">
        <div className="min-w-0">
          <div className="text-[9px] uppercase text-[var(--color-muted)]">A</div>
          <div className="truncate mono">{diff.a.name}</div>
        </div>
        <div className="min-w-0">
          <div className="text-[9px] uppercase text-[var(--color-muted)]">B</div>
          <div className="truncate mono">{diff.b.name}</div>
        </div>
      </div>

      {diff.confounded && (
        <div className="mb-2 rounded border border-[var(--color-modified)]/30 bg-[var(--color-modified)]/10 px-2 py-1.5 text-[10px] text-[var(--color-modified)]">
          {diff.params.length} parameters differ — this comparison is confounded. You can't
          attribute the metric change to any single one of them.
        </div>
      )}

      <table className="w-full text-[11px]">
        <tbody>
          {diff.params.map((p: Json) => (
            <tr key={p.key} className="border-b border-[var(--color-border)]/40 last:border-0">
              <td className="py-1 pr-2 mono text-[var(--color-muted)]">{p.key}</td>
              <td className="w-32 py-1 text-right tnum">{p.a ?? "—"}</td>
              <td className="w-6 py-1 text-center text-[var(--color-muted)]">→</td>
              <td className="w-32 py-1 tnum text-[var(--color-accent)]">{p.b ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {diff.params.length === 0 && (
        <div className="py-3 text-center text-[11px] text-[var(--color-muted)]">
          These two runs used identical hyperparameters — any metric difference is run-to-run
          variance (or a different seed).
        </div>
      )}

      <div className="mt-2 border-t border-[var(--color-border)] pt-2">
        <div className="mb-1 text-[9px] uppercase text-[var(--color-muted)]">All metric deltas</div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-px sm:grid-cols-3">
          {diff.metrics.filter((m: Json) => m.delta != null).slice(0, 12).map((m: Json) => {
            const better = metricHigherIsBetter(m.key) ? m.delta > 0 : m.delta < 0;
            return (
              <div key={m.key} className="flex items-baseline justify-between text-[10px]">
                <span className="truncate text-[var(--color-muted)]">{shortLabel(m.key)}</span>
                <span className={cn("tnum", m.delta === 0 ? ""
                  : better ? "text-[var(--color-ok)]" : "text-[var(--color-danger)]")}>
                  {m.delta > 0 ? "+" : ""}{fmtNum(m.delta, 3)}
                </span>
              </div>
            );
          })}
        </div>
        <div className="pt-1.5 text-[10px] text-[var(--color-muted)]">
          {diff.identical_params} identical parameters hidden
          {diff.hidden_noise_params > 0 &&
            ` · ${diff.hidden_noise_params} always-differ fields (git commit, paths, timings) excluded`}
        </div>
      </div>
    </Panel>
  );
}
