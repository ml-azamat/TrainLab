import * as React from "react";
import { Ban, RefreshCw } from "lucide-react";
import { api, type Json, type RunSummary } from "@/lib/api";
import { cn, fmtDuration, fmtNum } from "@/lib/utils";
import { RunConsole } from "@/components/RunConsole";
import { ResizableSplit } from "@/components/ResizableSplit";
import { Badge, Button, Panel } from "@/components/ui/primitives";

const TONE: Record<string, "ok" | "warn" | "danger" | "accent" | "default"> = {
  running: "accent", starting: "accent", finished: "ok",
  failed: "danger", cancelled: "warn", cancelling: "warn",
};

export function RunsTab({
  activeRunId, setActiveRunId,
}: { activeRunId: string | null; setActiveRunId: (id: string | null) => void }) {
  const [runs, setRuns] = React.useState<RunSummary[]>([]);
  const [active, setActive] = React.useState(0);

  const load = React.useCallback(() => {
    api.runs().then((r) => { setRuns(r.runs); setActive(r.active); }).catch(() => {});
  }, []);

  React.useEffect(() => {
    load();
    // Poll while anything is live; the SSE stream carries per-run detail separately.
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [load]);

  return (
    <ResizableSplit
      storageKey="runs" initial={1 / 2.1} minLeft={320} minRight={400}
      className="h-full min-h-0 overflow-hidden p-3"
    >
      <div className="min-h-0 overflow-y-auto">
        <Panel
          title={
            <span className="flex items-center gap-2">
              This session
              {active > 0 && <Badge variant="accent">{active} running</Badge>}
            </span>
          }
          actions={
            <Button variant="ghost" size="sm" onClick={load}><RefreshCw className="h-3 w-3" /></Button>
          }
        >
          {runs.length === 0 && (
            <div className="py-8 text-center text-[11px] text-[var(--color-muted)]">
              No runs launched from this session yet. Everything ever recorded lives in the
              Compare tab.
            </div>
          )}
          <div className="space-y-1">
            {runs.map((r) => {
              const m = r.latest?.metrics ?? {};
              const primary = r.latest?.primary_metric ?? "acc@1";
              return (
                <button
                  key={r.id}
                  onClick={() => setActiveRunId(r.id)}
                  className={cn(
                    "flex w-full items-center gap-2 rounded border border-transparent px-2 py-1.5 text-left transition-colors hover:bg-[var(--color-panel2)]",
                    activeRunId === r.id && "border-[var(--color-accent)]/40 bg-[var(--color-accent)]/10",
                  )}
                >
                  <Badge variant={TONE[r.status] ?? "default"}>{r.status}</Badge>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-[11px] mono">{r.run_name || r.id}</div>
                    <div className="flex gap-2 text-[10px] text-[var(--color-muted)]">
                      {r.latest?.epoch != null && (
                        <span className="tnum">ep {r.latest.epoch + 1}/{r.latest.total}</span>
                      )}
                      <span className="tnum">{fmtDuration(r.duration_s)}</span>
                      {r.sweep_id && <span>sweep {r.sweep_id.slice(0, 6)}</span>}
                    </div>
                  </div>
                  {r.latest?.best != null && (
                    <span className="shrink-0 text-[11px] tnum text-[var(--color-ok)]">
                      {fmtNum(r.latest.best)}
                    </span>
                  )}
                  {(r.status === "running" || r.status === "starting") && (
                    <span
                      role="button" tabIndex={0}
                      onClick={(e) => { e.stopPropagation(); api.cancelRun(r.id).then(load); }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") { e.stopPropagation(); api.cancelRun(r.id).then(load); }
                      }}
                      className="shrink-0 rounded p-1 text-[var(--color-muted)] hover:text-[var(--color-danger)]"
                      title="Cancel run"
                    >
                      <Ban className="h-3 w-3" />
                    </span>
                  )}
                  {m[primary] != null && (
                    <span className="hidden shrink-0 text-[10px] tnum text-[var(--color-muted)] sm:block">
                      {primary} {fmtNum(m[primary])}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </Panel>
      </div>

      <div className="min-h-0 overflow-y-auto">
        {activeRunId
          ? <RunConsole runId={activeRunId} />
          : (
            <Panel>
              <div className="py-12 text-center text-[11px] text-[var(--color-muted)]">
                Select a run to watch its log, per-epoch metrics and progress.
              </div>
            </Panel>
          )}
      </div>
    </ResizableSplit>
  );
}

export type { Json };
