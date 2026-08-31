import * as React from "react";
import { Ban, Terminal } from "lucide-react";
import { api, streamRun, type Json, type RunSummary } from "@/lib/api";
import { cn, fmtDuration, fmtNum } from "@/lib/utils";
import { Badge, Button, Panel } from "./ui/primitives";

const STATUS_TONE: Record<string, "ok" | "warn" | "danger" | "accent" | "default"> = {
  running: "accent", starting: "accent", finished: "ok",
  failed: "danger", cancelled: "warn", cancelling: "warn",
};

export function RunConsole({ runId, onClose }: { runId: string; onClose?: () => void }) {
  const [lines, setLines] = React.useState<string[]>([]);
  const [status, setStatus] = React.useState("starting");
  const [epochs, setEpochs] = React.useState<Json[]>([]);
  const [progress, setProgress] = React.useState<Json | null>(null);
  const [meta, setMeta] = React.useState<RunSummary | null>(null);
  const logRef = React.useRef<HTMLDivElement>(null);
  const [stick, setStick] = React.useState(true);

  React.useEffect(() => {
    setLines([]); setEpochs([]); setProgress(null);
    // Only metadata is prefetched. Lines and epoch events come exclusively from the
    // SSE stream, whose subscribe already replays recent history — fetching log_tail
    // and events here as well rendered the whole startup block (and epoch rows) twice.
    api.run(runId).then((r) => {
      setMeta(r); setStatus(r.status);
    }).catch(() => {});

    return streamRun(runId, (m) => {
      if (m.type === "log") setLines((l) => [...l.slice(-800), m.line]);
      else if (m.type === "status") setStatus(m.status);
      else if (m.type === "event") {
        if (m.event === "epoch") {
          // Keyed by epoch so a reconnect's replay updates rather than duplicates.
          setEpochs((e) => e.some((x) => x.epoch === m.epoch)
            ? e.map((x) => (x.epoch === m.epoch ? m : x))
            : [...e, m]);
          setProgress(null);
        }
        else if (m.event === "progress") setProgress(m);
        else if (m.event === "started") setMeta((x) => ({ ...(x as any), run_name: m.run_name }));
        else if (m.event === "finished" || m.event === "failed") setProgress(null);
      }
    });
  }, [runId]);

  React.useEffect(() => {
    if (stick && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [lines, stick]);

  const last = epochs[epochs.length - 1];
  const primary = last?.primary_metric ?? "acc@1";
  const totalEpochs = last?.total ?? meta?.epochs ?? 0;
  const done = last ? last.epoch + 1 : 0;
  const pct = totalEpochs ? (done / totalEpochs) * 100 : 0;
  const live = status === "running" || status === "starting";

  return (
    <Panel
      title={
        <span className="flex items-center gap-2">
          <Terminal className="h-3 w-3" />
          <span className="mono normal-case tracking-normal text-[var(--color-fg)]">
            {meta?.run_name || runId}
          </span>
          <Badge variant={STATUS_TONE[status] ?? "default"}>{status}</Badge>
        </span>
      }
      actions={
        <div className="flex items-center gap-1">
          {live && (
            <Button variant="ghost" size="sm" onClick={() => api.cancelRun(runId)}>
              <Ban className="h-3 w-3" /> Cancel
            </Button>
          )}
          {onClose && <Button variant="ghost" size="sm" onClick={onClose}>Close</Button>}
        </div>
      }
    >
      <div className="mb-2">
        <div className="flex items-baseline justify-between text-[11px]">
          <span className="text-[var(--color-muted)]">
            epoch {done}/{totalEpochs || "?"}
            {progress?.batch != null && (
              <span className="ml-2 tnum opacity-70">
                batch {progress.batch}/{progress.total_batches}
              </span>
            )}
          </span>
          {last && (
            <span className="tnum">
              best {primary} <b className="text-[var(--color-ok)]">{fmtNum(last.best)}</b>
            </span>
          )}
        </div>
        <div className="mt-1 h-1 overflow-hidden rounded-full bg-[var(--color-border)]">
          <div
            className={cn("h-full rounded-full transition-all",
              status === "failed" ? "bg-[var(--color-danger)]" : "bg-[var(--color-accent)]")}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {progress?.ms_per_step != null && <ThroughputBar p={progress} />}

      {epochs.length > 0 && <EpochTable epochs={epochs} primary={primary} />}

      <div
        ref={logRef}
        onScroll={(e) => {
          const el = e.currentTarget;
          setStick(el.scrollHeight - el.scrollTop - el.clientHeight < 24);
        }}
        className="mt-2 max-h-56 overflow-y-auto rounded border border-[var(--color-border)] bg-[var(--color-bg)] p-2 mono text-[10px] leading-relaxed"
      >
        {lines.length === 0 && <span className="text-[var(--color-muted)]">waiting for output…</span>}
        {lines.map((l, i) => (
          <div key={i} className={cn(
            "whitespace-pre-wrap break-all",
            l.startsWith("  !") && "text-[var(--color-modified)]",
            l.startsWith("✗") && "text-[var(--color-danger)]",
            l.startsWith("✓") && "text-[var(--color-ok)]",
            l.includes("[warning]") && "text-[var(--color-modified)]",
          )}>{l}</div>
        ))}
      </div>
    </Panel>
  );
}

/**
 * Where the time actually goes, live.
 *
 * The split is the point: the model can only run while a batch is in hand, so a low
 * compute share is the GPU idling on the input pipeline — the case where a bigger model
 * would cost nothing and a faster disk would cost everything. The bar shows it directly
 * rather than making you compare two numbers.
 */
function ThroughputBar({ p }: { p: Json }) {
  const share = typeof p.compute_share === "number" ? p.compute_share : 1;
  const starved = share < 0.7;
  return (
    <div className="mb-2 rounded border border-[var(--color-border)] bg-[var(--color-bg)] px-2 py-1.5">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 text-[10px] tnum">
        <span className="text-[var(--color-fg)]">
          {fmtNum(p.imgs_per_s, 0)} <span className="text-[var(--color-muted)]">img/s</span>
        </span>
        <span className="text-[var(--color-muted)]">{fmtNum(p.ms_per_step, 0)} ms/step</span>
        <span className="text-[var(--color-muted)]">
          data <span className={cn(starved && "text-[var(--color-modified)]")}>
            {fmtNum(p.ms_data_wait, 0)} ms
          </span>
          {" + "}compute {fmtNum(p.ms_compute, 0)} ms
        </span>
        <span className={cn("ml-auto", starved ? "text-[var(--color-modified)]" : "text-[var(--color-ok)]")}>
          {(share * 100).toFixed(0)}% compute
        </span>
        <span className="text-[var(--color-muted)]">eta {fmtDuration(p.eta_run_s)}</span>
      </div>
      {/* One bar, two parts: what the GPU worked through vs what it waited for. */}
      <div className="mt-1 flex h-[3px] overflow-hidden rounded-full bg-[var(--color-border)]">
        <div className="h-full bg-[var(--color-ok)]" style={{ width: `${share * 100}%` }} />
        <div className="h-full bg-[var(--color-modified)]" style={{ width: `${(1 - share) * 100}%` }} />
      </div>
    </div>
  );
}

function EpochTable({ epochs, primary }: { epochs: Json[]; primary: string }) {
  const rows = epochs.slice(-6).reverse();
  const cols = ["train_loss", "val_loss", primary, `ema/${primary}`];
  return (
    <div className="overflow-x-auto rounded border border-[var(--color-border)]">
      <table className="w-full text-[10px] tnum">
        <thead>
          <tr className="border-b border-[var(--color-border)] text-[var(--color-muted)]">
            <th className="px-2 py-1 text-left font-medium">ep</th>
            {cols.map((c) => <th key={c} className="px-2 py-1 text-right font-medium">{c}</th>)}
            <th className="px-2 py-1 text-right font-medium">time</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((e) => (
            <tr key={e.epoch} className="border-b border-[var(--color-border)]/50 last:border-0">
              <td className="px-2 py-0.5">{e.epoch + 1}</td>
              {cols.map((c) => (
                <td key={c} className="px-2 py-0.5 text-right">{fmtNum(e.metrics?.[c])}</td>
              ))}
              <td className="px-2 py-0.5 text-right text-[var(--color-muted)]">
                {fmtDuration(e.metrics?.epoch_time_s)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
