import { AlertCircle, AlertTriangle, Cpu, HardDrive, Info, Wand2 } from "lucide-react";
import type { Json, ValidateResult } from "@/lib/api";
import { cn, fmtDuration } from "@/lib/utils";
import { Badge, Button, Panel } from "./ui/primitives";

const ICON = {
  error: AlertCircle,
  warning: AlertTriangle,
  info: Info,
};
const TONE = {
  error: "border-[var(--color-danger)]/30 bg-[var(--color-danger)]/10 text-[var(--color-danger)]",
  warning: "border-[var(--color-modified)]/30 bg-[var(--color-modified)]/10 text-[var(--color-modified)]",
  info: "border-[var(--color-border)] bg-[var(--color-panel2)] text-[var(--color-muted)]",
};

/**
 * Maps a warning to the concrete config change that resolves it.
 * Only warnings with an unambiguous single fix get a button — the rest are advice.
 */
function quickFix(field: string | null, config: Json): { path: string; value: any } | null {
  switch (field) {
    case "augmentation.mixup_alpha":
      return { path: "augmentation.mixup_alpha", value: 0 };
    case "model.ema_decay":
      return { path: "model.ema_decay", value: "auto" };
    case "input.rrc_scale":
      return { path: "input.rrc_scale", value: [0.65, 1.0] };
    case "schedule.amp":
      return { path: "schedule.amp", value: "fp16" };
    case "input.channels_last":
      return { path: "input.channels_last", value: false };
    case "validation.primary_metric":
      return { path: "validation.primary_metric", value: "macro-F1" };
    case "loss.loss":
      return config.loss?.loss === "focal"
        ? { path: "loss.loss", value: "cross_entropy" }
        : null;
    case "model.torch_compile":
      return { path: "model.torch_compile", value: false };
    case "schedule.batch_size":
      return { path: "model.freeze_bn", value: true };
    default:
      return null;
  }
}

export function WarningsPanel({
  result, config, estimate, onApplyFix,
}: {
  result: ValidateResult | null;
  config: Json;
  estimate: Json | null;
  onApplyFix: (path: string, value: any) => void;
}) {
  const errors = result?.errors ?? [];
  const warnings = result?.warnings ?? [];
  const counts = {
    warning: warnings.filter((w) => w.severity === "warning").length,
    info: warnings.filter((w) => w.severity === "info").length,
  };

  return (
    <Panel
      title={
        <span className="flex items-center gap-2">
          Validation
          {errors.length > 0 && <Badge variant="danger">{errors.length} error</Badge>}
          {counts.warning > 0 && <Badge variant="warn">{counts.warning}</Badge>}
          {counts.info > 0 && <Badge>{counts.info}</Badge>}
          {errors.length === 0 && counts.warning === 0 && <Badge variant="ok">clean</Badge>}
        </span>
      }
    >
      <div className="space-y-1.5">
        {errors.map((e, i) => (
          <div key={`e${i}`} className={cn("rounded border p-2 text-[11px] leading-relaxed", TONE.error)}>
            <div className="mono text-[10px] opacity-70">{e.field}</div>
            {e.message}
          </div>
        ))}

        {warnings.map((w, i) => {
          const Icon = ICON[w.severity];
          const fix = quickFix(w.field, config);
          return (
            <div key={i} className={cn("rounded border p-2 text-[11px] leading-relaxed", TONE[w.severity])}>
              <div className="flex items-start gap-1.5">
                <Icon className="mt-px h-3 w-3 shrink-0" />
                <div className="min-w-0 flex-1">
                  {w.field && <div className="mono text-[10px] opacity-60">{w.field}</div>}
                  <div>{w.message}</div>
                  {w.fix && <div className="mt-0.5 opacity-75">→ {w.fix}</div>}
                </div>
                {fix && (
                  <Button
                    variant="ghost" size="sm"
                    className="shrink-0 !text-[10px] !h-5"
                    onClick={() => onApplyFix(fix.path, fix.value)}
                  >
                    <Wand2 className="h-2.5 w-2.5" /> Fix
                  </Button>
                )}
              </div>
            </div>
          );
        })}

        {errors.length === 0 && warnings.length === 0 && (
          <div className="py-3 text-center text-[11px] text-[var(--color-muted)]">
            No issues detected in this configuration.
          </div>
        )}
      </div>

      {estimate && <Estimates estimate={estimate} config={config} />}
    </Panel>
  );
}

function Estimates({ estimate, config }: { estimate: Json; config: Json }) {
  const mem = estimate.memory ?? {};
  const total = mem.total_gb ?? 0;
  const cap = estimate.device_memory_gb;
  const pct = cap ? Math.min(100, (total / cap) * 100) : 0;
  const tone = estimate.risk === "high" ? "var(--color-danger)"
    : estimate.risk === "medium" ? "var(--color-modified)" : "var(--color-ok)";

  const epochs = config.schedule?.epochs ?? 0;

  return (
    <div className="mt-3 space-y-2 border-t border-[var(--color-border)] pt-2">
      <div className="flex items-center justify-between text-[10px] text-[var(--color-muted)]">
        <span className="flex items-center gap-1">
          <Cpu className="h-2.5 w-2.5" /> {estimate.device_name}
        </span>
        <span className="tnum">{epochs} epochs</span>
      </div>

      <div>
        <div className="flex items-center justify-between text-[10px]">
          <span className="flex items-center gap-1 text-[var(--color-muted)]">
            <HardDrive className="h-2.5 w-2.5" /> Est. memory
          </span>
          <span className="tnum" style={{ color: tone }}>
            ~{total.toFixed(1)} GB{cap ? ` / ${cap.toFixed(0)} GB` : ""}
          </span>
        </div>
        <div className="mt-1 h-1 overflow-hidden rounded-full bg-[var(--color-border)]">
          <div className="h-full rounded-full transition-all"
               style={{ width: `${pct}%`, background: tone }} />
        </div>
        <div className="pt-0.5 text-[9px] text-[var(--color-muted)]">
          Heuristic from batch × resolution² × backbone — not a measurement.
        </div>
      </div>

      {estimate.downgrades?.length > 0 && (
        <div className="space-y-0.5">
          {estimate.downgrades.map((d: string, i: number) => (
            <div key={i} className="text-[10px] text-[var(--color-muted)]">↓ {d}</div>
          ))}
        </div>
      )}
    </div>
  );
}

export function DatasetPanel({ info }: { info: Json | null }) {
  if (!info) return null;
  if (!info.ok) {
    return (
      <Panel title="Dataset">
        <div className="text-[11px] text-[var(--color-danger)] mono break-all">{info.error}</div>
      </Panel>
    );
  }
  const counts: [string, number][] = Object.entries(info.class_counts ?? {});
  const max = Math.max(1, ...counts.map(([, v]) => v));
  const imbalanced = info.imbalance_ratio >= 3;

  return (
    <Panel
      title={
        <span className="flex items-center gap-2">
          Dataset
          {imbalanced && <Badge variant="warn">{info.imbalance_ratio}:1 imbalance</Badge>}
        </span>
      }
    >
      <div className="mb-2 flex items-center gap-3 text-[11px] tnum">
        <span>{info.num_images.toLocaleString()} images</span>
        <span className="text-[var(--color-muted)]">·</span>
        <span>{info.num_classes} classes</span>
        <span className="ml-auto text-[10px] text-[var(--color-muted)] mono">
          {info.fingerprint?.sha256}
        </span>
      </div>
      <div className="max-h-32 space-y-px overflow-y-auto pr-1">
        {counts.sort((a, b) => b[1] - a[1]).map(([name, n]) => (
          <div key={name} className="flex items-center gap-2">
            <span className="w-20 shrink-0 truncate text-[10px] text-[var(--color-muted)]">{name}</span>
            <div className="h-2 flex-1 overflow-hidden rounded-sm bg-[var(--color-bg)]">
              <div className="h-full rounded-sm bg-[var(--color-accent)]/60"
                   style={{ width: `${(n / max) * 100}%` }} />
            </div>
            <span className="w-9 shrink-0 text-right text-[10px] tnum text-[var(--color-muted)]">{n}</span>
          </div>
        ))}
      </div>
    </Panel>
  );
}

export function RunTimePanel({ seconds }: { seconds: number | null }) {
  if (seconds == null) return null;
  return <div className="text-[10px] text-[var(--color-muted)]">≈ {fmtDuration(seconds)}</div>;
}
