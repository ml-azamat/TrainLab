import * as React from "react";
import { Plus, Sparkles, Trash2, X } from "lucide-react";
import { api, type FieldDef, type Json } from "@/lib/api";
import { cn, fmtNum, shortLabel } from "@/lib/utils";
import {
  Badge, Button, Input, NumberBox, Panel, Select, Tooltip,
} from "@/components/ui/primitives";
import { ResizableSplit } from "@/components/ResizableSplit";

type Space = {
  kind: "list" | "range" | "log_range" | "int_range";
  values?: any[]; low?: number; high?: number; step?: number;
};

/** Sensible starting search space for the parameters people actually sweep. */
const SUGGESTED: Record<string, Space> = {
  "optimization.lr": { kind: "log_range", low: 1e-5, high: 3e-3 },
  "optimization.weight_decay": { kind: "log_range", low: 1e-5, high: 0.3 },
  "optimization.layer_lr_decay": { kind: "list", values: [1.0, 0.85, 0.75, 0.65] },
  "model.drop_path_rate": { kind: "range", low: 0.0, high: 0.4, step: 0.1 },
  "augmentation.randaugment_m": { kind: "int_range", low: 3, high: 12, step: 1 },
  "schedule.batch_size": { kind: "list", values: [16, 32, 64, 128] },
  "loss.label_smoothing": { kind: "list", values: [0.0, 0.05, 0.1, 0.2] },
  "input.input_size": { kind: "list", values: [160, 192, 224, 256] },
};

export function SweepsTab({ config, fields }: { config: Json; fields: FieldDef[] }) {
  const [spaces, setSpaces] = React.useState<Record<string, Space>>({
    "optimization.lr": SUGGESTED["optimization.lr"],
    "optimization.weight_decay": SUGGESTED["optimization.weight_decay"],
  });
  const [algorithm, setAlgorithm] = React.useState("tpe");
  const [budget, setBudget] = React.useState(12);
  const [pruning, setPruning] = React.useState("median");
  const [metric, setMetric] = React.useState("acc_at_1");
  const [sweeps, setSweeps] = React.useState<Json[]>([]);
  const [detail, setDetail] = React.useState<Json | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [adding, setAdding] = React.useState(false);

  const sweepable = React.useMemo(
    () => fields.filter((f) => ["number", "integer", "enum"].includes(f.type)
                              && f.ui.widget !== "readonly"),
    [fields],
  );

  const refresh = React.useCallback(() => {
    api.sweeps().then((r) => setSweeps(r.sweeps)).catch(() => {});
  }, []);

  React.useEffect(() => {
    refresh();
    const t = setInterval(() => {
      refresh();
      if (detail?.id) api.sweep(detail.id).then(setDetail).catch(() => {});
    }, 3000);
    return () => clearInterval(t);
  }, [refresh, detail?.id]);

  const launch = async () => {
    setError(null);
    try {
      const s = await api.startSweep({
        base_config: config,
        parameters: spaces,
        algorithm, budget, metric, pruning,
        // Direction is deliberately omitted: the backend derives it from the metric
        // via metric_higher_is_better, which is the single source of truth. The old
        // client-side `metric.includes("loss")` rule would have sent maximize for a
        // lower-is-better metric like ece the day it was added to the picker.
        experiment_name: config.tracking?.experiment_name ?? "default",
      });
      setDetail(s);
      refresh();
    } catch (e: any) { setError(e.message ?? String(e)); }
  };

  const nTrials = React.useMemo(() => {
    if (algorithm !== "grid") return budget;
    return Object.values(spaces).reduce((n, s) => {
      if (s.kind === "list") return n * (s.values?.length ?? 1);
      if (s.kind === "int_range") return n * Math.max(1, Math.floor(((s.high! - s.low!) / (s.step || 1)) + 1));
      return n * 5;
    }, 1);
  }, [spaces, algorithm, budget]);

  return (
    <ResizableSplit
      storageKey="sweeps" initial={0.5} minLeft={320} minRight={380}
      className="h-full min-h-0 overflow-hidden p-3"
    >
      <div className="min-h-0 space-y-3 overflow-y-auto">
        <Panel
          title="Search space"
          actions={
            <Button variant="ghost" size="sm" onClick={() => setAdding((v) => !v)}>
              <Plus className="h-3 w-3" /> Add parameter
            </Button>
          }
        >
          {adding && (
            <div className="mb-2 max-h-52 overflow-y-auto rounded border border-[var(--color-border)] p-1">
              {sweepable.filter((f) => !(f.path in spaces)).map((f) => (
                <button
                  key={f.path}
                  onClick={() => {
                    setSpaces((s) => ({
                      ...s,
                      [f.path]: SUGGESTED[f.path] ?? defaultSpace(f),
                    }));
                    setAdding(false);
                  }}
                  className="flex w-full items-center gap-2 rounded px-2 py-1 text-left text-[11px] hover:bg-[var(--color-accent)]/15"
                >
                  <span className="mono flex-1 truncate">{f.path}</span>
                  {SUGGESTED[f.path] && <Badge variant="accent">suggested</Badge>}
                </button>
              ))}
            </div>
          )}

          <div className="space-y-1.5">
            {Object.entries(spaces).map(([path, space]) => (
              <SpaceRow
                key={path} path={path} space={space}
                onChange={(s) => setSpaces((p) => ({ ...p, [path]: s }))}
                onRemove={() => setSpaces((p) => { const n = { ...p }; delete n[path]; return n; })}
              />
            ))}
            {Object.keys(spaces).length === 0 && (
              <div className="py-4 text-center text-[11px] text-[var(--color-muted)]">
                Add at least one parameter to search over.
              </div>
            )}
          </div>
        </Panel>

        <Panel title="Strategy">
          <div className="grid grid-cols-2 gap-x-3 gap-y-2">
            <Labeled label="Algorithm" tip="TPE learns from finished trials and beats random search after ~10 runs. Grid is exhaustive and only sane for small discrete spaces.">
              <Select value={algorithm} onValueChange={setAlgorithm}
                      options={[{ value: "tpe", label: "Optuna TPE" }, { value: "random" }, { value: "grid" }]} />
            </Labeled>
            <Labeled label="Budget" tip="Maximum number of trials. Ignored by grid, which runs the full product.">
              <Input type="number" value={budget} disabled={algorithm === "grid"}
                     onChange={(e) => setBudget(Number(e.target.value))} />
            </Labeled>
            <Labeled label="Pruning" tip="Stops trials that are clearly behind the median at the same epoch. Roughly halves the cost of a sweep; disable it if your metric is noisy early on.">
              <Select value={pruning} onValueChange={setPruning}
                      options={[{ value: "median" }, { value: "hyperband" }, { value: "none" }]} />
            </Labeled>
            <Labeled label="Objective" tip="Which logged metric the sweep optimises. Loss metrics are minimised, everything else maximised.">
              <Select value={metric} onValueChange={setMetric}
                      options={[{ value: "acc_at_1" }, { value: "macro-F1" },
                                { value: "balanced-accuracy" }, { value: "val_loss" }]} />
            </Labeled>
          </div>

          <div className="mt-3 flex items-center gap-2 border-t border-[var(--color-border)] pt-2">
            <span className="text-[10px] text-[var(--color-muted)]">
              {algorithm === "grid"
                ? `${nTrials} trials (full grid)`
                : `up to ${budget} trials`} · runs sequentially · each trial is a normal tracked run
            </span>
            <Button
              variant="default" className="ml-auto"
              disabled={Object.keys(spaces).length === 0 || !config.data?.train_dir}
              onClick={launch}
            >
              <Sparkles className="h-3 w-3" /> Launch sweep
            </Button>
          </div>
          {error && <div className="mt-2 text-[11px] text-[var(--color-danger)]">{error}</div>}
          {!config.data?.train_dir && (
            <div className="mt-2 text-[11px] text-[var(--color-muted)]">
              Set a train directory on the Configure tab first.
            </div>
          )}
        </Panel>
      </div>

      <div className="min-h-0 space-y-3 overflow-y-auto">
        <Panel title="Sweeps">
          {sweeps.length === 0 && (
            <div className="py-6 text-center text-[11px] text-[var(--color-muted)]">
              No sweeps yet.
            </div>
          )}
          <div className="space-y-1">
            {sweeps.map((s) => (
              <button
                key={s.id} onClick={() => api.sweep(s.id).then(setDetail)}
                className={cn("flex w-full items-center gap-2 rounded px-2 py-1.5 text-left hover:bg-[var(--color-panel2)]",
                  detail?.id === s.id && "bg-[var(--color-accent)]/10")}
              >
                <Badge variant={s.status === "running" ? "accent" : s.status === "finished" ? "ok" : "default"}>
                  {s.status}
                </Badge>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[11px] mono">{s.parameters.join(", ")}</div>
                  <div className="text-[10px] text-[var(--color-muted)]">
                    {s.algorithm} · {s.completed}/{s.budget} trials
                  </div>
                </div>
                {s.best && (
                  <span className="text-[11px] tnum text-[var(--color-ok)]">{fmtNum(s.best.value)}</span>
                )}
              </button>
            ))}
          </div>
        </Panel>

        {detail && <SweepDetail detail={detail} onStop={() => api.stopSweep(detail.id).then(refresh)} />}
      </div>
    </ResizableSplit>
  );
}

function defaultSpace(f: FieldDef): Space {
  if (f.type === "enum") return { kind: "list", values: f.enumValues ?? [] };
  if (f.type === "integer") return { kind: "int_range", low: f.min ?? 1, high: f.max ?? 10, step: 1 };
  return { kind: "range", low: f.min ?? 0, high: f.max ?? 1, step: undefined };
}

function SpaceRow({
  path, space, onChange, onRemove,
}: { path: string; space: Space; onChange: (s: Space) => void; onRemove: () => void }) {
  return (
    <div className="rounded border border-[var(--color-border)] bg-[var(--color-bg)] p-2">
      <div className="mb-1.5 flex items-center gap-2">
        <span className="flex-1 truncate mono text-[11px]">{path}</span>
        <Select
          className="w-28" value={space.kind}
          onValueChange={(k) => onChange({ ...space, kind: k as Space["kind"] })}
          options={[{ value: "list" }, { value: "range" },
                    { value: "log_range", label: "log range" }, { value: "int_range", label: "int range" }]}
        />
        <Button variant="ghost" size="sm" onClick={onRemove}><Trash2 className="h-3 w-3" /></Button>
      </div>
      {space.kind === "list" ? (
        <Input
          value={(space.values ?? []).join(", ")}
          placeholder="0.1, 0.2, 0.3"
          onChange={(e) => onChange({
            ...space,
            values: e.target.value.split(",").map((s) => {
              const t = s.trim();
              const n = Number(t);
              return t !== "" && Number.isFinite(n) ? n : t;
            }).filter((v) => v !== ""),
          })}
        />
      ) : (
        // A bound left empty is a bound the sweep does not get, so `null` becomes
        // `undefined` — it must be absent from the request, not present and null.
        <div className="flex items-center gap-1.5">
          <NumberBox placeholder="min" value={space.low}
                     onChange={(v) => onChange({ ...space, low: v ?? undefined })} />
          <span className="text-[10px] text-[var(--color-muted)]">to</span>
          <NumberBox placeholder="max" value={space.high}
                     onChange={(v) => onChange({ ...space, high: v ?? undefined })} />
          {space.kind !== "log_range" && (
            <NumberBox placeholder="step" value={space.step}
                       onChange={(v) => onChange({ ...space, step: v ?? undefined })} />
          )}
        </div>
      )}
    </div>
  );
}

function SweepDetail({ detail, onStop }: { detail: Json; onStop: () => void }) {
  const trials: Json[] = detail.trials ?? [];
  const values = trials.filter((t) => t.value != null).map((t) => t.value);
  const best = values.length ? Math.max(...values) : null;

  return (
    <Panel
      title={<span className="mono normal-case tracking-normal">sweep {detail.id}</span>}
      actions={detail.status === "running" && (
        <Button variant="ghost" size="sm" onClick={onStop}><X className="h-3 w-3" /> Stop</Button>
      )}
    >
      {detail.best && (
        <div className="mb-2 rounded border border-[var(--color-ok)]/30 bg-[var(--color-ok)]/10 p-2">
          <div className="text-[10px] uppercase text-[var(--color-muted)]">Best so far</div>
          <div className="text-sm font-semibold tnum text-[var(--color-ok)]">
            {fmtNum(detail.best.value)}
          </div>
          <div className="mt-1 space-y-px">
            {Object.entries(detail.best.params).map(([k, v]) => (
              <div key={k} className="flex justify-between text-[10px]">
                <span className="mono text-[var(--color-muted)]">{shortLabel(k)}</span>
                <span className="tnum">{String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <table className="w-full text-[10px]">
        <thead>
          <tr className="border-b border-[var(--color-border)] text-[var(--color-muted)]">
            <th className="px-1 py-1 text-left font-medium">#</th>
            {Object.keys(trials[0]?.params ?? {}).map((p) => (
              <th key={p} className="px-1 py-1 text-right font-medium">
                <Tooltip content={p}><span>{shortLabel(p)}</span></Tooltip>
              </th>
            ))}
            <th className="px-1 py-1 text-right font-medium">value</th>
            <th className="px-1 py-1 text-right font-medium">status</th>
          </tr>
        </thead>
        <tbody>
          {trials.map((t) => (
            <tr key={t.trial} className="border-b border-[var(--color-border)]/40 last:border-0">
              <td className="px-1 py-0.5">{t.trial}</td>
              {Object.values(t.params).map((v, i) => (
                <td key={i} className="px-1 py-0.5 text-right tnum">
                  {typeof v === "number" ? fmtNum(v, 5) : String(v)}
                </td>
              ))}
              <td className={cn("px-1 py-0.5 text-right tnum",
                t.value === best && "font-semibold text-[var(--color-ok)]")}>
                {t.value != null ? fmtNum(t.value) : "—"}
              </td>
              <td className="px-1 py-0.5 text-right text-[var(--color-muted)]">{t.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {trials.length === 0 && (
        <div className="py-4 text-center text-[11px] text-[var(--color-muted)]">
          Waiting for the first trial…
        </div>
      )}
      <div className="pt-2 text-[10px] text-[var(--color-muted)]">
        Every trial is tagged <span className="mono">sweep_id={detail.id}</span> and shows up in
        the Compare tab alongside hand-launched runs.
      </div>
    </Panel>
  );
}

function Labeled({
  label, tip, children,
}: { label: string; tip: string; children: React.ReactNode }) {
  return (
    <div>
      <Tooltip content={tip}>
        <div className="mb-0.5 cursor-help text-[10px] text-[var(--color-muted)]">{label}</div>
      </Tooltip>
      {children}
    </div>
  );
}
