import * as React from "react";
import { AlertTriangle, Eye, RefreshCw } from "lucide-react";
import { api, type Json } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge, Button, Panel, Tooltip } from "./ui/primitives";

/**
 * Live augmentation preview.
 *
 * Renders the real pipeline built from the real config — the same code path training
 * uses — because an approximation would defeat the point. Images are sampled with a
 * fixed seed so that changing one knob shows the effect of that knob rather than a
 * different random draw.
 */
export function AugPreview({ config, datasetOk }: { config: Json; datasetOk: boolean }) {
  const [data, setData] = React.useState<Json | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [seed, setSeed] = React.useState(0);
  const [showOriginal, setShowOriginal] = React.useState(false);
  const [mode, setMode] = React.useState<"train" | "eval">("train");

  // Only the fields that actually change pixels should trigger a re-render.
  const key = React.useMemo(
    () => JSON.stringify([config.augmentation, config.input, config.data?.train_dir,
                          config.data?.dataset_format, mode]),
    [config.augmentation, config.input, config.data?.train_dir,
     config.data?.dataset_format, mode],
  );

  React.useEffect(() => {
    if (!datasetOk) { setData(null); return; }
    let cancelled = false;
    setLoading(true);
    const t = setTimeout(() => {
      const fn = mode === "train"
        ? api.previewAug(config, 8, seed)
        : api.previewEval(config, 8, seed);
      fn.then((d) => !cancelled && setData(d))
        .catch((e) => !cancelled && setData({ ok: false, error: String(e) }))
        .finally(() => !cancelled && setLoading(false));
    }, 300); // debounce: dragging a slider must not fire a request per frame
    return () => { cancelled = true; clearTimeout(t); };
  }, [key, seed, datasetOk]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <Panel
      title={
        <span className="flex items-center gap-1.5">
          <Eye className="h-3 w-3" /> Augmentation preview
          {loading && <RefreshCw className="h-2.5 w-2.5 spin opacity-50" />}
        </span>
      }
      actions={
        <div className="flex items-center gap-1">
          <button
            onClick={() => setMode(mode === "train" ? "eval" : "train")}
            className="rounded px-1.5 py-px text-[10px] text-[var(--color-muted)] hover:text-[var(--color-fg)]"
          >
            {mode === "train" ? "train" : "val"}
          </button>
          <Tooltip content="Compare against the un-augmented source image">
            <button
              onClick={() => setShowOriginal((v) => !v)}
              className={cn("rounded px-1.5 py-px text-[10px]",
                showOriginal ? "text-[var(--color-accent)]" : "text-[var(--color-muted)] hover:text-[var(--color-fg)]")}
            >
              A/B
            </button>
          </Tooltip>
          <Button variant="ghost" size="sm" onClick={() => setSeed((s) => s + 1)} title="Resample images">
            <RefreshCw className="h-3 w-3" />
          </Button>
        </div>
      }
    >
      {!datasetOk && (
        <div className="py-6 text-center text-[11px] text-[var(--color-muted)]">
          Point <span className="mono">Data → Train directory</span> at your dataset to see
          what the model will actually be fed.
        </div>
      )}

      {datasetOk && data?.ok === false && (
        <div className="flex items-start gap-2 rounded border border-[var(--color-danger)]/30 bg-[var(--color-danger)]/10 p-2 text-[11px] text-[var(--color-danger)]">
          <AlertTriangle className="mt-px h-3 w-3 shrink-0" />
          <span className="mono break-all">{data.error}</span>
        </div>
      )}

      {data?.ok && (
        <>
          <div className={cn("grid gap-1.5", showOriginal ? "grid-cols-2" : "grid-cols-4")}>
            {data.items.map((it: Json, i: number) => (
              <div key={i} className="min-w-0">
                {showOriginal && it.original ? (
                  <div className="grid grid-cols-2 gap-px overflow-hidden rounded bg-[var(--color-border)]">
                    <img src={it.original} alt="original" className="aspect-square w-full object-cover" />
                    <img src={it.augmented} alt="augmented" className="aspect-square w-full object-cover" />
                  </div>
                ) : it.augmented ? (
                  <img src={it.augmented} alt={it.label}
                       className="aspect-square w-full rounded object-cover" />
                ) : (
                  <div className="flex aspect-square w-full items-center justify-center rounded bg-[var(--color-panel2)] text-[9px] text-[var(--color-danger)]">
                    failed
                  </div>
                )}
                <div className="truncate pt-0.5 text-center text-[9px] text-[var(--color-muted)]">
                  {it.label}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-2 border-t border-[var(--color-border)] pt-1.5">
            <div className="flex flex-wrap gap-1">
              {data.pipeline?.map((p: string, i: number) => (
                <Badge key={i} className="mono">{p}</Badge>
              ))}
            </div>
            {mode === "train" && (
              <div className="pt-1 text-[9px] text-[var(--color-muted)] mono">{data.normalization}</div>
            )}
          </div>
        </>
      )}
    </Panel>
  );
}
