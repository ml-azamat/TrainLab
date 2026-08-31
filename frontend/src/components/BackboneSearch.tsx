import * as React from "react";
import { Check, Search } from "lucide-react";
import { api, type Json } from "@/lib/api";
import { cn, fmtParams } from "@/lib/utils";
import { Badge, Checkbox, Input, Popover, PopoverContent, PopoverTrigger } from "./ui/primitives";

/**
 * Searchable picker over the whole timm catalogue (~1400 architectures).
 * Parameter counts require instantiating the model server-side, so the list shows
 * whatever the server has already memoised and the highlighted row is fetched in full.
 */
export function BackboneSearch({
  value, onChange, config,
}: { value: string; onChange: (v: string) => void; config: Json }) {
  const [open, setOpen] = React.useState(false);
  const [q, setQ] = React.useState("");
  const [pretrainedOnly, setPretrainedOnly] = React.useState(true);
  const [items, setItems] = React.useState<Json[]>([]);
  const [total, setTotal] = React.useState(0);
  const [detail, setDetail] = React.useState<Json | null>(null);
  const [loading, setLoading] = React.useState(false);

  React.useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    const t = setTimeout(() => {
      api.backbones(q, pretrainedOnly, 60)
        .then((r) => { if (!cancelled) { setItems(r.items); setTotal(r.total); } })
        .finally(() => !cancelled && setLoading(false));
    }, 180);
    return () => { cancelled = true; clearTimeout(t); };
  }, [q, pretrainedOnly, open]);

  React.useEffect(() => {
    let cancelled = false;
    api.backbone(value).then((d) => !cancelled && setDetail(d)).catch(() => {});
    return () => { cancelled = true; };
  }, [value]);

  const inputSize = config?.input?.input_size;
  const mismatch = detail?.input_size && inputSize && detail.input_size !== inputSize;

  return (
    <div className="flex w-full min-w-0 flex-col gap-0.5">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <button className="flex h-7 w-full items-center gap-1.5 rounded-md border border-[var(--color-border)] bg-[var(--color-bg)] px-2 text-left text-xs hover:border-[var(--color-muted)]">
            <Search className="h-3 w-3 shrink-0 opacity-40" />
            <span className="truncate mono">{value}</span>
          </button>
        </PopoverTrigger>
        <PopoverContent className="w-[460px] p-0">
          <div className="border-b border-[var(--color-border)] p-2">
            <Input
              autoFocus placeholder="Search 1,400+ timm architectures…"
              value={q} onChange={(e) => setQ(e.target.value)}
            />
            <label className="mt-2 flex cursor-pointer items-center gap-1.5 text-[11px] text-[var(--color-muted)]">
              <Checkbox checked={pretrainedOnly} onCheckedChange={setPretrainedOnly} />
              Pretrained weights available
              <span className="ml-auto tnum">
                {loading ? "…" : `${Math.min(60, total)} of ${total}`}
              </span>
            </label>
          </div>
          <div className="max-h-72 overflow-y-auto p-1">
            {items.map((m) => (
              <button
                key={m.name}
                onClick={() => { onChange(m.name); setOpen(false); }}
                className={cn(
                  "flex w-full items-center gap-2 rounded px-2 py-1 text-left text-xs hover:bg-[var(--color-accent)]/15",
                  m.name === value && "bg-[var(--color-accent)]/10",
                )}
              >
                <span className="w-3 shrink-0">
                  {m.name === value && <Check className="h-3 w-3 text-[var(--color-accent)]" />}
                </span>
                <span className="mono flex-1 truncate">{m.name}</span>
                {m.params != null && (
                  <span className="shrink-0 tnum text-[10px] text-[var(--color-muted)]">
                    {fmtParams(m.params)}
                  </span>
                )}
                <span className="shrink-0 tnum text-[10px] text-[var(--color-muted)]">
                  {m.input_size}px
                </span>
                {m.pretrained
                  ? <Badge variant="ok" className="shrink-0">pt</Badge>
                  : <Badge className="shrink-0">—</Badge>}
              </button>
            ))}
            {!loading && items.length === 0 && (
              <div className="p-4 text-center text-[11px] text-[var(--color-muted)]">
                No architecture matches “{q}”.
              </div>
            )}
          </div>
        </PopoverContent>
      </Popover>

      <div className="flex items-center gap-2 pl-0.5 text-[10px] text-[var(--color-muted)]">
        {detail?.params != null && <span className="tnum">{fmtParams(detail.params)} params</span>}
        {detail?.input_size && <span className="tnum">native {detail.input_size}px</span>}
        {detail?.pretrained === false && <Badge variant="warn">no pretrained weights</Badge>}
        {mismatch && (
          <Badge variant="warn">
            trained at {detail.input_size}px, you set {inputSize}px
          </Badge>
        )}
      </div>
    </div>
  );
}
