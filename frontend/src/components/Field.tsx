import * as React from "react";
import { Info, RotateCcw } from "lucide-react";
import type { FieldDef, Json } from "@/lib/api";
import { cn, getPath, parseNumber } from "@/lib/utils";
import {
  Badge, Checkbox, Input, NumberBox, Select, Slider, Switch, Tooltip,
} from "./ui/primitives";
import { BackboneSearch } from "./BackboneSearch";
import { PretrainedTagSelect } from "./PretrainedTagSelect";

interface Props {
  field: FieldDef;
  config: Json;
  defaults: Json;
  disabled?: boolean;
  modified: boolean;
  autoNote?: string | null;
  onChange: (path: string, value: any) => void;
  onReset: (path: string) => void;
}

function enumLabel(v: string) {
  return v.replace(/[-_]/g, " ");
}

export function Field({
  field: f, config, disabled, modified, autoNote, onChange, onReset,
}: Props) {
  const value = getPath(config, f.path);
  const set = (v: any) => onChange(f.path, v);
  const readOnly = f.ui.widget === "readonly";

  return (
    <div className={cn(
      "group grid grid-cols-[minmax(130px,1fr)_minmax(0,1.35fr)] items-center gap-3 py-[3px] pl-2",
      "border-l-2 transition-colors",
      modified ? "border-l-[var(--color-modified)]" : "border-l-transparent",
      disabled && "opacity-40 pointer-events-none",
    )}>
      <div className="flex min-w-0 items-center gap-1">
        {modified && (
          <button
            onClick={() => onReset(f.path)}
            title="Modified from default — click to revert"
            className="shrink-0 text-[var(--color-modified)] hover:text-[var(--color-fg)]"
          >
            <RotateCcw className="h-2.5 w-2.5" />
          </button>
        )}
        <Tooltip content={f.ui.tooltip}>
          <span className="truncate text-[11px] text-[var(--color-muted)] hover:text-[var(--color-fg)] cursor-help">
            {f.ui.label}
            {f.ui.unit && <span className="ml-1 opacity-50">({f.ui.unit})</span>}
          </span>
        </Tooltip>
        <Tooltip content={f.ui.tooltip}>
          <Info className="h-2.5 w-2.5 shrink-0 opacity-0 transition-opacity group-hover:opacity-40" />
        </Tooltip>
      </div>

      <div className="flex min-w-0 items-center gap-2">
        <Control field={f} value={value} set={set} readOnly={readOnly} config={config} />
        {autoNote && <Badge variant="accent" className="shrink-0">{autoNote}</Badge>}
      </div>
    </div>
  );
}

function Control({
  field: f, value, set, readOnly, config,
}: { field: FieldDef; value: any; set: (v: any) => void; readOnly: boolean; config: Json }) {
  if (f.ui.widget === "timm-search") {
    return <BackboneSearch value={value} onChange={set} config={config} />;
  }

  if (f.ui.widget === "timm-pretrained-tag") {
    return <PretrainedTagSelect value={value} onChange={set} config={config} />;
  }

  if (f.ui.widget === "class-select") {
    return <ClassSelect value={value} onChange={set} config={config} />;
  }

  if (readOnly) {
    return (
      <div className="truncate rounded border border-transparent bg-[var(--color-panel2)] px-2 py-[3px] text-[11px] text-[var(--color-muted)]">
        {Array.isArray(value) ? (value.length ? `${value.length} detected` : "auto") : String(value ?? "auto")}
      </div>
    );
  }

  switch (f.type) {
    case "boolean":
      return <Switch checked={Boolean(value)} onCheckedChange={set} />;

    case "enum":
      return (
        <Select
          value={String(value ?? "")}
          onValueChange={set}
          options={(f.enumValues ?? []).map((v) => ({ value: v, label: enumLabel(v) }))}
        />
      );

    case "array":
      if (f.itemEnum) return <MultiSelect options={f.itemEnum} value={value ?? []} onChange={set} />;
      // A list the schema types as numbers has to store numbers: splitting on commas
      // yields strings, and a `list[float]` field silently held them until the server
      // coerced a copy the form never saw.
      if (f.itemType) {
        return <NumberListBox value={value} onChange={set} integer={f.itemType === "integer"} />;
      }
      return (
        <Input
          value={Array.isArray(value) ? value.join(", ") : ""}
          placeholder="comma separated"
          onChange={(e) => set(e.target.value ? e.target.value.split(",").map((s) => s.trim()) : [])}
        />
      );

    case "tuple":
      return <TupleInput value={value ?? []} onChange={set} step={f.ui.step ?? 0.01} />;

    case "integer":
    case "number": {
      // A bounded 0..1 knob is far easier to reason about as a slider than a text box.
      const sliderish =
        f.type === "number" && f.min != null && f.max != null &&
        f.max - f.min <= 2 && f.ui.widget !== "log";
      if (sliderish) {
        return (
          <div className="flex w-full items-center gap-2">
            <Slider
              value={[Number(value ?? 0)]} onValueChange={([v]) => set(v)}
              min={f.min} max={f.max} step={f.ui.step ?? 0.01}
            />
            <span className="w-9 shrink-0 text-right text-[11px] tnum text-[var(--color-muted)]">
              {Number(value ?? 0).toFixed(2)}
            </span>
          </div>
        );
      }
      return <NumberBox value={value} onChange={set} integer={f.type === "integer"} />;
    }

    default:
      return (
        <Input
          value={value ?? ""} placeholder={f.ui.widget === "path" ? "/path/to/data" : ""}
          onChange={(e) => set(e.target.value === "" ? null : e.target.value)}
        />
      );
  }
}

/**
 * Comma-separated numbers.
 *
 * Keeps the typed text while the box has focus for the same reason `NumberBox` does: the
 * value it parses to cannot represent "0.01, 0." mid-entry, so re-rendering from it eats
 * the separator or the decimal point as fast as you type them. Segments that are not yet
 * numbers are dropped from the value and left in the text.
 */
function NumberListBox({
  value, onChange, integer,
}: { value: any; onChange: (v: number[]) => void; integer?: boolean }) {
  const [draft, setDraft] = React.useState<string | null>(null);
  return (
    <Input
      inputMode="decimal"
      value={draft ?? (Array.isArray(value) ? value.join(", ") : "")}
      placeholder="comma separated"
      onChange={(e) => {
        setDraft(e.target.value);
        onChange(e.target.value.split(",")
          .map((part) => parseNumber(part, integer))
          .filter((v): v is number => typeof v === "number"));
      }}
      onBlur={() => setDraft(null)}
    />
  );
}

/**
 * Picks one of the dataset's classes.
 *
 * The list is whatever the dataset scan detected, in the sorted order that defines the
 * class indices — so the choice made here is the same string the engine matches against.
 * Before a dataset is set there is nothing to choose from, and the box falls back to free
 * text rather than an empty dropdown that looks broken.
 */
function ClassSelect({
  value, onChange, config,
}: { value: any; onChange: (v: string) => void; config: Json }) {
  const names: string[] = config?.data?.class_names ?? [];
  if (names.length === 0) {
    return (
      <Input
        value={value ?? ""} placeholder="set a dataset to list its classes"
        onChange={(e) => onChange(e.target.value)}
      />
    );
  }
  // A name from a cloned config that this dataset does not have stays visible; validation
  // rejects it by name, which is a clearer message than a silently blank control.
  const options = value && !names.includes(value) ? [...names, value] : names;
  return (
    <Select
      value={value ?? ""} onValueChange={onChange} placeholder="choose a class…"
      options={options.map((n) => ({ value: n }))}
    />
  );
}

function TupleInput({
  value, onChange, step,
}: { value: any[]; onChange: (v: any[]) => void; step: number }) {
  return (
    <div className="flex items-center gap-1">
      {[0, 1].map((i) => (
        <React.Fragment key={i}>
          <Input
            className="h-6 w-full" type="number" step={step} value={value[i] ?? ""}
            onChange={(e) => {
              const next = [...value];
              next[i] = e.target.value === "" ? null : Number(e.target.value);
              onChange(next);
            }}
          />
          {i === 0 && <span className="text-[10px] text-[var(--color-muted)]">–</span>}
        </React.Fragment>
      ))}
    </div>
  );
}

function MultiSelect({
  options, value, onChange,
}: { options: string[]; value: string[]; onChange: (v: string[]) => void }) {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 py-0.5">
      {options.map((o) => {
        const on = value.includes(o);
        return (
          <label key={o} className="flex cursor-pointer items-center gap-1.5 text-[11px]">
            <Checkbox
              checked={on}
              onCheckedChange={(c) => onChange(c ? [...value, o] : value.filter((x) => x !== o))}
            />
            <span className={on ? "" : "text-[var(--color-muted)]"}>{o}</span>
          </label>
        );
      })}
    </div>
  );
}
