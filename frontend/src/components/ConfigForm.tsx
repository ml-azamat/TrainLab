import * as React from "react";
import { FlaskConical, RotateCcw } from "lucide-react";
import type { FieldDef, GroupDef, Json } from "@/lib/api";
import { evalExpr } from "@/lib/expr";
import { buildScope, groupHeadline, groupSummary, isModified, modifiedInGroup } from "@/lib/schema";
import { cn } from "@/lib/utils";
import {
  Accordion, AccordionContent, AccordionItem, AccordionTrigger, Badge, Switch,
} from "./ui/primitives";
import { Field } from "./Field";

interface Props {
  fields: FieldDef[];
  groups: GroupDef[];
  config: Json;
  defaults: Json;
  resolved?: Json;
  onChange: (path: string, value: any) => void;
  onResetField: (path: string) => void;
  onResetGroup: (group: string) => void;
}

export function ConfigForm({
  fields, groups, config, defaults, resolved, onChange, onResetField, onResetGroup,
}: Props) {
  const [open, setOpen] = React.useState<string[]>(
    () => groups.filter((g) => g.expandedByDefault).map((g) => g.key),
  );
  const [advanced, setAdvanced] = React.useState<Record<string, boolean>>({});

  return (
    <Accordion
      type="multiple" value={open} onValueChange={setOpen}
      className="divide-y divide-[var(--color-border)] rounded-lg border border-[var(--color-border)] bg-[var(--color-panel)]"
    >
      {groups.map((g) => (
        <Group
          key={g.key} group={g} fields={fields} config={config} defaults={defaults}
          resolved={resolved} isOpen={open.includes(g.key)}
          showAdvanced={advanced[g.key] ?? false}
          setShowAdvanced={(v) => setAdvanced((a) => ({ ...a, [g.key]: v }))}
          onChange={onChange} onResetField={onResetField} onResetGroup={onResetGroup}
        />
      ))}
    </Accordion>
  );
}

function Group({
  group: g, fields, config, defaults, resolved, isOpen, showAdvanced, setShowAdvanced,
  onChange, onResetField, onResetGroup,
}: {
  group: GroupDef; fields: FieldDef[]; config: Json; defaults: Json; resolved?: Json;
  isOpen: boolean; showAdvanced: boolean; setShowAdvanced: (v: boolean) => void;
  onChange: (p: string, v: any) => void;
  onResetField: (p: string) => void; onResetGroup: (g: string) => void;
}) {
  const groupFields = fields.filter((f) => f.group === g.key);
  const scope = React.useMemo(
    () => buildScope(config, fields, g.key, { resolved_device: resolved?.device ?? "cpu" }),
    [config, fields, g.key, resolved?.device],
  );

  // A broken `showIf` must leave the control visible; a broken `disableIf` must leave it
  // usable. Both default to "the user can still reach it".
  const visible = groupFields.filter((f) => evalExpr(f.ui.showIf, scope, true));
  const basic = visible.filter((f) => !f.ui.advanced);
  const adv = visible.filter((f) => f.ui.advanced);

  const modified = modifiedInGroup(config, defaults, fields, g.key);
  const summary = groupSummary(config, defaults, fields, g.key);
  const headline = groupHeadline(config, g.key);
  const experimental = g.key === "experimental";

  return (
    <AccordionItem value={g.key} className="border-0">
      <AccordionTrigger>
        <div className="flex min-w-0 flex-1 items-baseline gap-2">
          <span className="shrink-0 text-xs font-semibold">{g.title}</span>
          {experimental && (
            <FlaskConical className="h-3 w-3 shrink-0 text-[var(--color-modified)]" />
          )}

          {!isOpen && (
            <span className="flex min-w-0 flex-1 items-baseline gap-1.5 text-[11px] text-[var(--color-muted)]">
              {headline && <span className="shrink-0 mono">{headline}</span>}
              {summary ? (
                <>
                  {headline && <span className="opacity-40">·</span>}
                  <span className="truncate text-[var(--color-modified)]">{summary.text}</span>
                  {summary.extra > 0 && (
                    <span className="shrink-0 opacity-70">+{summary.extra} more</span>
                  )}
                </>
              ) : (
                !headline && <span className="opacity-50">defaults</span>
              )}
            </span>
          )}

          <div className="ml-auto flex shrink-0 items-center gap-2">
            {modified.length > 0 && (
              <span
                role="button" tabIndex={0}
                onClick={(e) => { e.stopPropagation(); onResetGroup(g.key); }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") { e.stopPropagation(); onResetGroup(g.key); }
                }}
                title={`Reset ${g.title} to defaults`}
                className="flex items-center gap-1 rounded px-1 py-px text-[10px] text-[var(--color-modified)] hover:bg-[var(--color-modified)]/15"
              >
                <RotateCcw className="h-2.5 w-2.5" />
                {modified.length}
              </span>
            )}
          </div>
        </div>
      </AccordionTrigger>

      <AccordionContent>
        {experimental && (
          <div className="mb-2 rounded border border-[var(--color-modified)]/30 bg-[var(--color-modified)]/10 px-2 py-1.5 text-[10px] leading-relaxed text-[var(--color-modified)]">
            Higher variance than everything above. These techniques can help a lot or not at
            all depending on your data — expect to spend runs finding out.
          </div>
        )}

        <div className="space-y-0">
          {basic.map((f) => (
            <FieldRow
              key={f.path} f={f} config={config} defaults={defaults} scope={scope}
              resolved={resolved} onChange={onChange} onResetField={onResetField}
            />
          ))}
        </div>

        {adv.length > 0 && (
          <>
            <label className="mt-2 flex cursor-pointer select-none items-center gap-2 border-t border-[var(--color-border)] pt-2 text-[10px] uppercase tracking-wide text-[var(--color-muted)]">
              <Switch checked={showAdvanced} onCheckedChange={setShowAdvanced} />
              Advanced
              <span className="opacity-50">({adv.length})</span>
              {!showAdvanced && adv.some((f) => isModified(config, defaults, f.path)) && (
                <Badge variant="warn" className="ml-1">
                  {adv.filter((f) => isModified(config, defaults, f.path)).length} modified
                </Badge>
              )}
            </label>
            {showAdvanced && (
              <div className="mt-1 space-y-0">
                {adv.map((f) => (
                  <FieldRow
                    key={f.path} f={f} config={config} defaults={defaults} scope={scope}
                    resolved={resolved} onChange={onChange} onResetField={onResetField}
                  />
                ))}
              </div>
            )}
          </>
        )}
      </AccordionContent>
    </AccordionItem>
  );
}

function FieldRow({
  f, config, defaults, scope, resolved, onChange, onResetField,
}: {
  f: FieldDef; config: Json; defaults: Json; scope: Json; resolved?: Json;
  onChange: (p: string, v: any) => void; onResetField: (p: string) => void;
}) {
  const isDisabled = f.ui.disableIf ? evalExpr(f.ui.disableIf, scope, false) : false;

  // Surface server-resolved values inline so the user sees what will actually run.
  let autoNote: string | null = null;
  if (f.path === "loss.loss" && resolved?.loss_autoswitched) {
    autoNote = `auto: ${resolved.loss}`;
  } else if (f.path === "input.test_input_size" && config.input?.test_input_size == null) {
    autoNote = `auto: ${resolved?.test_input_size ?? config.input?.input_size}px`;
  } else if (f.path === "model.ema_decay" && config.model?.ema_decay === "auto") {
    autoNote = "sized to run length";
  }

  return (
    <Field
      field={f} config={config} defaults={defaults} disabled={isDisabled}
      modified={isModified(config, defaults, f.path)} autoNote={autoNote}
      onChange={onChange} onReset={onResetField}
    />
  );
}

export function FormSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 6 }).map((_, i) => (
        <div
          key={i}
          className={cn("h-9 animate-pulse rounded-lg bg-[var(--color-panel)]")}
          style={{ opacity: 1 - i * 0.12 }}
        />
      ))}
    </div>
  );
}
