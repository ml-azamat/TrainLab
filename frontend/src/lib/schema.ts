import type { FieldDef, Json, UiMeta } from "./api";
import { deepEqual, getPath } from "./utils";

/**
 * Turns the Pydantic-generated JSON Schema into a flat list of form fields.
 * This is the whole reason the form never has to be maintained by hand: adding a
 * hyperparameter in `trainlab/config.py` makes it appear here automatically.
 */

function deref(schema: Json, node: Json): Json {
  if (node?.$ref) {
    const key = String(node.$ref).replace("#/$defs/", "");
    return schema.$defs?.[key] ?? {};
  }
  return node;
}

/** Optionals arrive as anyOf[T, null]; unwrap to the meaningful branch. */
function unwrapAnyOf(schema: Json, node: Json): Json {
  if (!node?.anyOf) return node;
  const branches = node.anyOf.filter((b: Json) => b.type !== "null");
  if (branches.length === 0) return node;
  const enumBranch = branches.find((b: Json) => b.$ref || deref(schema, b).enum);
  return deref(schema, enumBranch ?? branches[0]);
}

function classify(schema: Json, raw: Json): Pick<FieldDef, "type" | "enumValues" | "itemEnum" | "itemType" | "min" | "max"> {
  const node = unwrapAnyOf(schema, deref(schema, raw));

  if (node.enum) return { type: "enum", enumValues: node.enum.map(String) };
  if (node.const !== undefined) return { type: "string" };

  if (node.type === "array" || node.prefixItems) {
    if (node.prefixItems) return { type: "tuple" };
    const item = unwrapAnyOf(schema, deref(schema, node.items ?? {}));
    if (item.enum) return { type: "array", itemEnum: item.enum.map(String) };
    // What the elements are, so a list of floats is edited as numbers rather than as the
    // strings a comma-split produces — which is what a `list[float]` field used to store.
    const itemType = item.type === "number" || item.type === "integer" ? item.type : undefined;
    return { type: "array", itemType };
  }

  const min = node.minimum ?? node.exclusiveMinimum;
  const max = node.maximum ?? node.exclusiveMaximum;

  if (node.type === "integer") return { type: "integer", min, max };
  if (node.type === "number") return { type: "number", min, max };
  if (node.type === "boolean") return { type: "boolean" };
  if (node.type === "string") return { type: "string" };

  // Unions like `float | Literal["auto"]` — treat as free text with a numeric hint.
  if (raw.anyOf) {
    const hasNum = raw.anyOf.some((b: Json) => b.type === "number" || b.type === "integer");
    if (hasNum) return { type: "string" };
  }
  return { type: "unknown" };
}

export function buildFields(schema: Json, defaults: Json): FieldDef[] {
  const out: FieldDef[] = [];
  const rootProps = schema.properties ?? {};

  for (const [groupKey, groupNode] of Object.entries<Json>(rootProps)) {
    const groupDef = deref(schema, groupNode as Json);
    if (!groupDef.properties) continue; // scalar root fields (schema_version) are hidden

    for (const [name, propRaw] of Object.entries<Json>(groupDef.properties)) {
      const ui: UiMeta | undefined = (propRaw as Json)["x-ui"];
      if (!ui) continue;
      const path = `${groupKey}.${name}`;
      out.push({
        path,
        name,
        group: groupKey,
        ui,
        default: getPath(defaults, path),
        ...classify(schema, propRaw),
      });
    }
  }
  return out;
}

/**
 * Scope for showIf/disableIf. Fields in the same group win, then any other group's
 * leaf name, then runtime extras like `resolved_device`.
 */
export function buildScope(config: Json, fields: FieldDef[], group: string,
                           extras: Json = {}): Json {
  const scope: Json = {};
  for (const f of fields) {
    if (f.group === group) continue;
    if (!(f.name in scope)) scope[f.name] = getPath(config, f.path);
  }
  for (const f of fields) {
    if (f.group === group) scope[f.name] = getPath(config, f.path);
  }
  return { ...scope, ...extras };
}

export function isModified(config: Json, defaults: Json, path: string): boolean {
  return !deepEqual(getPath(config, path), getPath(defaults, path));
}

export function modifiedInGroup(config: Json, defaults: Json, fields: FieldDef[],
                                group: string): FieldDef[] {
  return fields.filter((f) => f.group === group && isModified(config, defaults, f.path));
}

function formatValue(f: FieldDef, v: any): string {
  if (v == null) return "none";
  if (typeof v === "boolean") return v ? "on" : "off";
  if (Array.isArray(v)) {
    if (f.type === "tuple") return v.map((x) => (typeof x === "number" ? +x.toFixed(2) : x)).join("–");
    return v.length > 2 ? `${v.length} selected` : v.join(", ");
  }
  if (typeof v === "number") {
    if (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e6)) return v.toExponential(1);
    return String(+v.toFixed(4));
  }
  return String(v);
}

/**
 * One-line summary of a collapsed group's non-default values, highest
 * summaryPriority first, so the header answers "what's active here?" without
 * expanding. Returns null when the group is entirely at defaults.
 */
export function groupSummary(config: Json, defaults: Json, fields: FieldDef[],
                             group: string, maxItems = 3): { text: string; extra: number } | null {
  const mod = modifiedInGroup(config, defaults, fields, group)
    .sort((a, b) => b.ui.summaryPriority - a.ui.summaryPriority);
  if (mod.length === 0) return null;

  const shown = mod.slice(0, maxItems).map((f) => {
    const v = getPath(config, f.path);
    const dflt = getPath(defaults, f.path);
    if (f.type === "boolean") return v ? f.ui.label : `no ${f.ui.label.toLowerCase()}`;
    // "Mixup alpha 0" reads as a value; "no mixup" reads as a state. Strip the
    // magnitude noun so a disabled knob announces itself as disabled.
    if (v === 0 && typeof dflt === "number" && dflt !== 0) {
      return `no ${f.ui.label.toLowerCase().replace(/ (alpha|p|rate|deg|prob)$/, "")}`;
    }
    return `${f.ui.label} ${formatValue(f, v)}`;
  });
  return { text: shown.join(", "), extra: Math.max(0, mod.length - maxItems) };
}

/** Key config facts shown in every collapsed header regardless of modification. */
export function groupHeadline(config: Json, group: string): string | null {
  switch (group) {
    case "model": return getPath(config, "model.backbone");
    case "input": {
      const s = getPath(config, "input.input_size");
      const t = getPath(config, "input.test_input_size");
      return t && t !== s ? `${s}→${t}px` : `${s}px`;
    }
    case "schedule":
      return `${getPath(config, "schedule.epochs")}ep · bs ${getPath(config, "schedule.batch_size")}`;
    case "optimization":
      return `${getPath(config, "optimization.optimizer")} · lr ${formatValue(
        { type: "number" } as FieldDef, getPath(config, "optimization.lr"))}`;
    case "loss": return getPath(config, "loss.loss");
    case "validation": return getPath(config, "validation.primary_metric");
    default: return null;
  }
}
