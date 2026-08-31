import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Read a dotted path out of a nested object. */
export function getPath(obj: any, path: string): any {
  return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
}

/** Immutably write a dotted path into a nested object. */
export function setPath<T>(obj: T, path: string, value: any): T {
  const parts = path.split(".");
  const clone: any = Array.isArray(obj) ? [...(obj as any)] : { ...(obj as any) };
  let node = clone;
  for (let i = 0; i < parts.length - 1; i++) {
    const k = parts[i];
    node[k] = Array.isArray(node[k]) ? [...node[k]] : { ...node[k] };
    node = node[k];
  }
  node[parts[parts.length - 1]] = value;
  return clone;
}

export function deepEqual(a: any, b: any): boolean {
  if (a === b) return true;
  if (a == null || b == null) return a === b;
  if (typeof a !== typeof b) return false;
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((x, i) => deepEqual(x, b[i]));
  }
  if (typeof a === "object") {
    const ka = Object.keys(a), kb = Object.keys(b);
    return ka.length === kb.length && ka.every((k) => deepEqual(a[k], b[k]));
  }
  return false;
}

/**
 * What a numeric text box's contents should commit.
 *
 * Three outcomes, and the third is the point: `null` for an empty box, the number when the
 * text is one, and `undefined` while the text is still on its way to being one ("0.", "3e-",
 * "-"), which must leave the current value alone. Committing `Number("0.")` — a perfectly
 * finite 0 — is what made the float fields impossible to type into: the box was rewritten
 * from the parsed value on every keystroke, so the decimal point never survived.
 *
 * A comma typed as a decimal separator is accepted; under comma-decimal locales it is what
 * the keyboard offers, and `Number("0,05")` is NaN.
 */
export function parseNumber(raw: string, integer = false): number | null | undefined {
  const t = raw.trim().replace(",", ".");
  if (t === "") return null;
  const n = Number(t);
  if (!Number.isFinite(n)) return undefined;
  return integer ? Math.round(n) : n;
}

export function fmtNum(v: unknown, digits = 4): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "—";
  if (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e6)) return v.toExponential(1);
  return v.toFixed(digits).replace(/\.?0+$/, "");
}

export function fmtParams(n?: number | null): string {
  if (!n) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  return `${(n / 1e3).toFixed(0)}K`;
}

export function fmtDuration(s?: number | null): string {
  if (s == null) return "—";
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
}

/** Human label for a dotted config path, e.g. "optimization.lr" -> "lr". */
/** Metric keys where a SMALLER number is a better result. Mirrors
 *  `LOWER_IS_BETTER_KEYS` in trainlab/config.py; keep the two in step. */
const LOWER_IS_BETTER = new Set([
  "val_loss", "train_loss", "ece", "epoch_time_s", "peak_vram_gb",
  "data_wait_s", "compute_s", "val_time_s", "ms_per_step",
]);

/**
 * Direction of a logged metric key.
 *
 * Accepts tracker-sanitised keys (`acc_at_1`) and stage prefixes (`ema/val_loss`), so
 * the leaderboard, the diff deltas and the parallel-coordinates colour scale all agree.
 */
export function metricHigherIsBetter(key: string): boolean {
  const base = key.split("/").pop() ?? key;
  // `fpr@fnr0.01` and its tracker spelling `fpr_at_fnr0.01` carry the target in the name,
  // so this family is matched by prefix rather than by set membership. Mirrors
  // `_FPR_AT_FNR_PREFIXES` in trainlab/config.py.
  if (base.startsWith("fpr@fnr") || base.startsWith("fpr_at_fnr")) return false;
  return !LOWER_IS_BETTER.has(base.replace("_at_", "@")) && !LOWER_IS_BETTER.has(base);
}

export function shortLabel(path: string): string {
  return path.split(".").slice(-1)[0];
}

export function debounce<F extends (...args: any[]) => void>(fn: F, ms: number) {
  let t: ReturnType<typeof setTimeout>;
  return (...args: Parameters<F>) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}
