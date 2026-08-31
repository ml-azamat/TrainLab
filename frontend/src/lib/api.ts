export type Json = Record<string, any>;

export interface UiMeta {
  label: string;
  tooltip: string;
  advanced: boolean;
  widget: string | null;
  step: number | null;
  unit: string | null;
  showIf: string | null;
  disableIf: string | null;
  summaryPriority: number;
}

export interface FieldDef {
  path: string;          // dotted, e.g. "optimization.lr"
  name: string;          // leaf name
  group: string;
  ui: UiMeta;
  type: "number" | "integer" | "boolean" | "string" | "enum" | "array" | "tuple" | "unknown";
  enumValues?: string[];
  itemEnum?: string[];   // for multi-select arrays
  itemType?: "number" | "integer";   // element type of a numeric list
  min?: number;
  max?: number;
  default: any;
}

/** One rung of the augmentation strength ladder, with the values it stands for. */
export interface AugPresetDef {
  key: string;
  values: Json;
}

export interface GroupDef {
  key: string;
  title: string;
  description: string;
  expandedByDefault: boolean;
}

export interface Warning {
  severity: "error" | "warning" | "info";
  field: string | null;
  message: string;
  fix: string | null;
}

export interface ValidateResult {
  valid: boolean;
  errors: { field: string; message: string }[];
  warnings: Warning[];
  resolved?: {
    loss: string;
    loss_autoswitched: boolean;
    test_input_size: number;
    effective_batch_size: number;
    device: string;
  };
  run_name?: string;
  tags?: Record<string, string>;
}

export interface RunSummary {
  id: string;
  status: string;
  run_name: string;
  mlflow_run_id: string | null;
  started_at: number;
  ended_at: number | null;
  duration_s: number;
  latest: Json;
  error: string | null;
  sweep_id: string | null;
  output_dir: string | null;
  backbone?: string;
  epochs?: number;
}

export interface TrackedRun {
  run_id: string;
  name: string;
  status: string;
  start_time: number;
  end_time: number;
  duration_s: number;
  params: Record<string, string>;
  metrics: Record<string, number>;
  tags: Record<string, string>;
}

const BASE = "/api";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }
  const ct = res.headers.get("content-type") ?? "";
  return (ct.includes("json") ? res.json() : res.text()) as Promise<T>;
}

const post = <T,>(p: string, body: unknown) =>
  req<T>(p, { method: "POST", body: JSON.stringify(body) });

/** Query string from the params that have a value, each encoded. */
function qs(params: Record<string, string | number | undefined | null>): string {
  const pairs = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`);
  return pairs.length ? `?${pairs.join("&")}` : "";
}

export const api = {
  schema: () => req<{ schema: Json; defaults: Json; groups: GroupDef[] }>("/schema"),
  presets: () => req<{
    presets: { key: string; label: string; description: string; config: Json }[];
    aug_presets: AugPresetDef[];
  }>("/presets"),
  validate: (config: Json, n_train?: number, imbalance_ratio?: number) =>
    post<ValidateResult>("/validate", { config, n_train, imbalance_ratio }),
  toYaml: (config: Json) => post<string>("/config/yaml", { config }),
  fromYaml: (yaml: string) =>
    post<{ config: Json; dropped_fields: string[] }>("/config/parse", { yaml }),
  lrSuggestion: (optimizer: string, batch_size: number, rule: string) =>
    req<{ suggested_lr: number; rule: string; explanation: string }>(
      `/lr-suggestion?optimizer=${optimizer}&batch_size=${batch_size}&rule=${rule}`),

  backbones: (q: string, pretrainedOnly: boolean, limit = 60) =>
    req<{ total: number; items: Json[] }>(
      `/backbones?q=${encodeURIComponent(q)}&pretrained_only=${pretrainedOnly}&limit=${limit}`),
  backbone: (name: string) => req<Json>(`/backbones/${encodeURIComponent(name)}`),
  estimate: (config: Json) => post<Json>("/estimate", { config }),

  inspectDataset: (config: Json) => post<Json>("/dataset/inspect", { config }),
  previewAug: (config: Json, n = 8, seed = 0) =>
    post<Json>("/preview/augmentation", { config, n, seed }),
  previewEval: (config: Json, n = 4, seed = 0) =>
    post<Json>("/preview/eval", { config, n, seed }),

  startRun: (config: Json) => post<RunSummary>("/runs", { config }),
  runs: () => req<{ runs: RunSummary[]; active: number }>("/runs"),
  run: (id: string) => req<RunSummary & { log_tail: string[]; events: Json[] }>(`/runs/${id}`),
  cancelRun: (id: string) => post<{ cancelled: boolean }>(`/runs/${id}/cancel`, {}),

  // Every tracker read carries the URI the form is configured with. Without it the
  // server answers from its own default, so pointing the app at another MLflow moved
  // where runs were recorded while the comparison views kept reading the old one.
  experiments: (uri?: string) =>
    req<{ experiments: { id: string; name: string }[] }>(
      `/tracker/experiments${qs({ tracking_uri: uri })}`),
  trackerRuns: (experiment: string, uri?: string) =>
    req<{ runs: TrackedRun[]; varying_params: string[]; metrics: string[] }>(
      `/tracker/runs${qs({ experiment, tracking_uri: uri })}`),
  parallel: (experiment: string, metric: string, params?: string[], uri?: string) =>
    req<Json>(`/tracker/parallel${qs({
      experiment, metric, params: params?.join(","), tracking_uri: uri })}`),
  diff: (a: string, b: string, uri?: string) =>
    req<Json>(`/tracker/diff${qs({ a, b, tracking_uri: uri })}`),
  history: (runId: string, key: string, uri?: string) =>
    req<{ history: { step: number; value: number }[] }>(
      `/tracker/history${qs({ run_id: runId, key, tracking_uri: uri })}`),
  clone: (runId: string, uri?: string) =>
    req<{ config: Json; cloned_from: string; dropped_fields: string[] }>(
      `/tracker/clone${qs({ run_id: runId, tracking_uri: uri })}`),
  artifactUrl: (runId: string, path: string, uri?: string) =>
    `${BASE}/tracker/artifact${qs({ run_id: runId, path, tracking_uri: uri })}`,

  startSweep: (body: Json) => post<Json>("/sweeps", body),
  sweeps: () => req<{ sweeps: Json[] }>("/sweeps"),
  sweep: (id: string) => req<Json>(`/sweeps/${id}`),
  stopSweep: (id: string) => post<Json>(`/sweeps/${id}/stop`, {}),
};

/** Subscribe to a run's SSE stream. Returns an unsubscribe function. */
export function streamRun(runId: string, onMessage: (m: Json) => void): () => void {
  const es = new EventSource(`${BASE}/runs/${runId}/stream`);
  es.onmessage = (e) => {
    try { onMessage(JSON.parse(e.data)); } catch { /* keepalive comment */ }
  };
  es.onerror = () => es.close();
  return () => es.close();
}
