import * as React from "react";
import { Download, Play, RotateCcw, Upload } from "lucide-react";
import {
  api, type AugPresetDef, type FieldDef, type GroupDef, type Json, type ValidateResult,
} from "@/lib/api";
import { AUG_PRESET_PATH, applyAugPreset, demotesAugPreset } from "@/lib/aug";
import {
  PRESET_PATH, applyPreset, derivePreset, withDerivedPreset, type PresetDef,
} from "@/lib/preset";
import { getPath, setPath } from "@/lib/utils";
import { ConfigForm } from "@/components/ConfigForm";
import { AugPreview } from "@/components/AugPreview";
import { DatasetPanel, WarningsPanel } from "@/components/WarningsPanel";
import { ResizableSplit } from "@/components/ResizableSplit";
import { RunConsole } from "@/components/RunConsole";
import { Badge, Button, Dialog, Panel, Tooltip } from "@/components/ui/primitives";

interface Props {
  fields: FieldDef[];
  groups: GroupDef[];
  defaults: Json;
  config: Json;
  setConfig: React.Dispatch<React.SetStateAction<Json>>;
  presets: PresetDef[];
  augPresets: AugPresetDef[];
  onLaunched: (runId: string) => void;
  activeRunId: string | null;
  setActiveRunId: (id: string | null) => void;
  notice: string | null;
  setNotice: (n: string | null) => void;
}

export function ConfigureTab({
  fields, groups, defaults, config, setConfig, presets, augPresets, onLaunched,
  activeRunId, setActiveRunId, notice, setNotice,
}: Props) {
  const [validation, setValidation] = React.useState<ValidateResult | null>(null);
  const [dataset, setDataset] = React.useState<Json | null>(null);
  const [estimate, setEstimate] = React.useState<Json | null>(null);
  const [yamlOpen, setYamlOpen] = React.useState(false);
  const [yamlText, setYamlText] = React.useState("");
  const [launching, setLaunching] = React.useState(false);
  const [launchError, setLaunchError] = React.useState<string | null>(null);

  // Every edit re-derives `tracking.preset`, because that label is what the run is tagged
  // and compared by: it has to describe the config as it stands, not the preset button
  // that was clicked before the config diverged from it.
  const commit = React.useCallback(
    (next: Json) => withDerivedPreset(next, presets), [presets]);

  const change = React.useCallback((path: string, value: any) => {
    setConfig((prev: Json) => {
      // Picking a rung of the strength ladder sets the whole group, not just the label.
      if (path === AUG_PRESET_PATH) return commit(applyAugPreset(prev, value, augPresets));
      const next = setPath(prev, path, value);
      // Touching an individual augmentation control means the preset no longer describes
      // it. The server demotes the label the same way, so the form must not disagree.
      return commit(demotesAugPreset(path) ? setPath(next, AUG_PRESET_PATH, "custom") : next);
    });
  }, [setConfig, augPresets, commit]);

  // Reverting a field is just setting it to its default, so it goes through the same rules
  // — otherwise reverting one augmentation knob left the preset label claiming the group.
  const resetField = (path: string) => change(path, getPath(defaults, path));

  const resetGroup = (group: string) => {
    let next = config;
    for (const f of fields.filter((f) => f.group === group)) {
      next = setPath(next, f.path, getPath(defaults, f.path));
    }
    // Data paths are not "settings" — wiping them on a group reset would be hostile.
    if (group === "data") {
      next = setPath(next, "data.train_dir", config.data.train_dir);
      next = setPath(next, "data.val_dir", config.data.val_dir);
    }
    // The preset label is derived from every other field rather than set, so resetting the
    // Tracking group it lives in must not be a way to throw it away. Carry it over and let
    // `commit` re-answer it — which still gives it up if this reset changed the recipe.
    next = setPath(next, PRESET_PATH, getPath(config, PRESET_PATH));
    setConfig(commit(next));
  };

  const selectPreset = (key: string) => {
    const p = presets.find((x) => x.key === key);
    if (p) setConfig(applyPreset(config, p));
  };

  // Reads the config rather than remembering the last click, so it goes dark as soon as
  // an edit makes the config something the preset no longer describes.
  const activePreset = React.useMemo(
    () => derivePreset(config, presets), [config, presets]);

  // Validation + estimate follow every edit (debounced).
  React.useEffect(() => {
    const t = setTimeout(() => {
      api.validate(config, dataset?.num_images, dataset?.imbalance_ratio)
        .then(setValidation).catch(() => {});
      api.estimate(config).then(setEstimate).catch(() => {});
    }, 250);
    return () => clearTimeout(t);
  }, [config, dataset?.num_images, dataset?.imbalance_ratio]);

  // Dataset introspection only when the source actually changes.
  React.useEffect(() => {
    if (!config.data?.train_dir) { setDataset(null); return; }
    const t = setTimeout(() => {
      api.inspectDataset(config).then((d) => {
        setDataset(d);
        if (d.ok) {
          // No `commit`: the detected class list is not a setting, so no preset claims
          // to describe it and writing it back cannot cost the config its label.
          setConfig((prev: Json) => {
            const withN = setPath(prev, "data.num_classes", d.num_classes);
            return setPath(withN, "data.class_names", d.class_names);
          });
        }
      }).catch(() => setDataset(null));
    }, 400);
    return () => clearTimeout(t);
  }, [config.data?.train_dir, config.data?.dataset_format]); // eslint-disable-line

  const launch = async () => {
    setLaunching(true); setLaunchError(null);
    try {
      // `tracking.preset` is already current — it is re-derived on every edit — so the
      // config is launched as it stands. Overriding it here is what tagged edited runs
      // with the preset they had merely started from.
      const run = await api.startRun(config);
      onLaunched(run.id);
      setActiveRunId(run.id);
    } catch (e: any) {
      setLaunchError(e.message ?? String(e));
    } finally {
      setLaunching(false);
    }
  };

  const openYaml = async () => {
    setYamlText(await api.toYaml(config));
    setYamlOpen(true);
  };

  const importYaml = async () => {
    try {
      const { config: c, dropped_fields } = await api.fromYaml(yamlText);
      setConfig(commit(c));
      // The backend loads leniently and reports what it discarded; hiding that made
      // an import that silently lost keys look like a complete load.
      setNotice(dropped_fields?.length
        ? `Imported YAML: ${dropped_fields.length} unknown field(s) dropped — ${dropped_fields.join(", ")}`
        : null);
      setYamlOpen(false);
    } catch (e: any) {
      setLaunchError(e.message ?? String(e));
    }
  };

  const blocked = (validation && !validation.valid) || !config.data?.train_dir;

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Preset bar */}
      <div className="flex flex-wrap items-center gap-2 border-b border-[var(--color-border)] px-4 py-2">
        <span className="text-[10px] uppercase tracking-wide text-[var(--color-muted)]">Preset</span>
        {presets.map((p) => (
          <Tooltip key={p.key} content={p.description} side="bottom">
            <button
              onClick={() => selectPreset(p.key)}
              className={
                "rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors " +
                (activePreset === p.key
                  ? "bg-[var(--color-accent)] text-white"
                  : "border border-[var(--color-border)] text-[var(--color-muted)] hover:text-[var(--color-fg)]")
              }
            >
              {p.label}
            </button>
          </Tooltip>
        ))}

        <div className="ml-auto flex items-center gap-1.5">
          <Tooltip content="Reset every group to schema defaults" side="bottom">
            <Button
              variant="ghost" size="sm"
              onClick={() => setConfig(commit({ ...defaults, data: config.data }))}
            >
              <RotateCcw className="h-3 w-3" /> Reset all
            </Button>
          </Tooltip>
          <Button variant="ghost" size="sm" onClick={openYaml}>
            <Download className="h-3 w-3" /> YAML
          </Button>
          <Button
            variant="default" size="md" disabled={launching || Boolean(blocked)}
            onClick={launch}
            title={!config.data?.train_dir ? "Set a train directory first" : undefined}
          >
            <Play className="h-3 w-3" /> {launching ? "Starting…" : "Train"}
          </Button>
        </div>
      </div>

      {launchError && (
        <div className="border-b border-[var(--color-danger)]/30 bg-[var(--color-danger)]/10 px-4 py-1.5 text-[11px] text-[var(--color-danger)]">
          {launchError}
        </div>
      )}

      {notice && (
        <div className="flex items-center gap-2 border-b border-[var(--color-modified)]/30 bg-[var(--color-modified)]/10 px-4 py-1.5 text-[11px] text-[var(--color-modified)]">
          <span className="min-w-0 flex-1 truncate" title={notice}>{notice}</span>
          <button className="shrink-0 opacity-70 hover:opacity-100" onClick={() => setNotice(null)}>
            dismiss
          </button>
        </div>
      )}

      <ResizableSplit
        storageKey="configure" initial={1.55 / 2.55} minLeft={420} minRight={340}
        className="min-h-0 flex-1 overflow-hidden p-3"
      >
        <div className="min-h-0 overflow-y-auto pr-1">
          <ConfigForm
            fields={fields} groups={groups} config={config} defaults={defaults}
            resolved={validation?.resolved}
            onChange={change} onResetField={resetField} onResetGroup={resetGroup}
          />
          {validation?.run_name && (
            <div className="mt-2 flex items-center gap-2 px-1 text-[10px] text-[var(--color-muted)]">
              <span>auto run name:</span>
              <span className="mono text-[var(--color-fg)]">{validation.run_name}</span>
            </div>
          )}
        </div>

        <div className="min-h-0 space-y-3 overflow-y-auto pr-1">
          {activeRunId && <RunConsole runId={activeRunId} onClose={() => setActiveRunId(null)} />}
          <AugPreview config={config} datasetOk={Boolean(dataset?.ok)} />
          <WarningsPanel
            result={validation} config={config} estimate={estimate}
            onApplyFix={(p, v) => change(p, v)}
          />
          <DatasetPanel info={dataset} />
        </div>
      </ResizableSplit>

      <Dialog open={yamlOpen} onOpenChange={setYamlOpen} title="Config YAML" wide>
        <textarea
          value={yamlText} onChange={(e) => setYamlText(e.target.value)}
          spellCheck={false}
          className="h-[55vh] w-full resize-none rounded border border-[var(--color-border)] bg-[var(--color-bg)] p-2 mono text-[11px] outline-none focus:border-[var(--color-accent)]"
        />
        <div className="mt-3 flex items-center gap-2">
          <Badge>train.py --config run.yaml</Badge>
          <div className="ml-auto flex gap-2">
            <Button onClick={() => navigator.clipboard.writeText(yamlText)}>Copy</Button>
            <Button variant="default" onClick={importYaml}>
              <Upload className="h-3 w-3" /> Load into form
            </Button>
          </div>
        </div>
      </Dialog>
    </div>
  );
}

export function EmptyState({ children }: { children: React.ReactNode }) {
  return (
    <Panel>
      <div className="py-10 text-center text-[11px] text-[var(--color-muted)]">{children}</div>
    </Panel>
  );
}
