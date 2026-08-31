import * as React from "react";
import { Activity, FlaskConical, Sliders, Sparkles } from "lucide-react";
import { api, type AugPresetDef, type FieldDef, type GroupDef, type Json } from "@/lib/api";
import { type PresetDef } from "@/lib/preset";
import { buildFields } from "@/lib/schema";
import { ConfigureTab } from "@/tabs/ConfigureTab";
import { RunsTab } from "@/tabs/RunsTab";
import { CompareTab } from "@/tabs/CompareTab";
import { SweepsTab } from "@/tabs/SweepsTab";
import {
  Badge, Tabs, TabsContent, TabsList, TabsTrigger, TooltipProvider,
} from "@/components/ui/primitives";

export default function App() {
  const [fields, setFields] = React.useState<FieldDef[]>([]);
  const [groups, setGroups] = React.useState<GroupDef[]>([]);
  const [defaults, setDefaults] = React.useState<Json | null>(null);
  const [config, setConfig] = React.useState<Json | null>(null);
  const [presets, setPresets] = React.useState<PresetDef[]>([]);
  const [augPresets, setAugPresets] = React.useState<AugPresetDef[]>([]);
  const [tab, setTab] = React.useState("configure");
  const [activeRunId, setActiveRunId] = React.useState<string | null>(null);
  // Non-fatal outcomes worth telling the user about (e.g. obsolete keys dropped when
  // importing/cloning an old config). Rendered as a dismissible strip in Configure.
  const [notice, setNotice] = React.useState<string | null>(null);
  const [device, setDevice] = React.useState<string>("");
  const [bootError, setBootError] = React.useState<string | null>(null);

  React.useEffect(() => {
    Promise.all([api.schema(), api.presets(), api.runs().catch(() => null)])
      .then(([s, p]) => {
        setFields(buildFields(s.schema, s.defaults));
        setGroups(s.groups);
        setDefaults(s.defaults);
        // Open on the recommended preset rather than raw schema defaults.
        const balanced = p.presets.find((x) => x.key === "balanced");
        setConfig(balanced ? balanced.config : s.defaults);
        setPresets(p.presets);
        setAugPresets(p.aug_presets ?? []);
      })
      .catch((e) => setBootError(e.message ?? String(e)));
    api.validate({}).then(() => {}).catch(() => {});
    fetch("/api/health").then((r) => r.json()).then((h) => setDevice(h.device)).catch(() => {});
  }, []);

  if (bootError) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-center">
        <div>
          <div className="text-sm font-semibold text-[var(--color-danger)]">
            Can't reach the TrainLab API
          </div>
          <div className="mt-2 text-[11px] text-[var(--color-muted)]">{bootError}</div>
          <div className="mt-3 text-[11px] text-[var(--color-muted)]">
            Start it with <span className="mono">make api</span>, then reload.
          </div>
        </div>
      </div>
    );
  }

  if (!config || !defaults) {
    return (
      <div className="flex h-full items-center justify-center text-[11px] text-[var(--color-muted)]">
        Loading schema…
      </div>
    );
  }

  return (
    <TooltipProvider>
      <Tabs value={tab} onValueChange={setTab} className="flex h-full min-h-0 flex-col">
        <header className="flex items-center gap-3 border-b border-[var(--color-border)] px-4 py-2">
          <div className="flex items-baseline gap-1.5">
            <span className="text-sm font-bold tracking-tight">TrainLab</span>
            <span className="text-[10px] text-[var(--color-muted)]">image classification</span>
          </div>

          <TabsList>
            <TabsTrigger value="configure">
              <span className="flex items-center gap-1.5"><Sliders className="h-3 w-3" /> Configure</span>
            </TabsTrigger>
            <TabsTrigger value="runs">
              <span className="flex items-center gap-1.5"><Activity className="h-3 w-3" /> Runs</span>
            </TabsTrigger>
            <TabsTrigger value="compare">
              <span className="flex items-center gap-1.5"><FlaskConical className="h-3 w-3" /> Compare</span>
            </TabsTrigger>
            <TabsTrigger value="sweeps">
              <span className="flex items-center gap-1.5"><Sparkles className="h-3 w-3" /> Sweeps</span>
            </TabsTrigger>
          </TabsList>

          <div className="ml-auto flex items-center gap-2">
            {device && <Badge variant={device === "cpu" ? "warn" : "ok"}>{device}</Badge>}
            {/* Follows the configured tracking URI rather than a hardcoded port, so
                changing it in the form (or the schema default) keeps this link correct. */}
            <a
              href={config.tracking?.tracking_uri ?? defaults.tracking?.tracking_uri}
              target="_blank" rel="noreferrer"
              className="text-[10px] text-[var(--color-muted)] hover:text-[var(--color-accent)]"
            >
              MLflow ↗
            </a>
          </div>
        </header>

        <TabsContent value="configure" className="min-h-0 flex-1 outline-none">
          <ConfigureTab
            fields={fields} groups={groups} defaults={defaults}
            config={config} setConfig={setConfig as any} presets={presets}
            augPresets={augPresets}
            onLaunched={() => setTab("configure")}
            activeRunId={activeRunId} setActiveRunId={setActiveRunId}
            notice={notice} setNotice={setNotice}
          />
        </TabsContent>

        <TabsContent value="runs" className="min-h-0 flex-1 outline-none">
          <RunsTab activeRunId={activeRunId} setActiveRunId={setActiveRunId} />
        </TabsContent>

        <TabsContent value="compare" className="min-h-0 flex-1 outline-none">
          <CompareTab
            initialExperiment={config.tracking?.experiment_name ?? "default"}
            trackingUri={config.tracking?.tracking_uri}
            onClone={(c, dropped) => {
              setConfig(c);
              setTab("configure");
              // Lenient loading drops keys the schema no longer knows; say so
              // instead of letting the user believe the whole run was recovered.
              setNotice(dropped?.length
                ? `Cloned config: ${dropped.length} obsolete field(s) dropped — ${dropped.join(", ")}`
                : null);
            }}
          />
        </TabsContent>

        <TabsContent value="sweeps" className="min-h-0 flex-1 outline-none">
          <SweepsTab config={config} fields={fields} />
        </TabsContent>
      </Tabs>
    </TooltipProvider>
  );
}
