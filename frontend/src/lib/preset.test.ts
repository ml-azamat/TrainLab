import { describe, expect, it } from "vitest";
import {
  CUSTOM_PRESET, PRESET_PATH, applyPreset, derivePreset, withDerivedPreset, type PresetDef,
} from "./preset";
import { getPath, setPath } from "./utils";

/**
 * The preset bar used to remember which button was clicked. Nothing reset that memory when
 * the config was edited, so "Balanced" stayed highlighted over a config that was no longer
 * balanced, and the run was launched tagged `preset=balanced` — which is what the Compare
 * tab filters and groups by. These tests pin that the label follows the config.
 */

const preset = (key: string, config: Record<string, any>): PresetDef => ({
  key, label: key, description: "",
  config: {
    schema_version: "1.0",
    data: { train_dir: "", val_dir: null, num_classes: null, class_names: [], val_split: 0.1 },
    augmentation: { preset: "medium", hflip: 0.5 },
    model: { backbone: "convnext_tiny", drop_path_rate: 0.1 },
    schedule: { epochs: 30, batch_size: 64 },
    checkpoint: { output_dir: "./runs", resume_from: null, save_top_k: 1 },
    tracking: {
      preset: key, enabled: true, experiment_name: "default",
      tracking_uri: "http://127.0.0.1:5050", run_name: null, tags: {},
    },
    ...config,
  },
});

const BALANCED = preset("balanced", {});
const MAX_ACCURACY = preset("max-accuracy", {
  augmentation: { preset: "heavy", hflip: 0.5 },
  model: { backbone: "convnext_base", drop_path_rate: 0.3 },
  schedule: { epochs: 120, batch_size: 32 },
  checkpoint: { output_dir: "./runs", resume_from: null, save_top_k: 3 },
});
const PRESETS = [BALANCED, MAX_ACCURACY];

/** What the form actually holds: a preset applied over a real dataset and tracker setup. */
const IN_USE = {
  data: {
    train_dir: "/data/train", val_dir: "/data/val", num_classes: 10,
    class_names: ["a", "b"], val_split: 0.1,
  },
  checkpoint: { output_dir: "/scratch/runs", resume_from: null, save_top_k: 1 },
  tracking: {
    preset: "custom", enabled: true, experiment_name: "birds",
    tracking_uri: "http://10.0.0.2:5050", run_name: "second try", tags: { owner: "me" },
  },
};

const APPLIED = applyPreset(IN_USE, BALANCED);

describe("applyPreset", () => {
  it("takes the preset's values", () => {
    const out = applyPreset(IN_USE, MAX_ACCURACY);
    expect(out.schedule).toEqual({ epochs: 120, batch_size: 32 });
    expect(out.model.backbone).toBe("convnext_base");
    expect(out.augmentation.preset).toBe("heavy");
  });

  it("keeps what a preset does not describe", () => {
    expect(APPLIED.data).toEqual(IN_USE.data);
    expect(APPLIED.checkpoint.output_dir).toBe("/scratch/runs");
    expect(APPLIED.tracking.experiment_name).toBe("birds");
    expect(APPLIED.tracking.run_name).toBe("second try");
  });

  it("still takes the preset's own checkpoint settings", () => {
    expect(applyPreset(IN_USE, MAX_ACCURACY).checkpoint.save_top_k).toBe(3);
  });

  it("labels the config with the preset", () => {
    expect(getPath(APPLIED, PRESET_PATH)).toBe("balanced");
  });

  it("does not mutate the config it is given", () => {
    const before = JSON.stringify(IN_USE);
    applyPreset(IN_USE, MAX_ACCURACY);
    expect(JSON.stringify(IN_USE)).toBe(before);
  });
});

describe("derivePreset", () => {
  it("names the preset that was just applied", () => {
    expect(derivePreset(APPLIED, PRESETS)).toBe("balanced");
    expect(derivePreset(applyPreset(IN_USE, MAX_ACCURACY), PRESETS)).toBe("max-accuracy");
  });

  it("gives the label up as soon as the config diverges", () => {
    const edited = setPath(APPLIED, "schedule.epochs", 100);
    expect(derivePreset(edited, PRESETS)).toBe(CUSTOM_PRESET);
  });

  it("gives it up for a nested edit anywhere in the recipe", () => {
    for (const [path, value] of [
      ["model.backbone", "resnet50"], ["augmentation.hflip", 0], ["data.val_split", 0.3],
      ["checkpoint.save_top_k", 5],
    ] as const) {
      expect(derivePreset(setPath(APPLIED, path, value), PRESETS)).toBe(CUSTOM_PRESET);
    }
  });

  it("keeps it through the fields a preset does not describe", () => {
    for (const [path, value] of [
      ["data.train_dir", "/elsewhere"], ["data.num_classes", 42],
      ["checkpoint.output_dir", "/tmp"], ["tracking.experiment_name", "other"],
      ["tracking.enabled", false], ["tracking.run_name", "renamed"],
    ] as const) {
      expect(derivePreset(setPath(APPLIED, path, value), PRESETS)).toBe("balanced");
    }
  });

  it("is custom for a label no preset in the payload accounts for", () => {
    expect(derivePreset(setPath(APPLIED, PRESET_PATH, "custom"), PRESETS)).toBe(CUSTOM_PRESET);
    expect(derivePreset(setPath(APPLIED, PRESET_PATH, "retired-preset"), PRESETS))
      .toBe(CUSTOM_PRESET);
  });

  it("does not retitle a config that merely matches one, matching the server", () => {
    // Demotion only: `TrainConfig._demote_stale_preset` does the same on its side.
    const unlabelled = setPath(applyPreset(IN_USE, BALANCED), PRESET_PATH, CUSTOM_PRESET);
    expect(derivePreset(unlabelled, PRESETS)).toBe(CUSTOM_PRESET);
  });

  it("comes back when the preset is applied again", () => {
    const edited = setPath(APPLIED, "schedule.epochs", 100);
    expect(derivePreset(applyPreset(edited, BALANCED), PRESETS)).toBe("balanced");
  });
});

describe("withDerivedPreset", () => {
  it("writes the answer into the config the form launches", () => {
    const edited = setPath(APPLIED, "optimization", { lr: 1e-5 });
    expect(getPath(edited, PRESET_PATH)).toBe("balanced");          // stale until committed
    expect(getPath(withDerivedPreset(edited, PRESETS), PRESET_PATH)).toBe(CUSTOM_PRESET);
  });

  it("leaves a still-true label alone", () => {
    expect(withDerivedPreset(APPLIED, PRESETS)).toEqual(APPLIED);
  });
});
