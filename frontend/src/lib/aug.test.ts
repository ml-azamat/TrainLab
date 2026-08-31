import { describe, expect, it } from "vitest";
import { AUG_PRESET_PATH, applyAugPreset, demotesAugPreset, type AugPresetDef } from "./aug";
import { setPath } from "./utils";

/**
 * The augmentation preset used to be an inert label: selecting "heavy" changed the dropdown
 * and nothing else, so the preview, the warnings and the launched run all stayed on
 * whatever the sliders happened to say. These tests pin that picking a rung writes values,
 * and that editing a control gives the label up.
 */

const LADDER: AugPresetDef[] = [
  { key: "none", values: { hflip: 0, color_jitter: 0, auto_augment: "none", random_erasing_p: 0 } },
  { key: "light", values: { hflip: 0.5, color_jitter: 0, auto_augment: "none", random_erasing_p: 0 } },
  {
    key: "heavy",
    values: {
      hflip: 0.5, color_jitter: 0.4, auto_augment: "randaugment", randaugment_m: 9,
      random_erasing_p: 0.25,
    },
  },
];

const CONFIG = {
  augmentation: {
    preset: "medium", hflip: 0.5, color_jitter: 0.4, auto_augment: "randaugment",
    randaugment_m: 7, random_erasing_p: 0.1, mixup_alpha: 0,
  },
  schedule: { epochs: 30 },
};

describe("applyAugPreset", () => {
  it("writes the rung's values into the group, not just the label", () => {
    const out = applyAugPreset(CONFIG, "heavy", LADDER);
    expect(out.augmentation.preset).toBe("heavy");
    expect(out.augmentation.randaugment_m).toBe(9);
    expect(out.augmentation.random_erasing_p).toBe(0.25);
  });

  it("turns everything off for 'none'", () => {
    const out = applyAugPreset(CONFIG, "none", LADDER);
    expect(out.augmentation.hflip).toBe(0);
    expect(out.augmentation.auto_augment).toBe("none");
    expect(out.augmentation.random_erasing_p).toBe(0);
  });

  it("leaves fields the rung does not name alone", () => {
    const out = applyAugPreset(CONFIG, "light", LADDER);
    expect(out.augmentation.mixup_alpha).toBe(0);
    expect(out.schedule).toEqual({ epochs: 30 });
  });

  it("does not mutate the config it is given", () => {
    const before = JSON.stringify(CONFIG);
    applyAugPreset(CONFIG, "heavy", LADDER);
    expect(JSON.stringify(CONFIG)).toBe(before);
  });

  it("labels only for 'custom' — it describes what is already there", () => {
    const out = applyAugPreset(CONFIG, "custom", LADDER);
    expect(out.augmentation.preset).toBe("custom");
    expect(out.augmentation.randaugment_m).toBe(7);
  });

  it("falls back to labelling when the server sent no tables", () => {
    const out = applyAugPreset(CONFIG, "heavy", []);
    expect(out.augmentation.preset).toBe("heavy");
    expect(out.augmentation.randaugment_m).toBe(7);
  });
});

describe("demotesAugPreset", () => {
  it("is true for individual augmentation controls", () => {
    expect(demotesAugPreset("augmentation.hflip")).toBe(true);
    expect(demotesAugPreset("augmentation.mixup_alpha")).toBe(true);
  });

  it("is false for the preset itself and for other groups", () => {
    expect(demotesAugPreset(AUG_PRESET_PATH)).toBe(false);
    expect(demotesAugPreset("input.input_size")).toBe(false);
    expect(demotesAugPreset("schedule.epochs")).toBe(false);
  });

  it("composes with setPath the way the form applies an edit", () => {
    const edited = setPath(applyAugPreset(CONFIG, "heavy", LADDER), "augmentation.hflip", 0);
    expect(setPath(edited, AUG_PRESET_PATH, "custom").augmentation).toMatchObject({
      preset: "custom", hflip: 0, randaugment_m: 9,
    });
  });
});
