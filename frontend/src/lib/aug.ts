import type { Json } from "./api";

/**
 * The augmentation strength ladder.
 *
 * Picking a rung must move the actual controls, not just relabel the group — the preview,
 * the warnings panel and the launched run all read the individual fields, so a label on
 * its own changes nothing about training. The value tables come from the server
 * (`GET /api/presets` -> `aug_presets`, generated from `AUG_PRESETS` in trainlab/config.py)
 * so the client cannot hold a stale copy of what "heavy" means.
 */

export interface AugPresetDef {
  key: string;
  values: Json;
}

export const AUG_PRESET_PATH = "augmentation.preset";

/**
 * Config with the named rung applied: its values written into the augmentation group and
 * the label set. `custom` (and any key the server didn't send) sets the label only —
 * "custom" describes whatever is already there rather than a set of values.
 */
export function applyAugPreset(config: Json, key: string, presets: AugPresetDef[]): Json {
  const found = presets.find((p) => p.key === key);
  return {
    ...config,
    augmentation: { ...config.augmentation, ...(found?.values ?? {}), preset: key },
  };
}

/**
 * True when `path` is an individual augmentation control, i.e. one whose edit means the
 * rung no longer describes the group. Mirrors the server-side demotion in
 * `AugmentationConfig._expand_preset`.
 */
export function demotesAugPreset(path: string): boolean {
  return path.startsWith("augmentation.") && path !== AUG_PRESET_PATH;
}
