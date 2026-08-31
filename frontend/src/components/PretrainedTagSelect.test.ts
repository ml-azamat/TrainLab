import { describe, expect, it } from "vitest";
import { DEFAULT_TAG, tagOptions } from "./PretrainedTagSelect";

/**
 * The option list is fetched per backbone, so it can disagree with the value the config
 * already holds — a cloned run's tag, a backbone switched underneath it, an older timm.
 * A Radix Select whose value matches no item renders blank, so "not offered" would look
 * like "not set" while the run still trained with it.
 */

const TAGS = ["default", "fb_in1k", "fb_in22k_ft_in1k", "in12k_ft_in1k"];

describe("tagOptions", () => {
  it("offers what the catalogue returned", () => {
    expect(tagOptions(TAGS, DEFAULT_TAG)).toEqual(TAGS);
  });

  it("keeps a tag the catalogue does not list", () => {
    expect(tagOptions(TAGS, "augreg_in21k")).toEqual([...TAGS, "augreg_in21k"]);
  });

  it("does not duplicate a tag that is already listed", () => {
    expect(tagOptions(TAGS, "in12k_ft_in1k")).toEqual(TAGS);
  });

  it("is empty while nothing has been fetched, so the caller falls back to free text", () => {
    expect(tagOptions(null, DEFAULT_TAG)).toEqual([]);
    expect(tagOptions([], "fb_in1k")).toEqual([]);
  });
});
