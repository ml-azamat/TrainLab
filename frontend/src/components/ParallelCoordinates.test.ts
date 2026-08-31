import { describe, expect, it } from "vitest";
import { categoryPos } from "./ParallelCoordinates";
import { metricHigherIsBetter } from "@/lib/utils";

/**
 * Degenerate cases in the hand-rolled plot. Each of these produced a chart that looked
 * fine but told you the wrong thing.
 */

describe("categoryPos", () => {
  it("centres a single category", () => {
    // The line used to be drawn at 0 and the label at 0.5, because the `n === 0` guard
    // sat behind a `Math.max(1, …)` that made it unreachable.
    expect(categoryPos(["adamw"], 0)).toBe(0.5);
  });

  it("spreads multiple categories across the full axis", () => {
    expect(categoryPos(["a", "b", "c"], 0)).toBe(0);
    expect(categoryPos(["a", "b", "c"], 1)).toBe(0.5);
    expect(categoryPos(["a", "b", "c"], 2)).toBe(1);
  });

  it("puts the two endpoints of a two-category axis at the extremes", () => {
    expect(categoryPos(["off", "on"], 0)).toBe(0);
    expect(categoryPos(["off", "on"], 1)).toBe(1);
  });
});

/**
 * The polyline builder: a run missing a value on the FIRST axis used to emit a path
 * starting with "L", which SVG rejects outright — the whole line vanished rather than
 * showing a gap. This replicates the builder to pin the command sequence.
 */
function buildPath(points: (string | null)[]): string {
  return points
    .filter((p): p is string => p !== null)
    .map((p, i) => `${i === 0 ? "M" : "L"}${p}`)
    .join(" ");
}

describe("path construction", () => {
  it("starts with M even when the first axis has no value", () => {
    const d = buildPath([null, "10,20", "30,40"]);
    expect(d.startsWith("M")).toBe(true);
    expect(d).toBe("M10,20 L30,40");
  });

  it("is unchanged when every axis has a value", () => {
    expect(buildPath(["0,0", "1,1", "2,2"])).toBe("M0,0 L1,1 L2,2");
  });

  it("produces an empty path when nothing is plottable", () => {
    expect(buildPath([null, null])).toBe("");
  });

  it("handles a single point", () => {
    expect(buildPath(["5,5"])).toBe("M5,5");
  });
});

describe("colour direction", () => {
  it("inverts the quality scale for lower-is-better metrics", () => {
    const frac = 0.9;                       // near the top of the value range
    const good = (hib: boolean) => (hib ? frac : 1 - frac);
    expect(good(metricHigherIsBetter("acc_at_1"))).toBeCloseTo(0.9);
    // A high val_loss is a BAD run, so it must colour as one.
    expect(good(metricHigherIsBetter("val_loss"))).toBeCloseTo(0.1);
  });
});

describe("metricHigherIsBetter", () => {
  it.each([
    ["acc_at_1", true],
    ["acc@1", true],
    ["macro-F1", true],
    ["auroc", true],
    ["val_loss", false],
    ["train_loss", false],
    ["ece", false],
    ["ema/val_loss", false],
    ["ema/acc_at_1", true],
  ])("%s -> %s", (key, expected) => {
    expect(metricHigherIsBetter(key)).toBe(expected);
  });
});
