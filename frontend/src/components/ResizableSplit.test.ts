import { describe, expect, it } from "vitest";
import { splitFraction, storedFraction } from "./ResizableSplit";

/**
 * The two pure decisions behind the divider: where a drag lands, and what to restore.
 * Both have to hold at the edges — a fraction outside the minimums produces a pane that
 * cannot render its contents, and a bad stored value would persist that state forever.
 */

const rect = { left: 100, width: 1000 };

describe("splitFraction", () => {
  it("follows the pointer in the middle of the range", () => {
    expect(splitFraction(600, rect, 200, 200)).toBeCloseTo(0.5);
    expect(splitFraction(400, rect, 200, 200)).toBeCloseTo(0.3);
  });

  it("is relative to the container, not the viewport", () => {
    expect(splitFraction(600, { left: 0, width: 1000 }, 100, 100)).toBeCloseTo(0.6);
    expect(splitFraction(600, { left: 500, width: 1000 }, 100, 100)).toBeCloseTo(0.1);
  });

  it("stops at each pane's minimum", () => {
    // Dragging far past either edge parks the divider at the last legal position.
    expect(splitFraction(-9999, rect, 300, 400)).toBeCloseTo(0.3);
    expect(splitFraction(9999, rect, 300, 400)).toBeCloseTo(0.6);
  });

  it("keeps the divider inside a container too narrow for both minimums", () => {
    // No legal position exists; the panes must not invert or the divider escape the box.
    const f = splitFraction(9999, { left: 0, width: 500 }, 400, 400);
    expect(f).toBeGreaterThan(0);
    expect(f).toBeLessThan(1);
  });

  it("does not divide by a zero width", () => {
    expect(splitFraction(50, { left: 0, width: 0 }, 300, 300)).toBe(0.5);
  });
});

describe("storedFraction", () => {
  it("restores a usable saved position", () => {
    expect(storedFraction("0.42", 0.6)).toBe(0.42);
  });

  it("falls back when there is nothing saved or it is unusable", () => {
    for (const bad of [null, "", "abc", "NaN", "Infinity", "-1", "0", "1", "0.01", "0.99"]) {
      expect(storedFraction(bad, 0.6), String(bad)).toBe(0.6);
    }
  });
});
