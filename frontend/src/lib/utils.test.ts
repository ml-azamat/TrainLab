import { describe, expect, it } from "vitest";
import { parseNumber } from "./utils";

/**
 * The float fields (learning rate, weight decay, head init scale) could only be given whole
 * numbers: the box was re-rendered from the parsed value on every keystroke, and since
 * `Number("0.")` is a finite 0, the decimal point was swallowed the moment it was typed.
 * The fix is a box that keeps its own text plus this rule about when there is anything to
 * commit at all.
 */

describe("parseNumber", () => {
  it("parses what a float field is given", () => {
    expect(parseNumber("0.05")).toBe(0.05);
    expect(parseNumber("3e-4")).toBe(0.0003);
    expect(parseNumber("0.001")).toBe(0.001);
    expect(parseNumber("-0.5")).toBe(-0.5);
  });

  it("commits nothing while the text is still on its way to being a number", () => {
    // The keystroke that used to destroy the value: committing 0 here rewrote the box.
    for (const partial of ["3e-", "-", "e5", "1e", "abc", "0.0.1"]) {
      expect(parseNumber(partial), partial).toBeUndefined();
    }
  });

  it("does keep a trailing point, which parses to a number the box goes on refining", () => {
    // "0." commits 0 — harmless, because the box shows the text rather than the value, so
    // the next keystroke makes "0.5". It is the rewrite that was the bug, not the parse.
    expect(parseNumber("0.")).toBe(0);
    expect(parseNumber("5.")).toBe(5);
  });

  it("clears the value for an empty box", () => {
    expect(parseNumber("")).toBeNull();
    expect(parseNumber("   ")).toBeNull();
  });

  it("rounds only where the schema says integer", () => {
    expect(parseNumber("30.6", true)).toBe(31);
    expect(parseNumber("30.6", false)).toBe(30.6);
    expect(parseNumber("1e2", true)).toBe(100);
  });

  it("accepts a comma as the decimal separator", () => {
    // What the keyboard offers under comma-decimal locales; `Number("0,05")` is NaN, and
    // the old handler wrote that raw string into a field the schema types as a float.
    expect(parseNumber("0,05")).toBe(0.05);
  });
});
