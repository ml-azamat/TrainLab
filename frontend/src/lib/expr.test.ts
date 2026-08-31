import { describe, expect, it } from "vitest";
import { ExprError, evalExpr, evalExprChecked, referencedIdents } from "./expr";

/**
 * The schema's showIf/disableIf expressions decide whether a control appears at all, so
 * a mis-evaluation silently removes a hyperparameter from the form. These tests pin both
 * the grammar the schema uses and the forms it doesn't (yet), because the old parser
 * accepted those and returned a confident, wrong boolean.
 */

const SCOPE = {
  optimizer: "adamw",
  ssl_method: "none",
  ema: true,
  pretrained: false,
  epochs: 10,
  lr: 0.001,
  pseudo_label_dir: null as string | null,
  resolved_device: "mps",
  random_erasing_p: 0.25,
  freeze_policy: "none",
  zero: 0,
  empty: "",
};

describe("expressions the schema actually uses", () => {
  const cases: [string, boolean][] = [
    ["optimizer == 'adamw'", true],
    ["optimizer != 'adamw'", false],
    ["optimizer in ('adamw', 'lion')", true],
    ["optimizer in ('sgd', 'rmsprop')", false],
    ["ssl_method != 'none'", false],
    ["ssl_method == 'none'", true],
    ["ema == true", true],
    ["pretrained == true", false],
    ["pseudo_label_dir != null", false],
    ["random_erasing_p > 0", true],
    ["resolved_device != 'cuda'", true],
    ["freeze_policy != 'none'", false],
    ["epochs >= 10", true],
    ["epochs > 10", false],
    ["lr <= 0.001", true],
  ];
  it.each(cases)("%s -> %s", (src, expected) => {
    expect(evalExprChecked(src, SCOPE)).toBe(expected);
  });
});

describe("boolean structure", () => {
  it("gives 'and' tighter binding than 'or'", () => {
    // a or (b and c) — not (a or b) and c
    expect(evalExprChecked("optimizer == 'sgd' or epochs > 5 and lr < 0.01", SCOPE)).toBe(true);
    expect(evalExprChecked("epochs > 5 and lr > 1 or optimizer == 'adamw'", SCOPE)).toBe(true);
    expect(evalExprChecked("epochs > 100 and optimizer == 'adamw'", SCOPE)).toBe(false);
  });

  it("tests membership in a multi-valued field, not just a literal list", () => {
    // How a control declares "only when this metric is checked": the right-hand side is
    // the multi-select's own value.
    const scope = { metrics: ["acc@1", "fpr@fnr"] };
    expect(evalExprChecked("'fpr@fnr' in metrics", scope)).toBe(true);
    expect(evalExprChecked("'ece' in metrics", scope)).toBe(false);
    expect(evalExprChecked("'ece' not in metrics", scope)).toBe(true);
    expect(evalExprChecked("'fpr@fnr' in metrics", { metrics: [] })).toBe(false);
  });

  it("refuses membership in something that is not a list", () => {
    expect(() => evalExprChecked("'x' in optimizer", { optimizer: "adamw" })).toThrow();
  });

  it("evaluates both sides of 'and' so tokens stay in sync", () => {
    // If the parser short-circuited it would leave `lr < 0.01` unconsumed and the
    // trailing-token check would fire.
    expect(evalExprChecked("epochs > 100 and lr < 0.01", SCOPE)).toBe(false);
  });

  it("honours parentheses", () => {
    // Previously always false: '(' fell through to `Boolean(undefined)`.
    expect(evalExprChecked("(optimizer == 'sgd') or (epochs > 5)", SCOPE)).toBe(true);
    expect(evalExprChecked("(optimizer == 'adamw' or epochs > 100) and lr < 0.01", SCOPE)).toBe(true);
    expect(evalExprChecked("optimizer == 'adamw' and (epochs > 100 or lr > 1)", SCOPE)).toBe(false);
  });

  it("supports 'not'", () => {
    // Previously always false regardless of the operand.
    expect(evalExprChecked("not ssl_method == 'none'", SCOPE)).toBe(false);
    expect(evalExprChecked("not pretrained == true", SCOPE)).toBe(true);
    expect(evalExprChecked("not (epochs > 100)", SCOPE)).toBe(true);
  });

  it("supports 'not in'", () => {
    // Previously always true: `not` parsed as an identifier and short-circuited.
    expect(evalExprChecked("ssl_method not in ('mae', 'simmim')", SCOPE)).toBe(true);
    expect(evalExprChecked("optimizer not in ('adamw', 'lion')", SCOPE)).toBe(false);
  });
});

describe("value coercion", () => {
  it("compares enum strings, numbers and booleans loosely", () => {
    expect(evalExprChecked("zero == 0", SCOPE)).toBe(true);
    expect(evalExprChecked("empty == ''", SCOPE)).toBe(true);
    expect(evalExprChecked("ema != false", SCOPE)).toBe(true);
  });

  it("treats null and undefined as absent", () => {
    expect(evalExprChecked("pseudo_label_dir == null", SCOPE)).toBe(true);
    expect(evalExprChecked("pseudo_label_dir != null", { ...SCOPE, pseudo_label_dir: "/x" }))
      .toBe(true);
  });

  it("parses exponent notation as one number", () => {
    expect(evalExprChecked("lr < 1e-2", SCOPE)).toBe(true);
    expect(evalExprChecked("lr > 1e-5", SCOPE)).toBe(true);
    expect(evalExprChecked("epochs > -1", SCOPE)).toBe(true);
  });
});

describe("malformed input", () => {
  const bad = [
    "optimizer ==",
    "== 'adamw'",
    "optimizer @@@ 'adamw'",
    "optimizer == 'adamw",       // unterminated string
    "(optimizer == 'adamw'",     // unclosed paren
    "optimizer == 'adamw' extra",
    "optimizer in 'adamw'",      // `in` without a list
  ];
  it.each(bad)("throws on %s", (src) => {
    expect(() => evalExprChecked(src, SCOPE)).toThrow(ExprError);
  });

  it("rejects an unknown field under strictIdents", () => {
    // This is how a showIf left behind by a renamed field is caught, instead of
    // evaluating to false and silently erasing the control from the form.
    expect(() => evalExprChecked("renamed_away == 'x'", SCOPE, { strictIdents: true }))
      .toThrow(ExprError);
    expect(evalExprChecked("renamed_away == 'x'", SCOPE)).toBe(false);
  });

  it("treats an empty expression as 'always'", () => {
    expect(evalExprChecked("", SCOPE)).toBe(true);
    expect(evalExprChecked(null, SCOPE)).toBe(true);
    expect(evalExprChecked(undefined, SCOPE)).toBe(true);
  });
});

describe("evalExpr fallback", () => {
  it("defaults to the caller's safe value, not a fixed true", () => {
    // showIf: broken -> still show the control.
    expect(evalExpr("optimizer @@@ 'x'", SCOPE, true)).toBe(true);
    // disableIf: broken -> do NOT disable. Sharing one `true` fallback meant a malformed
    // disableIf greyed the control out, the opposite of the intent.
    expect(evalExpr("optimizer @@@ 'x'", SCOPE, false)).toBe(false);
  });

  it("still evaluates valid expressions normally", () => {
    expect(evalExpr("optimizer == 'adamw'", SCOPE, false)).toBe(true);
  });
});

describe("referencedIdents", () => {
  it("returns field names and ignores string literals and keywords", () => {
    expect(referencedIdents("optimizer in ('adamw', 'lion')")).toEqual(["optimizer"]);
    expect(referencedIdents("a == 1 and b != null").sort()).toEqual(["a", "b"]);
    expect(referencedIdents(null)).toEqual([]);
  });
});
