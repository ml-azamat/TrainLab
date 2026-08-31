/**
 * Tiny evaluator for the `showIf` / `disableIf` expressions declared in the Python schema.
 *
 * Deliberately NOT `eval` / `new Function`: these strings come from the API, and a form
 * that executes arbitrary server-supplied JavaScript is a needless liability.
 *
 *   expr    := or
 *   or      := and ('or' and)*
 *   and     := not ('and' not)*
 *   not     := 'not' not | cmp
 *   cmp     := operand (('=='|'!='|'>='|'<='|'>'|'<') operand
 *                      | 'not'? 'in' ('(' list ')' | operand))?
 *   operand := '(' expr ')' | ident | number | 'string' | true | false | null
 *
 * Parentheses, `not` and `not in` are supported because they are the forms a schema
 * author reaches for. They used to parse as garbage that evaluated to a plausible
 * boolean — `(a) or (b)` was always false, `not x` was always false, `x not in (...)`
 * was always true — which silently hid or showed controls with nothing to indicate a
 * problem.
 *
 * Errors are reported rather than swallowed: `evalExprChecked` returns the failure so
 * callers can decide, and `evalExpr` keeps a safe default for render paths.
 */

type Token = { t: "id" | "num" | "str" | "op" | "punc" | "kw"; v: string };

const KEYWORDS = new Set(["in", "and", "or", "not", "true", "false", "null", "None"]);

export class ExprError extends Error {}

function tokenize(src: string): Token[] {
  const out: Token[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (/\s/.test(c)) { i++; continue; }
    if (c === "'" || c === '"') {
      let j = i + 1;
      while (j < src.length && src[j] !== c) j++;
      if (j >= src.length) throw new ExprError(`unterminated string in ${JSON.stringify(src)}`);
      out.push({ t: "str", v: src.slice(i + 1, j) });
      i = j + 1;
      continue;
    }
    if (/[0-9]/.test(c) || (c === "-" && /[0-9]/.test(src[i + 1] ?? ""))) {
      let j = i + 1;
      while (j < src.length && /[0-9.]/.test(src[j])) j++;
      // Exponent, and only then may a sign follow: `1e-5` is one number, `1-5` is not.
      if (j < src.length && /[eE]/.test(src[j])) {
        let k = j + 1;
        if (k < src.length && /[+-]/.test(src[k])) k++;
        if (k < src.length && /[0-9]/.test(src[k])) {
          while (k < src.length && /[0-9]/.test(src[k])) k++;
          j = k;
        }
      }
      out.push({ t: "num", v: src.slice(i, j) });
      i = j;
      continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      let j = i;
      while (j < src.length && /[A-Za-z0-9_.]/.test(src[j])) j++;
      const word = src.slice(i, j);
      out.push({ t: KEYWORDS.has(word) ? "kw" : "id", v: word });
      i = j;
      continue;
    }
    if ("()".includes(c) || c === ",") { out.push({ t: "punc", v: c }); i++; continue; }
    const two = src.slice(i, i + 2);
    if (["==", "!=", ">=", "<="].includes(two)) { out.push({ t: "op", v: two }); i += 2; continue; }
    if (c === ">" || c === "<") { out.push({ t: "op", v: c }); i++; continue; }
    throw new ExprError(`unexpected character ${JSON.stringify(c)} in ${JSON.stringify(src)}`);
  }
  return out;
}

/** Identifiers an expression reads, so callers can check them against the schema. */
export function referencedIdents(src: string | null | undefined): string[] {
  if (!src) return [];
  try {
    return [...new Set(tokenize(src).filter((t) => t.t === "id").map((t) => t.v))];
  } catch {
    return [];
  }
}

function looseEq(a: any, b: any): boolean {
  if (a === b) return true;
  if (a == null || b == null) return a == null && b == null;
  // Enum values arrive as strings; booleans and numbers may not be normalised.
  if (typeof a === "boolean" || typeof b === "boolean") return Boolean(a) === Boolean(b);
  return String(a) === String(b);
}

function compare(op: string, left: any, right: any): boolean {
  switch (op) {
    case "==": return looseEq(left, right);
    case "!=": return !looseEq(left, right);
    case ">": return Number(left) > Number(right);
    case "<": return Number(left) < Number(right);
    case ">=": return Number(left) >= Number(right);
    case "<=": return Number(left) <= Number(right);
    default: throw new ExprError(`unknown operator '${op}'`);
  }
}

/**
 * Evaluate, throwing `ExprError` on anything malformed.
 *
 * `strictIdents` additionally rejects identifiers absent from `scope`, which is how a
 * `showIf` left behind by a renamed field is caught instead of silently evaluating to
 * false and erasing the control from the form.
 */
export function evalExprChecked(
  src: string | null | undefined,
  scope: Record<string, any>,
  opts: { strictIdents?: boolean } = {},
): boolean {
  if (src == null || src.trim() === "") return true;
  const toks = tokenize(src);
  if (toks.length === 0) throw new ExprError(`expression ${JSON.stringify(src)} is empty`);
  let pos = 0;
  const peek = () => toks[pos];
  const expect = (v: string) => {
    if (peek()?.v !== v) throw new ExprError(`expected '${v}' in ${JSON.stringify(src)}`);
    pos++;
  };

  function operand(): any {
    const tok = toks[pos];
    if (!tok) throw new ExprError(`unexpected end of ${JSON.stringify(src)}`);
    if (tok.t === "punc" && tok.v === "(") {
      pos++;
      const v = parseOr();
      expect(")");
      return v;
    }
    pos++;
    switch (tok.t) {
      case "num": return Number(tok.v);
      case "str": return tok.v;
      case "kw":
        if (tok.v === "true") return true;
        if (tok.v === "false") return false;
        if (tok.v === "null" || tok.v === "None") return null;
        throw new ExprError(`'${tok.v}' is not a value in ${JSON.stringify(src)}`);
      case "id":
        if (opts.strictIdents && !(tok.v in scope)) {
          throw new ExprError(`unknown field '${tok.v}' in ${JSON.stringify(src)}`);
        }
        return scope[tok.v];
      default:
        throw new ExprError(`unexpected '${tok.v}' in ${JSON.stringify(src)}`);
    }
  }

  function parseCmp(): boolean {
    const left = operand();
    const op = peek();
    if (!op) return Boolean(left);

    // `not in` — the negation binds to the membership test, not to the operand.
    let negate = false;
    let cursor = op;
    if (cursor.t === "kw" && cursor.v === "not" && toks[pos + 1]?.v === "in") {
      negate = true;
      pos++;
      cursor = peek()!;
    }
    if (cursor.t === "kw" && cursor.v === "in") {
      pos++;
      let values: any[];
      if (peek()?.v === "(") {
        pos++;
        values = [];
        while (peek() && peek().v !== ")") {
          if (peek().v === ",") { pos++; continue; }
          values.push(operand());
        }
        expect(")");
      } else {
        // `'x' in some_field` — membership in a multi-select's value, which is how a
        // control declares that it only applies when a particular metric is checked.
        const right = operand();
        if (!Array.isArray(right)) {
          throw new ExprError(
            `'in' needs a list or a multi-valued field on the right in ${JSON.stringify(src)}`);
        }
        values = right;
      }
      const hit = values.some((v) => looseEq(left, v));
      return negate ? !hit : hit;
    }

    if (op.t !== "op") return Boolean(left);
    pos++;
    return compare(op.v, left, operand());
  }

  function parseNot(): boolean {
    if (peek()?.t === "kw" && peek().v === "not") { pos++; return !parseNot(); }
    return parseCmp();
  }

  function parseAnd(): boolean {
    let v = parseNot();
    // Both sides are always evaluated: short-circuiting would leave the right-hand
    // tokens unconsumed and desynchronise the parser.
    while (peek()?.t === "kw" && peek().v === "and") { pos++; v = parseNot() && v; }
    return v;
  }

  function parseOr(): boolean {
    let v = parseAnd();
    while (peek()?.t === "kw" && peek().v === "or") { pos++; v = parseAnd() || v; }
    return v;
  }

  const value = parseOr();
  if (pos !== toks.length) {
    throw new ExprError(
      `unexpected '${toks[pos].v}' after a complete expression in ${JSON.stringify(src)}`,
    );
  }
  return value;
}

/**
 * Render-path wrapper. `fallback` is what a broken expression evaluates to.
 *
 * The right fallback differs by caller, which the previous single `return true` got
 * wrong: for `showIf` it means "show the control" (safe), but for `disableIf` the same
 * `true` DISABLES the control — the opposite of not hiding something the user needs.
 * Callers pass what safe means for them.
 */
export function evalExpr(
  src: string | null | undefined,
  scope: Record<string, any>,
  fallback = true,
): boolean {
  try {
    return evalExprChecked(src, scope);
  } catch (e) {
    if (typeof console !== "undefined") {
      console.warn(`[trainlab] bad schema expression, defaulting to ${fallback}:`, e);
    }
    return fallback;
  }
}
