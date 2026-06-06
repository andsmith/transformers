/**
 * Grokking support: a held-out test set defined by regular expressions over the
 * vocabulary glyphs. Training never sees matching samples; the test set is
 * built only from matches, so the model must generalize the rule.
 *
 * Matching is case-insensitive and NOT auto-anchored — write `^…$` for a full
 * match, or leave unanchored for an open (contains) search. A sample is held
 * out if it matches ANY of the (comma-separated) filters.
 */

import type { Task } from "./types";
import { tokenChar } from "./grammar";

/** The display string a regex is matched against (the vocab glyphs). */
export function inputToGlyphs(task: Task, vocabSize: number, ids: number[]): string {
  let s = "";
  for (const id of ids) s += tokenChar(task, id, vocabSize);
  return s;
}

export interface CompiledFilters {
  regexes: RegExp[];
  /** Per-pattern error messages (parallel to the raw split, empties dropped). */
  errors: string[];
}

/** Compile a comma-separated list of patterns into case-insensitive RegExps. */
export function compileFilters(text: string): CompiledFilters {
  const regexes: RegExp[] = [];
  const errors: string[] = [];
  for (const raw of text.split(",")) {
    const pat = raw.trim();
    if (!pat) continue;
    try {
      regexes.push(new RegExp(pat, "i"));
    } catch (e) {
      errors.push(`"${pat}": ${(e as Error).message}`);
    }
  }
  return { regexes, errors };
}

export function matchesAny(regexes: RegExp[], str: string): boolean {
  for (const r of regexes) if (r.test(str)) return true;
  return false;
}

/**
 * Enumerate the whole input space and return the keys (sampleKey form) of
 * every string matching any filter. Cost depends on space size, not on how
 * rare matches are — so even a maximally constrained filter is found
 * instantly. The caller decides when enumeration is the right move (see
 * generateTestSet's enumerate-vs-sample heuristic).
 */
export function enumerateMatches(
  task: Task,
  vocabSize: number,
  minLen: number,
  maxLen: number,
  regexes: RegExp[],
): string[] {
  const glyph: string[] = [];
  for (let d = 0; d < vocabSize; d++) glyph.push(tokenChar(task, d, vocabSize));

  const keys: string[] = [];
  for (let L = minLen; L <= maxLen; L++) {
    const digits = new Array<number>(L).fill(0);
    for (;;) {
      let s = "";
      for (let i = 0; i < L; i++) s += glyph[digits[i]];
      if (matchesAny(regexes, s)) keys.push(digits.join(","));
      // odometer increment
      let i = L - 1;
      for (; i >= 0; i--) {
        if (++digits[i] < vocabSize) break;
        digits[i] = 0;
      }
      if (i < 0) break;
    }
  }
  return keys;
}

// --- random regex generator (the 🎲 button) ---

function esc(ch: string): string {
  return ch.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Generate 1–3 varied, readable patterns over the current vocabulary glyphs. */
export function randomRegexes(task: Task, vocabSize: number): string {
  const glyphs: string[] = [];
  for (let d = 0; d < vocabSize; d++) glyphs.push(esc(tokenChar(task, d, vocabSize)));
  const pick = () => glyphs[Math.floor(Math.random() * glyphs.length)];

  const templates: Array<() => string> = [
    () => `${pick()}${pick()}`, // contains pair
    () => `[${pick()}${pick()}]+`, // run of two symbols
    () => `${pick()}.*${pick()}`, // X ... Y
    () => `${pick()}{2,}`, // 2+ in a row
    () => `(.)\\1`, // any doubled symbol (backreference)
    () => `^${pick()}`, // starts with
    () => `${pick()}$`, // ends with
    () => `.${pick()}.`, // X surrounded
  ];

  const n = 1 + Math.floor(Math.random() * 3);
  const out = new Set<string>();
  let guard = 0;
  while (out.size < n && guard++ < 20) {
    out.add(templates[Math.floor(Math.random() * templates.length)]());
  }
  return [...out].join(", ");
}
