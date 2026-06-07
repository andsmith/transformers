/**
 * The toy grammar: how token ids map to display characters and colors, plus
 * the delimiter/distractor role assignment used by the parenthesis-matching
 * task.
 */

import type { Task } from "./types";

/** Hard cap on vocabulary size (token ids 0 .. MAX_VOCAB-1). */
export const MAX_VOCAB = 10;

/** Generic per-token display letters for the transduction tasks. */
const ALPHABET = "abcdefghij"; // MAX_VOCAB letters

/** Bracket glyphs used when displaying the parens task as characters. */
const OPEN_GLYPHS = "([{<«";
const CLOSE_GLYPHS = ")]}>»";
const DISTRACTOR_GLYPHS = ".,;:*+~";

/** Max distinct delimiter pair kinds (limited by available glyph pairs). */
export const MAX_DELIMS = OPEN_GLYPHS.length; // 5

/** Largest number of delimiter pairs that fits the vocabulary. */
export function maxDelims(vocabSize: number): number {
  return Math.max(1, Math.min(MAX_DELIMS, Math.floor(vocabSize / 2)));
}

/** Default number of delimiter pairs for a vocabulary (≈ a quarter). */
export function defaultDelims(vocabSize: number): number {
  return Math.max(1, Math.min(maxDelims(vocabSize), Math.floor(vocabSize / 4) || 1));
}

/**
 * Role assignment for the parens task. The first `nDelims` pairs of ids are
 * matched delimiters (open = 2k, close = 2k+1); the rest are distractors.
 * `openIds[k]` is closed by `closeIds[k]`.
 */
export function parensRoles(vocabSize: number, nDelims: number): ParensRoles {
  const pairs = Math.max(1, Math.min(maxDelims(vocabSize), Math.floor(nDelims)));
  const openIds: number[] = [];
  const closeIds: number[] = [];
  for (let k = 0; k < pairs; k++) {
    openIds.push(2 * k);
    closeIds.push(2 * k + 1);
  }
  const distractorIds: number[] = [];
  for (let id = 2 * pairs; id < vocabSize; id++) distractorIds.push(id);
  return { openIds, closeIds, distractorIds };
}

export interface ParensRoles {
  openIds: number[];
  closeIds: number[];
  distractorIds: number[];
}

/** Display character for a token id under the given task. */
export function tokenChar(
  task: Task,
  id: number,
  vocabSize: number,
  nDelims: number,
): string {
  if (task === "parens") {
    const { openIds, closeIds } = parensRoles(vocabSize, nDelims);
    const oi = openIds.indexOf(id);
    if (oi >= 0) return OPEN_GLYPHS[oi % OPEN_GLYPHS.length];
    const ci = closeIds.indexOf(id);
    if (ci >= 0) return CLOSE_GLYPHS[ci % CLOSE_GLYPHS.length];
    const di = id - openIds.length * 2;
    return DISTRACTOR_GLYPHS[di % DISTRACTOR_GLYPHS.length] ?? "?";
  }
  return ALPHABET[id] ?? "?";
}

/** Stable HSL color for a token id, for the colored-squares display mode. */
export function tokenColor(id: number, vocabSize: number): string {
  const hue = Math.round((360 * id) / Math.max(1, vocabSize));
  return `hsl(${hue}, 70%, 60%)`;
}

/**
 * Whether a delimiter sequence is balanced. Distractor tokens are ignored;
 * each close must match the most recently opened (unclosed) delimiter of the
 * same pair index.
 */
export function isBalanced(seq: number[], roles: ParensRoles): boolean {
  const openOf = new Map<number, number>();
  const closeOf = new Map<number, number>();
  roles.openIds.forEach((id, k) => openOf.set(id, k));
  roles.closeIds.forEach((id, k) => closeOf.set(id, k));

  const stack: number[] = [];
  for (const id of seq) {
    if (openOf.has(id)) {
      stack.push(openOf.get(id)!);
    } else if (closeOf.has(id)) {
      if (stack.length === 0 || stack.pop() !== closeOf.get(id)) return false;
    }
    // distractors: ignored
  }
  return stack.length === 0;
}
