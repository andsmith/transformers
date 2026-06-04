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
const OPEN_GLYPHS = "([{<";
const CLOSE_GLYPHS = ")]}>";
const DISTRACTOR_GLYPHS = ".,;:*";

/**
 * Role assignment for the parens task. Token ids are split into matched
 * delimiter pairs followed by distractors. `openIds[k]` is closed by
 * `closeIds[k]`.
 */
export interface ParensRoles {
  openIds: number[];
  closeIds: number[];
  distractorIds: number[];
}

/**
 * Assign delimiter/distractor roles for a given vocabulary size. Roughly a
 * quarter of the vocabulary becomes delimiter pairs (at least one), the rest
 * are distractors. Single bracket type for V<=3; more types as V grows.
 */
export function parensRoles(vocabSize: number): ParensRoles {
  const maxPairsByGlyph = Math.min(OPEN_GLYPHS.length, Math.floor(vocabSize / 2));
  const pairs = Math.max(1, Math.min(maxPairsByGlyph, Math.floor(vocabSize / 4) || 1));
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

/** Display character for a token id under the given task. */
export function tokenChar(task: Task, id: number, vocabSize: number): string {
  if (task === "parens") {
    const { openIds, closeIds } = parensRoles(vocabSize);
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
