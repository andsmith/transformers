/**
 * Sample generators for the four toy tasks.
 *
 * The TEST set is generated once from a seed (deduplicated, fixed). TRAINING
 * samples are drawn on the fly from the same distribution, rejected when they
 * collide with the test set — the test set is truly held out, and training
 * asymptotically covers the rest of the sample space.
 */

import type { Dataset, Example, Task } from "./types";
import { isBalanced, parensRoles, type ParensRoles } from "./grammar";

/** Default sequence-length range for generated strings (short strings). */
export const MIN_LEN = 3;
export const MAX_LEN = 7;
/** Hard upper bound for the max-sequence-length slider. */
export const MAX_SEQ_LEN_LIMIT = 12;
/** Anything above this counts as "effectively infinite" sample space. */
export const SPACE_HUGE = 1e15;

/** Generation rules (everything needed to draw one sample). */
export interface GenConfig {
  task: Task;
  vocabSize: number;
  minLen: number;
  maxLen: number;
}

/** Identity key for rejection sampling (input determines output). */
export function sampleKey(input: number[]): string {
  return input.join(",");
}

/**
 * Theoretical number of distinct samples under the generation rules:
 * Σ V^L over L in [minLen, maxLen] (just V^maxLen when fixed-length).
 * Capped at SPACE_HUGE — beyond that the exact value is irrelevant.
 */
export function sampleSpaceSize(
  vocabSize: number,
  minLen: number,
  maxLen: number,
  fixedLength: boolean,
): number {
  let total = 0;
  const lo = fixedLength ? maxLen : Math.min(minLen, maxLen);
  for (let L = lo; L <= maxLen; L++) {
    total += Math.pow(vocabSize, L);
    if (total >= SPACE_HUGE) return SPACE_HUGE;
  }
  return total;
}

/** mulberry32: tiny, fast, deterministic PRNG seeded from a 32-bit int. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randInt(rng: () => number, lo: number, hi: number): number {
  return lo + Math.floor(rng() * (hi - lo + 1));
}

function randomSeq(rng: () => number, vocabSize: number, len: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < len; i++) out.push(randInt(rng, 0, vocabSize - 1));
  return out;
}

/** A random balanced bracket sequence (Dyck word) over the available pairs. */
function randomDyck(rng: () => number, roles: ParensRoles, pairs: number): number[] {
  const res: number[] = [];
  const stack: number[] = [];
  const total = 2 * pairs;
  let opened = 0;
  for (let step = 0; step < total; step++) {
    const remaining = total - step;
    const canOpen = opened < pairs;
    const canClose = stack.length > 0;
    const mustClose = remaining <= stack.length;
    let doOpen: boolean;
    if (!canOpen) doOpen = false;
    else if (!canClose) doOpen = true;
    else if (mustClose) doOpen = false;
    else doOpen = rng() < 0.5;

    if (doOpen) {
      const k = randInt(rng, 0, roles.openIds.length - 1);
      res.push(roles.openIds[k]);
      stack.push(k);
      opened++;
    } else {
      const k = stack.pop()!;
      res.push(roles.closeIds[k]);
    }
  }
  return res;
}

/** Build a balanced parens example of the given total length, with distractors. */
function balancedParens(rng: () => number, roles: ParensRoles, len: number): number[] {
  const maxPairs = Math.floor(len / 2);
  const pairs = randInt(rng, 0, Math.min(maxPairs, len));
  const brackets = randomDyck(rng, roles, pairs);

  // Sprinkle the remaining slots with distractors (or extra opens/closes that
  // still keep balance is unnecessary — distractors are ignored by the checker).
  const result = brackets.slice();
  const distractCount = len - brackets.length;
  const pool = roles.distractorIds.length > 0 ? roles.distractorIds : roles.openIds; // fallback
  for (let i = 0; i < distractCount; i++) {
    const pos = randInt(rng, 0, result.length);
    const sym = pool[randInt(rng, 0, pool.length - 1)];
    result.splice(pos, 0, sym);
  }
  return result;
}

function generateOne(
  rng: () => number,
  task: Task,
  vocabSize: number,
  roles: ParensRoles,
  minLen: number,
  maxLen: number,
): Omit<Example, "index"> {
  const len = randInt(rng, minLen, maxLen);

  switch (task) {
    case "copy": {
      const input = randomSeq(rng, vocabSize, len);
      return { input, output: input.slice() };
    }
    case "reverse": {
      const input = randomSeq(rng, vocabSize, len);
      return { input, output: input.slice().reverse() };
    }
    case "sort": {
      const input = randomSeq(rng, vocabSize, len);
      return { input, output: input.slice().sort((a, b) => a - b) };
    }
    case "parens": {
      // Half the time construct a balanced string, half the time go fully
      // random — then label whatever we produced. This keeps the two classes
      // roughly balanced.
      const input =
        rng() < 0.5 ? balancedParens(rng, roles, len) : randomSeq(rng, vocabSize, len);
      return { input, output: [isBalanced(input, roles) ? 1 : 0] };
    }
  }
}

export interface TestSetOptions extends GenConfig {
  count: number;
  seed: number;
}

/**
 * Generate the fixed test set: `count` DISTINCT samples (deduplicated by
 * input key, bounded attempts so tiny sample spaces can't hang), seeded so
 * the same seed reproduces the same test set. Indices are 0..N-1.
 */
export function generateTestSet(opts: TestSetOptions): Dataset {
  const { task, vocabSize, count, seed } = opts;
  const minLen = Math.min(opts.minLen, opts.maxLen);
  const maxLen = opts.maxLen;
  const rng = mulberry32(seed);
  const roles = parensRoles(vocabSize);

  const test: Example[] = [];
  const testKeys = new Set<string>();
  const maxAttempts = Math.max(1000, count * 50);
  let attempts = 0;
  while (test.length < count && attempts < maxAttempts) {
    attempts++;
    const ex = generateOne(rng, task, vocabSize, roles, minLen, maxLen);
    const key = sampleKey(ex.input);
    if (testKeys.has(key)) continue; // duplicate — redraw
    testKeys.add(key);
    test.push({ index: test.length, ...ex });
  }

  return { task, vocabSize, minLen, maxLen, test, testKeys };
}

/**
 * Draw one fresh training sample from the generation distribution, rejecting
 * collisions with the test set (bounded retries; in the degenerate case where
 * the space is nearly all test, the last draw is accepted as-is).
 *
 * @param rng the loop's serializable RNG — saves restore the exact stream.
 * @param displayIndex shown as the sample's id (use the iteration count).
 */
export function generateTrainExample(
  ds: Dataset,
  rng: { next(): number },
  displayIndex: number,
): Example {
  const roles = parensRoles(ds.vocabSize);
  const r = () => rng.next();
  let ex = generateOne(r, ds.task, ds.vocabSize, roles, ds.minLen, ds.maxLen);
  for (let tries = 0; tries < 200 && ds.testKeys.has(sampleKey(ex.input)); tries++) {
    ex = generateOne(r, ds.task, ds.vocabSize, roles, ds.minLen, ds.maxLen);
  }
  return { index: displayIndex, ...ex };
}
