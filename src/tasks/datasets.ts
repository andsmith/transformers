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
import { enumerateMatches, inputToGlyphs, matchesAny } from "./grok";

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
  /** true = each length equally likely; false = length ∝ V^L. */
  uniformLen: boolean;
  /** Parens: max nesting depth. */
  parensMaxDepth: number;
  /** Parens: forbid mixing delimiter types within a nest. */
  parensNoMixedNesting: boolean;
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

/**
 * Sample a sequence length in [lo, hi]. uniform=true picks each length with
 * equal probability; uniform=false weights length L by V^L, i.e. samples
 * uniformly over the whole sample space (longer sequences are far more
 * numerous, so they dominate).
 */
function sampleLen(
  rng: () => number,
  lo: number,
  hi: number,
  vocabSize: number,
  uniform: boolean,
): number {
  if (uniform || hi <= lo) return randInt(rng, lo, hi);
  let total = 0;
  for (let L = lo; L <= hi; L++) total += Math.pow(vocabSize, L);
  let r = rng() * total;
  for (let L = lo; L <= hi; L++) {
    r -= Math.pow(vocabSize, L);
    if (r < 0) return L;
  }
  return hi;
}

function randomSeq(rng: () => number, vocabSize: number, len: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < len; i++) out.push(randInt(rng, 0, vocabSize - 1));
  return out;
}

/**
 * A random balanced bracket sequence (Dyck word) over the available pairs.
 * @param maxDepth caps nesting depth (stack height).
 * @param noMixed when true, a nested open reuses the enclosing pair type
 *   (only top-level opens pick a new delimiter type), so no nest mixes types.
 */
function randomDyck(
  rng: () => number,
  roles: ParensRoles,
  pairs: number,
  maxDepth: number,
  noMixed: boolean,
): number[] {
  const res: number[] = [];
  const stack: number[] = [];
  const total = 2 * pairs;
  let opened = 0;
  for (let step = 0; step < total; step++) {
    const remaining = total - step;
    const canOpen = opened < pairs && stack.length < maxDepth;
    const canClose = stack.length > 0;
    const mustClose = remaining <= stack.length;
    let doOpen: boolean;
    if (!canOpen) doOpen = false;
    else if (!canClose) doOpen = true;
    else if (mustClose) doOpen = false;
    else doOpen = rng() < 0.5;

    if (doOpen) {
      const k =
        noMixed && stack.length > 0
          ? stack[stack.length - 1]
          : randInt(rng, 0, roles.openIds.length - 1);
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
function balancedParens(
  rng: () => number,
  roles: ParensRoles,
  len: number,
  maxDepth: number,
  noMixed: boolean,
): number[] {
  const maxPairs = Math.floor(len / 2);
  const pairs = randInt(rng, 0, Math.min(maxPairs, len));
  const brackets = randomDyck(rng, roles, pairs, maxDepth, noMixed);

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

/** The output a task assigns to a given input (deterministic). */
export function taskOutput(task: Task, input: number[], roles: ParensRoles): number[] {
  switch (task) {
    case "copy":
      return input.slice();
    case "reverse":
      return input.slice().reverse();
    case "sort":
      return input.slice().sort((a, b) => a - b);
    case "parens":
      return [isBalanced(input, roles) ? 1 : 0];
  }
}

function generateOne(
  rng: () => number,
  cfg: GenConfig,
  roles: ParensRoles,
): Omit<Example, "index"> {
  const { task, vocabSize, minLen, maxLen, uniformLen } = cfg;
  const len = sampleLen(rng, minLen, maxLen, vocabSize, uniformLen);
  let input: number[];
  if (task === "parens") {
    // Half balanced (respecting depth/no-mixed options), half fully random.
    input =
      rng() < 0.5
        ? balancedParens(rng, roles, len, cfg.parensMaxDepth, cfg.parensNoMixedNesting)
        : randomSeq(rng, vocabSize, len);
  } else {
    input = randomSeq(rng, vocabSize, len);
  }
  return { input, output: taskOutput(task, input, roles) };
}

export interface TestSetOptions extends GenConfig {
  count: number;
  seed: number;
  /** Compiled grok filters; empty = ordinary (non-grok) test set. */
  filters: RegExp[];
  /** Enumerate the space when its size ≤ this; else sample. */
  enumCap: number;
}

function shuffleInPlace<T>(rng: () => number, a: T[]): void {
  for (let i = a.length - 1; i > 0; i--) {
    const j = randInt(rng, 0, i);
    [a[i], a[j]] = [a[j], a[i]];
  }
}

/**
 * Generate the fixed test set. Without grok filters it is `count` DISTINCT
 * random samples. With filters it is the held-out subset matching any filter:
 * exact (enumerated) when the space ≤ enumCap, else rejection-sampled.
 */
export function generateTestSet(opts: TestSetOptions): Dataset {
  const { task, vocabSize, count, seed, uniformLen, filters, enumCap } = opts;
  const minLen = Math.min(opts.minLen, opts.maxLen);
  const maxLen = opts.maxLen;
  const cfg: GenConfig = {
    task,
    vocabSize,
    minLen,
    maxLen,
    uniformLen,
    parensMaxDepth: opts.parensMaxDepth,
    parensNoMixedNesting: opts.parensNoMixedNesting,
  };
  const rng = mulberry32(seed);
  const roles = parensRoles(vocabSize);

  const test: Example[] = [];
  const testKeys = new Set<string>();
  let matchInfo: Dataset["matchInfo"];

  const pushKey = (key: string) => {
    const input = key.split(",").map(Number);
    testKeys.add(key);
    test.push({ index: test.length, input, output: taskOutput(task, input, roles) });
  };

  if (filters.length > 0) {
    const en = enumerateMatches(task, vocabSize, minLen, maxLen, filters, enumCap);
    if (en.mode === "enumerated") {
      shuffleInPlace(rng, en.keys);
      for (let i = 0; i < Math.min(count, en.keys.length); i++) pushKey(en.keys[i]);
      matchInfo = { count: en.count, mode: "enumerated" };
    } else {
      // Space too large to enumerate — rejection-sample matching draws.
      const maxAttempts = Math.max(20000, count * 200);
      let attempts = 0;
      while (test.length < count && attempts < maxAttempts) {
        attempts++;
        const ex = generateOne(rng, cfg, roles);
        const key = sampleKey(ex.input);
        if (testKeys.has(key)) continue;
        if (!matchesAny(filters, inputToGlyphs(task, vocabSize, ex.input))) continue;
        testKeys.add(key);
        test.push({ index: test.length, ...ex });
      }
      matchInfo = { count: test.length, mode: "sampled" };
    }
  } else {
    const maxAttempts = Math.max(1000, count * 50);
    let attempts = 0;
    while (test.length < count && attempts < maxAttempts) {
      attempts++;
      const ex = generateOne(rng, cfg, roles);
      const key = sampleKey(ex.input);
      if (testKeys.has(key)) continue;
      testKeys.add(key);
      test.push({ index: test.length, ...ex });
    }
  }

  return {
    task,
    vocabSize,
    minLen,
    maxLen,
    uniformLen,
    parensMaxDepth: opts.parensMaxDepth,
    parensNoMixedNesting: opts.parensNoMixedNesting,
    filters,
    test,
    testKeys,
    matchInfo,
  };
}

/**
 * Draw one fresh training sample from the generation distribution, rejecting
 * any sample that is in the test set OR matches a grok filter (the held-out
 * language). Bounded retries; the last draw is accepted as-is on exhaustion.
 *
 * @param rng the loop's serializable RNG — saves restore the exact stream.
 * @param displayIndex shown as the sample's id (use the iteration count).
 * @param stats optional counter: `rejections` increments per held-out hit.
 */
export function generateTrainExample(
  ds: Dataset,
  rng: { next(): number },
  displayIndex: number,
  stats?: { rejections: number },
): Example {
  const roles = parensRoles(ds.vocabSize);
  const r = () => rng.next();
  const cfg: GenConfig = {
    task: ds.task,
    vocabSize: ds.vocabSize,
    minLen: ds.minLen,
    maxLen: ds.maxLen,
    uniformLen: ds.uniformLen,
    parensMaxDepth: ds.parensMaxDepth,
    parensNoMixedNesting: ds.parensNoMixedNesting,
  };
  const heldOut = (input: number[]): boolean =>
    ds.testKeys.has(sampleKey(input)) ||
    (ds.filters.length > 0 &&
      matchesAny(ds.filters, inputToGlyphs(ds.task, ds.vocabSize, input)));

  let ex = generateOne(r, cfg, roles);
  let tries = 0;
  while (heldOut(ex.input) && tries < 200) {
    if (stats) stats.rejections++;
    ex = generateOne(r, cfg, roles);
    tries++;
  }
  return { index: displayIndex, ...ex };
}
