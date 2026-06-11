/**
 * Demonstration sets: curated input/output pairs per task, authored in
 * `demo_tasks.json` (repo top level, bundled via JSON import). Each demo has a
 * teaching name + description and an optional recommended model `setup`. They
 * are meant to be stepped through on a model the user has trained on real data,
 * to make a specific mechanism visible (reverse → anti-diagonal attention,
 * `)(` fooling a parens counter, sort routing by token identity, …).
 *
 * Inputs/outputs are stored as GLYPH strings (human-authorable, independent of
 * the chars-vs-squares display). `prepareDemos` converts them to the token-id
 * `Example`s the model consumes, computing the true task output from the input
 * and validating that the demo fits the current vocabulary / sequence length.
 */

import type { Dataset, Example, Task } from "./types";
import type { PEScheme } from "../model/embeddings";
import { parensRoles, tokenChar } from "./grammar";
import { sampleKey, taskOutput } from "./datasets";
import rawDemos from "../../demo_tasks.json";

/** Optional recommended model config for a demo (applied by "Apply setup"). */
export interface DemoSetup {
  numSymbols?: number;
  tokenOneHot?: boolean;
  peScheme?: PEScheme;
  numOutputLayers?: 1 | 2;
  learningRate?: number;
  minSeqLen?: number;
  maxSeqLen?: number;
  parensDelims?: number;
}

/** One authored demo (raw JSON shape). */
export interface RawDemo {
  name: string;
  description: string;
  /** Glyph string, e.g. "abc" (transduction) or "([])" (parens). */
  input: string;
  /** Glyph string (transduction) or "balanced"/"unbalanced" (parens). */
  output: string;
  setup?: DemoSetup;
}

export interface TaskDemos {
  task: Task;
  demos: RawDemo[];
}

// The JSON's literal type (task: string, …) doesn't structurally match the
// narrower TaskDemos[]; cast through unknown.
export const DEMO_TASKS: TaskDemos[] = rawDemos as unknown as TaskDemos[];

/** Authored demos for a task (empty if none). */
export function demosForTask(task: Task): RawDemo[] {
  return DEMO_TASKS.find((t) => t.task === task)?.demos ?? [];
}

/**
 * Invert {@link tokenChar}: map a glyph string to token ids under the given
 * vocabulary. Returns null if any glyph is unrepresentable (vocab too small, or
 * — for parens — too few delimiter kinds).
 */
export function glyphsToIds(
  task: Task,
  glyphs: string,
  numSymbols: number,
  nDelims: number,
): number[] | null {
  const byGlyph = new Map<string, number>();
  for (let id = 0; id < numSymbols; id++) {
    byGlyph.set(tokenChar(task, id, numSymbols, nDelims), id);
  }
  const ids: number[] = [];
  for (const ch of glyphs) {
    const id = byGlyph.get(ch);
    if (id === undefined) return null;
    ids.push(id);
  }
  return ids;
}

/** A demo converted to ids under a concrete config, with validity. */
export interface PreparedDemo {
  /** Stable index = position in the task's authored list. */
  index: number;
  name: string;
  description: string;
  setup?: DemoSetup;
  input: number[];
  output: number[];
  /** Fits the current vocabulary AND sequence length (safe to run). */
  valid: boolean;
  /** Why it doesn't fit (empty when valid). */
  hint: string;
}

/** Short "what to change" hint for a demo that doesn't fit the current config. */
function unfitHint(d: RawDemo): string {
  const s = d.setup;
  if (!s) return "doesn't fit the current vocabulary";
  const parts: string[] = [];
  if (s.numSymbols) parts.push(`|V| = ${s.numSymbols}`);
  if (s.parensDelims) parts.push(`${s.parensDelims} delimiter kind${s.parensDelims > 1 ? "s" : ""}`);
  if (s.maxSeqLen) parts.push(`max length ${s.maxSeqLen}`);
  return parts.length ? `needs ${parts.join(", ")} — click Apply setup` : "click Apply setup";
}

/**
 * Convert a task's authored demos to id `Example`s under a concrete config.
 * The output is computed from the input (the true task target); the authored
 * output string is for documentation and is validated by the smoke test.
 */
export function prepareDemos(
  task: Task,
  cfg: { numSymbols: number; parensDelims: number; maxSeqLen: number },
): PreparedDemo[] {
  const roles = parensRoles(cfg.numSymbols, cfg.parensDelims);
  return demosForTask(task).map((d, index) => {
    const base = { index, name: d.name, description: d.description, setup: d.setup };
    const ids = glyphsToIds(task, d.input, cfg.numSymbols, cfg.parensDelims);
    if (ids === null) {
      return { ...base, input: [], output: [], valid: false, hint: unfitHint(d) };
    }
    if (ids.length > cfg.maxSeqLen) {
      return {
        ...base,
        input: ids,
        output: [],
        valid: false,
        hint: `needs max length ≥ ${ids.length}`,
      };
    }
    return { ...base, input: ids, output: taskOutput(task, ids, roles), valid: true, hint: "" };
  });
}

/**
 * Build a {@link Dataset} whose test set IS the (valid) demos for the current
 * task. `isDemo` marks it so the training loop iterates the fixed list instead
 * of sampling. Invalid demos are excluded here (they would not fit the model);
 * the panel still lists them, greyed, from {@link prepareDemos}.
 */
export function buildDemoDataset(state: {
  task: Task;
  numSymbols: number;
  minSeqLen: number;
  maxSeqLen: number;
  uniformLen: boolean;
  parensMaxDepth: number;
  parensNoMixedNesting: boolean;
  parensDelims: number;
}): Dataset {
  const prepared = prepareDemos(state.task, state);
  const test: Example[] = prepared
    .filter((d) => d.valid)
    .map((d) => ({ index: d.index, input: d.input, output: d.output }));
  return {
    task: state.task,
    vocabSize: state.numSymbols,
    minLen: state.minSeqLen,
    maxLen: state.maxSeqLen,
    uniformLen: state.uniformLen,
    parensMaxDepth: state.parensMaxDepth,
    parensNoMixedNesting: state.parensNoMixedNesting,
    parensDelims: state.parensDelims,
    filters: [],
    test,
    testKeys: new Set(test.map((e) => sampleKey(e.input))),
    isDemo: true,
  };
}
