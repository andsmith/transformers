/**
 * Single mutable application state plus helpers to (re)build the dataset and
 * model from it. Panels read this object directly and call their `update()`
 * after `main.ts` mutates it (the whiteboard_web convention).
 */

import { generateTestSet, mulberry32, sampleSpaceSize, SPACE_HUGE } from "./tasks/datasets";
import { compileFilters } from "./tasks/grok";
import type { Dataset, Task } from "./tasks/types";
import type { PEScheme } from "./model/embeddings";
import { TransformerModel } from "./model/transformer";
import { SGD } from "./training/optimizer";
import { TrainingLoop, type StepGranularity } from "./training/loop";
import { Rng } from "./util/rng";

export type DisplayMode = "chars" | "squares";
export type LossView = "iteration" | "epoch";

/** Absolute cap on the test-set-size slider. */
export const TEST_SET_ABS_MAX = 500;
/** Cap on how many training samples make up one epoch. */
export const TRAIN_PER_EPOCH_MAX = 8192;

export interface AppState {
  // --- task & model hyperparameters ---
  task: Task;
  numSymbols: number; // vocabulary size (2..MAX_VOCAB)
  embedDim: number;
  peScheme: PEScheme;
  numOutputLayers: 1 | 2;
  numHeads: number; // single head for now
  learningRate: number;
  stepGranularity: StepGranularity;
  running: boolean;
  /** Tick-rate slider (0..100): 0 = 0.5 Hz, 99 = 5 Hz, 100 = unthrottled max. */
  speed: number;

  // --- dataset controls ---
  display: DisplayMode;
  /** On-the-fly training samples per epoch (10..TRAIN_PER_EPOCH_MAX). */
  trainPerEpoch: number;
  /** Fixed held-out test set size (0..min(500, 20% of the sample space)). */
  testSetSize: number;
  minSeqLen: number; // shortest generated sequence
  maxSeqLen: number; // longest generated sequence
  fixedLength: boolean; // if true, every sequence is exactly maxSeqLen
  uniformLen: boolean; // length prior: equal-per-length vs ∝ V^L
  // --- task-dependent generation options ---
  parensMaxDepth: number; // parens: max nesting depth
  parensNoMixedNesting: boolean; // parens: no mixed delimiter types per nest
  parensDelims: number; // parens: number of distinct delimiter pair kinds
  // --- grokking ---
  grokFilters: string; // comma-separated regexes; held-out subset
  seed: number;
  randomSeed: boolean; // Regenerate draws a fresh random seed each time
  dataset: Dataset;

  // --- model + training (rebuilt when hyperparameters change) ---
  model: TransformerModel;
  optim: SGD;
  loop: TrainingLoop;

  // --- loss panel ---
  lossView: LossView;
  lossLogScale: boolean; // log-scale y-axis
  lossGridLines: boolean; // horizontal grid + epoch boundary lines

  // --- network visualization colormaps (view-only) ---
  weightsCmap: string; // key into WEIGHT_CMAPS
  actsCmap: string; // key into ACT_CMAPS

  // --- misc view options ---
  /** Size viz elements for maxSeqLen so the layout doesn't shift when the
   *  sequence length varies (cells stretch; aspect ratio changes). */
  vizConstantSize: boolean;

  // --- collapsible panels ---
  topCollapsed: boolean; // top controls strip
  lossCollapsed: boolean; // loss plot

  // --- font sizes ---
  uiFontPx: number; // UI labels (everything but titles)
  vizFontPx: number; // visualization captions
}

/**
 * Wiring handed to every panel. Panels never mutate {@link AppState} directly;
 * they call these, and `main.ts` decides whether a rebuild is needed and then
 * refreshes all panels.
 */
export interface AppContext {
  readonly state: AppState;
  /** Merge a patch into state (rebuilding model/data if needed) and refresh UI. */
  apply(patch: Partial<AppState>): void;
  /** Regenerate dataset + model (drawing a fresh seed if randomSeed is on). */
  regenerate(): void;
  /** Advance the computation, or toggle continuous running (granularity-dependent). */
  step(): void;
  /** Re-initialize model weights and clear training history (same dataset/seed). */
  reset(): void;
  /** Apply a parsed save file (JSON text). Returns an error message or null. */
  loadSave(json: string): string | null;
}

export interface Defaults {
  task: Task;
  numSymbols: number;
  embedDim: number;
  peScheme: PEScheme;
  numOutputLayers: 1 | 2;
  learningRate: number;
  trainPerEpoch: number;
  testSetSize: number;
  minSeqLen: number;
  maxSeqLen: number;
  fixedLength: boolean;
  uniformLen: boolean;
  parensMaxDepth: number;
  parensNoMixedNesting: boolean;
  parensDelims: number;
  grokFilters: string;
}

const DEFAULTS: Defaults = {
  task: "copy",
  numSymbols: 4,
  embedDim: 8,
  peScheme: "sinusoidal",
  numOutputLayers: 1,
  learningRate: 0.05,
  trainPerEpoch: 100,
  testSetSize: 50,
  minSeqLen: 3,
  maxSeqLen: 7,
  fixedLength: false,
  uniformLen: true,
  parensMaxDepth: 3,
  parensNoMixedNesting: false,
  parensDelims: 1,
  grokFilters: "",
};

/** Theoretical max nesting depth at a given max sequence length. */
export function maxNestingDepth(maxSeqLen: number): number {
  return Math.max(1, Math.floor(maxSeqLen / 2));
}

export interface GenStateSlice {
  task: Task;
  numSymbols: number;
  testSetSize: number;
  minSeqLen: number;
  maxSeqLen: number;
  fixedLength: boolean;
  uniformLen: boolean;
  parensMaxDepth: number;
  parensNoMixedNesting: boolean;
  parensDelims: number;
  grokFilters: string;
  seed: number;
}

/**
 * Effective training samples per epoch: clamped to the number of distinct
 * non-test samples available, so when the requested count exceeds the sample
 * space the loop doesn't waste passes (and hammer rejection sampling) on
 * samples that can't be distinct.
 */
export function effectiveTrainPerEpoch(
  s: { numSymbols: number; minSeqLen: number; maxSeqLen: number; fixedLength: boolean; trainPerEpoch: number },
  testCount: number,
): number {
  const space = sampleSpaceSize(s.numSymbols, s.minSeqLen, s.maxSeqLen, s.fixedLength);
  const available = Math.max(1, Math.floor(Math.min(space, SPACE_HUGE)) - testCount);
  return Math.min(s.trainPerEpoch, available);
}

/** Largest allowed test set under the current generation rules:
 *  min(500, 20% of the theoretical sample space). */
export function testSetMax(s: {
  numSymbols: number;
  minSeqLen: number;
  maxSeqLen: number;
  fixedLength: boolean;
}): number {
  const space = sampleSpaceSize(s.numSymbols, s.minSeqLen, s.maxSeqLen, s.fixedLength);
  return Math.min(TEST_SET_ABS_MAX, Math.floor(0.2 * space));
}

/** Generate just the (test) dataset from the current settings. */
export function rebuildDataset(state: GenStateSlice): Dataset {
  const maxLen = state.maxSeqLen;
  const minLen = state.fixedLength ? maxLen : Math.min(state.minSeqLen, maxLen);
  const { regexes } = compileFilters(state.grokFilters);
  return generateTestSet({
    task: state.task,
    vocabSize: state.numSymbols,
    count: Math.min(state.testSetSize, testSetMax(state)),
    seed: state.seed,
    minLen,
    maxLen,
    uniformLen: state.uniformLen,
    parensMaxDepth: state.parensMaxDepth,
    parensNoMixedNesting: state.parensNoMixedNesting,
    parensDelims: state.parensDelims,
    filters: regexes,
  });
}

/** One-line model/experiment summary (top title + Status panel). */
export function modelSummary(s: {
  task: Task;
  numSymbols: number;
  embedDim: number;
  peScheme: PEScheme;
  numOutputLayers: 1 | 2;
}): string {
  const pe = s.peScheme === "sinusoidal" ? "positional" : "learned";
  return (
    `task: ${s.task} - |V| = ${s.numSymbols} - ` +
    `Model(D_embed=${s.embedDim}, P_embed=${pe}, FF-layers=${s.numOutputLayers})`
  );
}

/** (Re)build dataset, model, optimizer and training loop from the given state. */
export function rebuild(state: GenStateSlice & {
  embedDim: number;
  peScheme: PEScheme;
  numOutputLayers: 1 | 2;
  learningRate: number;
  trainPerEpoch: number;
}): {
  dataset: Dataset;
  model: TransformerModel;
  optim: SGD;
  loop: TrainingLoop;
} {
  const dataset = rebuildDataset(state);

  // Derive a model-init RNG from the same seed (offset so weights differ from
  // the data stream).
  const rng = mulberry32(state.seed ^ 0x9e3779b9);
  const model = new TransformerModel(
    {
      task: state.task,
      vocabSize: state.numSymbols,
      embedDim: state.embedDim,
      peScheme: state.peScheme,
      numOutputLayers: state.numOutputLayers,
      maxLen: state.maxSeqLen, // positional table must cover the longest sequence
    },
    rng,
  );
  const optim = new SGD(model.store, { learningRate: state.learningRate });
  // Separate (serializable) RNG stream for on-the-fly training-sample draws.
  const loop = new TrainingLoop(
    model,
    optim,
    dataset,
    new Rng(state.seed ^ 0x51ed270b),
    effectiveTrainPerEpoch(state, dataset.test.length),
  );

  return { dataset, model, optim, loop };
}

export function createInitialState(): AppState {
  const seed = 1;
  const built = rebuild({ ...DEFAULTS, seed });
  return {
    ...DEFAULTS,
    numHeads: 1,
    stepGranularity: "layer",
    running: false,
    speed: 100,
    display: "chars",
    seed,
    randomSeed: true,
    lossView: "iteration",
    lossLogScale: true,
    lossGridLines: true,
    weightsCmap: "viridis",
    actsCmap: "bwr",
    vizConstantSize: false,
    topCollapsed: false,
    lossCollapsed: false,
    uiFontPx: 12,
    vizFontPx: 16,
    ...built,
  };
}
