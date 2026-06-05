/**
 * Single mutable application state plus helpers to (re)build the dataset and
 * model from it. Panels read this object directly and call their `update()`
 * after `main.ts` mutates it (the whiteboard_web convention).
 */

import { generateDataset, mulberry32, MIN_LEN } from "./tasks/datasets";
import type { Dataset, Task } from "./tasks/types";
import type { PEScheme } from "./model/embeddings";
import { TransformerModel } from "./model/transformer";
import { SGD } from "./training/optimizer";
import { TrainingLoop, type StepGranularity } from "./training/loop";

export type DisplayMode = "chars" | "squares";
export type LossView = "iteration" | "epoch";
export type DatasetView = "train" | "test";

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

  // --- dataset controls ---
  display: DisplayMode;
  numExamples: number; // 10..5000
  maxSeqLen: number; // longest generated sequence
  fixedLength: boolean; // if true, every sequence is exactly maxSeqLen
  trainTestSplit: number; // 0..0.5 (fraction held out for test)
  seed: number;
  dataset: Dataset;

  // --- model + training (rebuilt when hyperparameters change) ---
  model: TransformerModel;
  optim: SGD;
  loop: TrainingLoop;

  // --- loss panel ---
  lossView: LossView;

  // --- which split is shown in the dataset list ---
  datasetView: DatasetView;

  // --- network visualization colormaps (view-only) ---
  weightsCmap: string; // key into WEIGHT_CMAPS
  actsCmap: string; // key into ACT_CMAPS

  // --- misc view options ---
  /** Size viz elements for maxSeqLen so the layout doesn't shift when the
   *  sequence length varies (cells stretch; aspect ratio changes). */
  vizConstantSize: boolean;
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
  /** Bump the seed and regenerate dataset + model. */
  regenerate(): void;
  /** Advance the computation, or toggle continuous running (granularity-dependent). */
  step(): void;
}

export interface Defaults {
  task: Task;
  numSymbols: number;
  embedDim: number;
  peScheme: PEScheme;
  numOutputLayers: 1 | 2;
  learningRate: number;
  numExamples: number;
  maxSeqLen: number;
  fixedLength: boolean;
  trainTestSplit: number;
}

const DEFAULTS: Defaults = {
  task: "copy",
  numSymbols: 4,
  embedDim: 8,
  peScheme: "sinusoidal",
  numOutputLayers: 1,
  learningRate: 0.05,
  numExamples: 100,
  maxSeqLen: 7,
  fixedLength: false,
  trainTestSplit: 0.2,
};

/** (Re)build dataset, model, optimizer and training loop from the given state. */
export function rebuild(state: {
  task: Task;
  numSymbols: number;
  embedDim: number;
  peScheme: PEScheme;
  numOutputLayers: 1 | 2;
  learningRate: number;
  numExamples: number;
  maxSeqLen: number;
  fixedLength: boolean;
  trainTestSplit: number;
  seed: number;
}): {
  dataset: Dataset;
  model: TransformerModel;
  optim: SGD;
  loop: TrainingLoop;
} {
  // When fixedLength is on every sequence is exactly maxSeqLen; otherwise they
  // vary from MIN_LEN up to maxSeqLen (clamped so min never exceeds max).
  const maxLen = state.maxSeqLen;
  const minLen = state.fixedLength ? maxLen : Math.min(MIN_LEN, maxLen);

  const dataset = generateDataset({
    task: state.task,
    vocabSize: state.numSymbols,
    count: state.numExamples,
    testFraction: state.trainTestSplit,
    seed: state.seed,
    minLen,
    maxLen,
  });

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
      maxLen, // positional table must cover the longest sequence
    },
    rng,
  );
  const optim = new SGD(model.store, { learningRate: state.learningRate });
  // Separate RNG stream for the per-epoch sample-order shuffle.
  const loop = new TrainingLoop(model, optim, dataset, mulberry32(state.seed ^ 0x51ed270b));

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
    display: "chars",
    seed,
    lossView: "iteration",
    datasetView: "train",
    weightsCmap: "viridis",
    actsCmap: "bwr",
    vizConstantSize: false,
    ...built,
  };
}
