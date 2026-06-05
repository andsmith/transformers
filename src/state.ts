/**
 * Single mutable application state plus helpers to (re)build the dataset and
 * model from it. Panels read this object directly and call their `update()`
 * after `main.ts` mutates it (the whiteboard_web convention).
 */

import { generateDataset, mulberry32 } from "./tasks/datasets";
import type { Dataset, Task } from "./tasks/types";
import type { PEScheme } from "./model/embeddings";
import { TransformerModel } from "./model/transformer";
import { SGD } from "./training/optimizer";
import { TrainingLoop, type StepGranularity } from "./training/loop";
import { Rng } from "./util/rng";

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
  minSeqLen: number; // shortest generated sequence
  maxSeqLen: number; // longest generated sequence
  fixedLength: boolean; // if true, every sequence is exactly maxSeqLen
  trainTestSplit: number; // 0..0.5 (fraction held out for test)
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

  // --- which split is shown in the dataset list ---
  datasetView: DatasetView;

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
  numExamples: number;
  minSeqLen: number;
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
  minSeqLen: 3,
  maxSeqLen: 7,
  fixedLength: false,
  trainTestSplit: 0.1,
};

/** Generate just the dataset from the current settings (no model rebuild). */
export function rebuildDataset(state: {
  task: Task;
  numSymbols: number;
  numExamples: number;
  minSeqLen: number;
  maxSeqLen: number;
  fixedLength: boolean;
  trainTestSplit: number;
  seed: number;
}): Dataset {
  const maxLen = state.maxSeqLen;
  const minLen = state.fixedLength ? maxLen : Math.min(state.minSeqLen, maxLen);
  return generateDataset({
    task: state.task,
    vocabSize: state.numSymbols,
    count: state.numExamples,
    testFraction: state.trainTestSplit,
    seed: state.seed,
    minLen,
    maxLen,
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
export function rebuild(state: {
  task: Task;
  numSymbols: number;
  embedDim: number;
  peScheme: PEScheme;
  numOutputLayers: 1 | 2;
  learningRate: number;
  numExamples: number;
  minSeqLen: number;
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
  // Separate (serializable) RNG stream for the per-epoch sample-order shuffle.
  const loop = new TrainingLoop(model, optim, dataset, new Rng(state.seed ^ 0x51ed270b));

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
    randomSeed: true,
    lossView: "iteration",
    lossLogScale: false,
    lossGridLines: true,
    datasetView: "train",
    weightsCmap: "viridis",
    actsCmap: "bwr",
    vizConstantSize: false,
    topCollapsed: false,
    lossCollapsed: false,
    uiFontPx: 12,
    vizFontPx: 9,
    ...built,
  };
}
