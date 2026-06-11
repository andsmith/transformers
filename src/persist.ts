/**
 * Experiment save/load (DOM-free; the UI handles file pickers/downloads).
 *
 * Three levels of detail:
 *  1. experiment only        — architecture + dataset options + hyperparams
 *  2. + weights              — resume training of the same model
 *  3. + weights + history    — continue exactly as if the run never stopped
 */

import type { AppState } from "./state";
import type { LoopSnapshotData } from "./training/loop";
import type { Task } from "./tasks/types";
import type { PEScheme } from "./model/embeddings";
import { VERSION } from "./version";

export interface ExperimentConfig {
  task: Task;
  numSymbols: number;
  embedDim: number;
  /** Added 0.0.43 (pedagogical encodings); older saves default to false. */
  tokenOneHot?: boolean;
  peScheme: PEScheme;
  numOutputLayers: 1 | 2;
  learningRate: number;
  trainPerEpoch: number;
  testSetSize: number;
  minSeqLen: number;
  maxSeqLen: number;
  fixedLength: boolean;
  /** Added 0.0.32; older saves default to true (the prior behavior). */
  uniformLen?: boolean;
  /** Added 0.0.44 (demonstration mode); older saves default to false. */
  demoExamples?: boolean;
  /** Added 0.0.36 (parens options + grokking); older saves use defaults. */
  parensMaxDepth?: number;
  parensNoMixedNesting?: boolean;
  parensDelims?: number;
  grokFilters?: string;
  /** Removed in 0.0.37 (auto-decided); ignored if present in old saves. */
  enumCap?: number;
  seed: number;
  randomSeed: boolean;
  /** Legacy fields (pre-0.0.23 fixed-train-set saves), used for migration. */
  numExamples?: number;
  trainTestSplit?: number;
}

export interface SaveFile {
  format: 1;
  app: string;
  version: string;
  experiment: ExperimentConfig;
  /** Param name -> values, matching the model's ParamStore. */
  weights?: Record<string, number[][]>;
  history?: LoopSnapshotData;
}

export interface SaveOptions {
  weights: boolean;
  /** Only honored when weights is true. */
  history: boolean;
}

export function buildSave(state: AppState, opts: SaveOptions): SaveFile {
  const experiment: ExperimentConfig = {
    task: state.task,
    numSymbols: state.numSymbols,
    embedDim: state.embedDim,
    tokenOneHot: state.tokenOneHot,
    peScheme: state.peScheme,
    numOutputLayers: state.numOutputLayers,
    learningRate: state.learningRate,
    trainPerEpoch: state.trainPerEpoch,
    testSetSize: state.testSetSize,
    minSeqLen: state.minSeqLen,
    maxSeqLen: state.maxSeqLen,
    fixedLength: state.fixedLength,
    uniformLen: state.uniformLen,
    demoExamples: state.demoExamples,
    parensMaxDepth: state.parensMaxDepth,
    parensNoMixedNesting: state.parensNoMixedNesting,
    parensDelims: state.parensDelims,
    grokFilters: state.grokFilters,
    seed: state.seed,
    randomSeed: state.randomSeed,
  };

  const file: SaveFile = {
    format: 1,
    app: "transformer-playground",
    version: VERSION,
    experiment,
  };

  if (opts.weights) {
    const weights: Record<string, number[][]> = {};
    for (const p of state.model.params) {
      weights[p.name] = p.w.map((row) => row.slice());
    }
    file.weights = weights;
    if (opts.history) file.history = state.loop.serialize();
  }
  return file;
}

/**
 * Apply a parsed save file: experiment config -> rebuild -> weights ->
 * history. Returns an error message, or null on success. Applies whatever
 * levels the file contains.
 */
export function applySave(
  file: SaveFile,
  state: AppState,
  doRebuild: () => void,
): string | null {
  if (!file || file.format !== 1 || !file.experiment) {
    return "Unrecognized save-file format.";
  }
  const e = file.experiment;

  // Migrate pre-0.0.23 saves (fixed train set with a split) to the new
  // trainPerEpoch / testSetSize fields.
  let trainPerEpoch = e.trainPerEpoch;
  let testSetSize = e.testSetSize;
  if (trainPerEpoch === undefined && e.numExamples !== undefined) {
    const split = e.trainTestSplit ?? 0.1;
    testSetSize = Math.max(0, Math.round(e.numExamples * split));
    trainPerEpoch = Math.min(8192, Math.max(10, e.numExamples - testSetSize));
  }
  if (trainPerEpoch === undefined || testSetSize === undefined) {
    return "Save file is missing dataset-size fields.";
  }

  // Copy the known experiment keys explicitly (ignore anything else).
  state.task = e.task;
  state.numSymbols = e.numSymbols;
  state.embedDim = e.embedDim;
  state.tokenOneHot = e.tokenOneHot ?? false;
  state.peScheme = e.peScheme;
  state.numOutputLayers = e.numOutputLayers;
  state.learningRate = e.learningRate;
  state.trainPerEpoch = trainPerEpoch;
  state.testSetSize = testSetSize;
  state.minSeqLen = e.minSeqLen;
  state.maxSeqLen = e.maxSeqLen;
  state.fixedLength = e.fixedLength;
  state.uniformLen = e.uniformLen ?? true;
  state.demoExamples = e.demoExamples ?? false;
  state.parensMaxDepth = e.parensMaxDepth ?? Math.max(1, Math.floor(e.maxSeqLen / 2));
  state.parensNoMixedNesting = e.parensNoMixedNesting ?? false;
  state.parensDelims = e.parensDelims ?? 1;
  state.grokFilters = e.grokFilters ?? "";
  state.seed = e.seed;
  // Force deterministic mode so the loaded seed is actually used (and the
  // "random" box reflects it) — otherwise the next Regenerate would discard it.
  state.randomSeed = false;
  state.running = false;

  doRebuild(); // same seed -> identical dataset + freshly initialized model

  if (file.weights) {
    for (const p of state.model.params) {
      const w = file.weights[p.name];
      if (!w || w.length !== p.rows || (w[0]?.length ?? 0) !== p.cols) {
        return `Weight matrix "${p.name}" missing or wrong shape in save file.`;
      }
      for (let r = 0; r < p.rows; r++) {
        for (let c = 0; c < p.cols; c++) p.w[r][c] = w[r][c];
      }
    }
    if (file.history) state.loop.restore(file.history);
  }
  return null;
}
