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
  peScheme: PEScheme;
  numOutputLayers: 1 | 2;
  learningRate: number;
  numExamples: number;
  minSeqLen: number;
  maxSeqLen: number;
  fixedLength: boolean;
  trainTestSplit: number;
  seed: number;
  randomSeed: boolean;
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
    peScheme: state.peScheme,
    numOutputLayers: state.numOutputLayers,
    learningRate: state.learningRate,
    numExamples: state.numExamples,
    minSeqLen: state.minSeqLen,
    maxSeqLen: state.maxSeqLen,
    fixedLength: state.fixedLength,
    trainTestSplit: state.trainTestSplit,
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
    for (const p of state.model.store.params) {
      weights[p.name] = p.values.map((row) => row.map((v) => v.data));
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

  // Copy the known experiment keys explicitly (ignore anything else).
  state.task = e.task;
  state.numSymbols = e.numSymbols;
  state.embedDim = e.embedDim;
  state.peScheme = e.peScheme;
  state.numOutputLayers = e.numOutputLayers;
  state.learningRate = e.learningRate;
  state.numExamples = e.numExamples;
  state.minSeqLen = e.minSeqLen;
  state.maxSeqLen = e.maxSeqLen;
  state.fixedLength = e.fixedLength;
  state.trainTestSplit = e.trainTestSplit;
  state.seed = e.seed;
  state.randomSeed = e.randomSeed;
  state.running = false;

  doRebuild(); // same seed -> identical dataset + freshly initialized model

  if (file.weights) {
    for (const p of state.model.store.params) {
      const w = file.weights[p.name];
      if (!w || w.length !== p.rows || (w[0]?.length ?? 0) !== p.cols) {
        return `Weight matrix "${p.name}" missing or wrong shape in save file.`;
      }
      for (let r = 0; r < p.rows; r++) {
        for (let c = 0; c < p.cols; c++) p.values[r][c].data = w[r][c];
      }
    }
    if (file.history) state.loop.restore(file.history);
  }
  return null;
}
