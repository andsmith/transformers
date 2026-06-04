/**
 * Training driver. Owns the step "clock": the UI advances the computation one
 * pipeline stage at a time ("1 layer" granularity) — revealing the forward
 * pass left-to-right, then sweeping gradients right-to-left as the
 * backpropagation pass — and records per-iteration / per-epoch loss series for
 * the loss panel. (Iteration/epoch/continuous stepping exist but are disabled
 * in the UI for now.)
 */

import { Value } from "../engine/value";
import { crossEntropy, sum } from "../engine/ops";
import { TransformerModel, type ForwardTrace } from "../model/transformer";
import { SGD } from "./optimizer";
import { isClassification } from "../tasks/types";
import type { Dataset, Example } from "../tasks/types";

export type StepGranularity = "layer" | "iteration" | "epoch" | "run";
export type PassPhase = "forward" | "backward";

export interface PipelineStage {
  id: string;
  title: string;
}

/** The visualization's stage columns, left to right. */
export const PIPELINE_STAGES: PipelineStage[] = [
  { id: "input", title: "Input" },
  { id: "embed", title: "Embeddings" },
  { id: "sum", title: "Embed Sum" },
  { id: "qkv", title: "Q · K · V" },
  { id: "scores", title: "Attention" },
  { id: "attnout", title: "Weighted Sum" },
  { id: "residual", title: "Residual" },
  { id: "output", title: "Output" },
];

/** The sample currently being stepped through, plus its cached computation. */
export interface StagedSample {
  sample: Example;
  trace: ForwardTrace;
  lossValue: Value;
  phase: PassPhase;
  /** Index into PIPELINE_STAGES of the active column. */
  stage: number;
}

export interface LossPoint {
  /** x-axis index (iteration number, or epoch number). */
  x: number;
  trainLoss: number;
  testLoss: number | null;
}

export class TrainingLoop {
  iteration = 0;
  epoch = 0;
  /** Index of the next train sample to process this epoch. */
  private cursor = 0;
  /** Per-iteration and per-epoch loss history (read by the loss panel). */
  readonly iterHistory: LossPoint[] = [];
  readonly epochHistory: LossPoint[] = [];
  /** In-progress per-stage walkthrough (null between samples). */
  staged: StagedSample | null = null;

  constructor(
    private readonly model: TransformerModel,
    private readonly optim: SGD,
    private readonly data: Dataset,
  ) {}

  /** Loss for already-computed logits. */
  private lossFrom(logits: Value[][], ex: Example): Value {
    if (isClassification(this.data.task)) {
      const logit = logits[0][0];
      const p = logit.neg().exp().add(1).pow(-1); // sigmoid
      const y = ex.output[0];
      const term1 = p.log().mul(y);
      const term2 = p.neg().add(1).log().mul(1 - y);
      return term1.add(term2).neg(); // binary cross-entropy
    }
    const perPos = ex.output.map((t, i) => crossEntropy(logits[i], t));
    return sum(perPos).div(ex.output.length);
  }

  /** Compute the (autograd) loss for one example (fresh forward). */
  private loss(ex: Example): Value {
    return this.lossFrom(this.model.forward(ex.input).logits, ex);
  }

  /** Average loss over a set, without updating weights. */
  private evalLoss(set: Example[]): number | null {
    if (set.length === 0) return null;
    let acc = 0;
    for (const ex of set) acc += this.loss(ex).data;
    return acc / set.length;
  }

  /** Bookkeeping after a sample's gradients have been applied. */
  private finishSample(trainLoss: number): void {
    this.cursor++;
    this.iteration++;
    if (this.cursor >= this.data.train.length) {
      this.cursor = 0;
      this.epoch++;
    }
    // Evaluate test loss periodically to keep the second series cheap.
    const testLoss =
      this.iteration % 10 === 0 ? this.evalLoss(this.data.test) : null;
    this.iterHistory.push({ x: this.iteration, trainLoss, testLoss });
  }

  /**
   * Advance one pipeline stage.
   *
   * Starting a sample runs the full forward + loss + backward() immediately
   * (so values AND gradients are cached on the graph); stepping is then a
   * presentation cursor over the cached computation. The weight update is
   * applied only when the backward sweep completes, so the weights shown
   * during both passes are the pre-update ones.
   */
  stepLayer(): void {
    if (this.data.train.length === 0) return;

    if (!this.staged) {
      const sample = this.data.train[this.cursor];
      const result = this.model.forward(sample.input);
      const lossValue = this.lossFrom(result.logits, sample);
      this.optim.zeroGrad();
      lossValue.backward();
      this.staged = {
        sample,
        trace: result.trace,
        lossValue,
        phase: "forward",
        stage: 0,
      };
      return;
    }

    const st = this.staged;
    if (st.phase === "forward") {
      if (st.stage < PIPELINE_STAGES.length - 1) st.stage++;
      else st.phase = "backward"; // gradients sweep starts at the last column
      return;
    }

    // Backward phase.
    if (st.stage > 0) {
      st.stage--;
      return;
    }

    // Backward sweep complete: apply the update, log the loss, and start the
    // next sample in the same click so stepping never feels like a no-op.
    this.optim.step();
    this.finishSample(st.lossValue.data);
    this.staged = null;
    this.stepLayer();
  }

  /** Train on the next single sample (whole iteration at once). */
  stepIteration(): void {
    if (this.data.train.length === 0) return;
    this.staged = null; // abandon any in-progress walkthrough
    const ex = this.data.train[this.cursor];
    const lossValue = this.loss(ex);
    this.optim.zeroGrad();
    lossValue.backward();
    this.optim.step();
    this.finishSample(lossValue.data);
  }

  /** Train one full pass over the train set, recording an epoch-level point. */
  stepEpoch(): void {
    const startEpoch = this.epoch;
    let guard = 0;
    const maxSteps = this.data.train.length + 1;
    while (this.epoch === startEpoch && guard < maxSteps) {
      this.stepIteration();
      guard++;
    }
    const trainLoss = this.evalLoss(this.data.train) ?? 0;
    const testLoss = this.evalLoss(this.data.test);
    this.epochHistory.push({ x: this.epoch, trainLoss, testLoss });
  }
}
