/**
 * Training driver. Owns the step "clock": the UI can advance the computation by
 * one layer, one iteration (single sample), one epoch (a full pass over the
 * train set), or run continuously. It records per-iteration and per-epoch loss
 * series for the loss panel.
 *
 * STUB: the loss computation and gradient plumbing are real (they exercise the
 * autograd engine), but because the model's `forward` currently returns zero
 * logits, no actual learning happens yet — the loss curve is a flat baseline
 * until the real forward path is implemented next.
 */

import { Value } from "../engine/value";
import { crossEntropy, sum } from "../engine/ops";
import { TransformerModel } from "../model/transformer";
import { SGD } from "./optimizer";
import { isClassification } from "../tasks/types";
import type { Dataset, Example } from "../tasks/types";

export type StepGranularity = "layer" | "iteration" | "epoch" | "run";

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

  constructor(
    private readonly model: TransformerModel,
    private readonly optim: SGD,
    private readonly data: Dataset,
  ) {}

  /** Compute the (autograd) loss for one example. */
  private loss(ex: Example): Value {
    const { logits } = this.model.forward(ex.input);
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

  /** Forward + backward + update on a single sample; returns its loss. */
  private trainSample(ex: Example): number {
    const loss = this.loss(ex);
    this.optim.zeroGrad();
    loss.backward();
    this.optim.step();
    return loss.data;
  }

  /** Average loss over a set, without updating weights. */
  private evalLoss(set: Example[]): number | null {
    if (set.length === 0) return null;
    let acc = 0;
    for (const ex of set) acc += this.loss(ex).data;
    return acc / set.length;
  }

  /**
   * Advance one "layer" of computation.
   * TODO(next milestone): step through embedding -> attention -> output one
   * stage at a time for the visualization. For the single-layer scaffold this
   * delegates to a full iteration.
   */
  stepLayer(): void {
    this.stepIteration();
  }

  /** Train on the next single sample. */
  stepIteration(): void {
    if (this.data.train.length === 0) return;
    const ex = this.data.train[this.cursor];
    const trainLoss = this.trainSample(ex);

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
