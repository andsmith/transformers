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
import { Rng } from "../util/rng";
import { isClassification } from "../tasks/types";
import type { Dataset, Example } from "../tasks/types";

/** "epochs" = continuous training that only refreshes the UI per epoch. */
export type StepGranularity = "layer" | "iteration" | "epoch" | "run" | "epochs";
/** "complete" = a whole-sample step (iteration/epoch/run) finished training
 *  this sample; the viz shows the full pipeline, no active stage. */
export type PassPhase = "forward" | "backward" | "complete";

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

/** Everything needed to continue a run exactly where it left off
 *  (sample-boundary granularity; weights are saved separately). */
export interface LoopSnapshotData {
  iterHistory: LossPoint[];
  epochHistory: LossPoint[];
  iteration: number;
  epoch: number;
  cursor: number;
  order: number[];
  epochIterStart: number;
  rngState: number;
}

export class TrainingLoop {
  iteration = 0;
  epoch = 0;
  /** Position within the current epoch's (shuffled) visit order. */
  private cursor = 0;
  /** This epoch's visit order: indices into data.train, reshuffled per epoch. */
  private order: number[] = [];
  /** Per-iteration and per-epoch loss history (read by the loss panel). */
  readonly iterHistory: LossPoint[] = [];
  readonly epochHistory: LossPoint[] = [];
  /** Index into iterHistory where the current epoch began (for epoch means). */
  private epochIterStart = 0;
  /** In-progress per-stage walkthrough (null between samples). */
  staged: StagedSample | null = null;

  constructor(
    private readonly model: TransformerModel,
    private readonly optim: SGD,
    private readonly data: Dataset,
    private readonly rng: Rng = new Rng((Math.random() * 2 ** 32) >>> 0),
  ) {
    this.reshuffle();
  }

  /** Position within the current epoch (for progress display). */
  get cursorPos(): number {
    return this.cursor;
  }

  get trainSize(): number {
    return this.data.train.length;
  }

  /** Fisher-Yates shuffle of the train-set visit order (called each epoch). */
  private reshuffle(): void {
    this.order = this.data.train.map((_, i) => i);
    for (let i = this.order.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng.next() * (i + 1));
      [this.order[i], this.order[j]] = [this.order[j], this.order[i]];
    }
  }

  /** The next train sample in this epoch's shuffled order. */
  private nextSample(): Example {
    return this.data.train[this.order[this.cursor]];
  }

  /** Capture continuation state (histories, counters, order, RNG). */
  serialize(): LoopSnapshotData {
    return {
      iterHistory: this.iterHistory.map((p) => ({ ...p })),
      epochHistory: this.epochHistory.map((p) => ({ ...p })),
      iteration: this.iteration,
      epoch: this.epoch,
      cursor: this.cursor,
      order: [...this.order],
      epochIterStart: this.epochIterStart,
      rngState: this.rng.state,
    };
  }

  /** Restore a serialized continuation state (after weights are loaded). */
  restore(h: LoopSnapshotData): void {
    this.iterHistory.length = 0;
    for (const p of h.iterHistory) this.iterHistory.push({ ...p });
    this.epochHistory.length = 0;
    for (const p of h.epochHistory) this.epochHistory.push({ ...p });
    this.iteration = h.iteration;
    this.epoch = h.epoch;
    this.cursor = h.cursor;
    this.order = [...h.order];
    this.epochIterStart = h.epochIterStart;
    this.rng.state = h.rngState;
    this.staged = null;
  }

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

  /**
   * Average loss over (a capped prefix of) a set, without updating weights.
   * The cap keeps periodic evaluation cheap on large datasets; the sets are
   * pre-shuffled so a prefix is an unbiased sample.
   */
  private evalLoss(set: Example[], cap = 200): number | null {
    if (set.length === 0) return null;
    const subset = set.slice(0, cap);
    let acc = 0;
    for (const ex of subset) acc += this.loss(ex).data;
    return acc / subset.length;
  }

  /**
   * Bookkeeping after a sample's gradients have been applied. On epoch
   * rollover, records an epoch-level point (mean train loss over the epoch's
   * iterations + a test evaluation) so the per-epoch plot works in every
   * stepping mode.
   */
  private finishSample(trainLoss: number): void {
    this.cursor++;
    this.iteration++;
    // Evaluate test loss periodically to keep the second series cheap.
    const testLoss =
      this.iteration % 10 === 0 ? this.evalLoss(this.data.test, 50) : null;
    this.iterHistory.push({ x: this.iteration, trainLoss, testLoss });

    if (this.cursor >= this.data.train.length) {
      this.cursor = 0;
      this.epoch++;
      this.reshuffle(); // new random visit order every epoch
      const pts = this.iterHistory.slice(this.epochIterStart);
      const mean =
        pts.reduce((a, p) => a + p.trainLoss, 0) / Math.max(1, pts.length);
      this.epochHistory.push({
        x: this.epoch,
        trainLoss: mean,
        testLoss: this.evalLoss(this.data.test),
      });
      this.epochIterStart = this.iterHistory.length;
    }
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

    if (!this.staged || this.staged.phase === "complete") {
      const sample = this.nextSample();
      const result = this.model.forward(sample.input);
      const lossValue = this.lossFrom(result.logits, sample);
      this.model.zeroGrad();
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

  /**
   * Train on the next single sample (whole iteration at once). Leaves a
   * fully-revealed "complete" snapshot for the network view, so the pipeline
   * animates as samples fly by in iteration/epoch/continuous modes.
   *
   * @param snapshot pass false (fast "epochs" mode) to skip the snapshot
   *   except on the epoch's final sample, so each epoch still ends with its
   *   last sample on display.
   */
  stepIteration(snapshot = true): void {
    if (this.data.train.length === 0) return;
    const sample = this.nextSample();
    const result = this.model.forward(sample.input);
    const lossValue = this.lossFrom(result.logits, sample);
    this.model.zeroGrad();
    lossValue.backward();
    this.optim.step();
    if (snapshot || this.cursor === this.data.train.length - 1) {
      this.staged = {
        sample,
        trace: result.trace,
        lossValue,
        phase: "complete",
        stage: PIPELINE_STAGES.length - 1,
      };
    }
    this.finishSample(lossValue.data);
  }

  /**
   * Train one full pass over the train set. The epoch-level loss point is
   * recorded by the rollover logic in finishSample.
   */
  stepEpoch(): void {
    if (this.data.train.length === 0) return;
    const startEpoch = this.epoch;
    let guard = 0;
    const maxSteps = this.data.train.length + 1;
    while (this.epoch === startEpoch && guard < maxSteps) {
      this.stepIteration();
      guard++;
    }
  }
}
