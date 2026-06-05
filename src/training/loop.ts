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
import { generateTrainExample } from "../tasks/datasets";
import { isClassification } from "../tasks/types";
import type { Dataset, Example } from "../tasks/types";

/**
 * What one Step click advances. The "step 1 ..." modes run on the Speed-paced
 * tick when Go is active; "run" (continuous iterations) and "epochs" (fastest,
 * UI refreshed once per epoch) are unthrottled — Speed does not apply.
 */
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

/** Training throughput sample (recorded ~once per second while running). */
export interface TimingPoint {
  /** Iteration count when the sample was taken. */
  x: number;
  /** Samples (iterations) per second over the preceding window. */
  sps: number;
}

/** Per-test-sample result from a post-epoch evaluation. */
export interface TestEval {
  index: number; // test sample id
  pred: number[]; // argmax tokens (classification: [predLabel])
  correct: boolean[]; // per position (classification: single element)
  pTrue: number[]; // probability assigned to the TRUE token per position
  wrong: number; // count of incorrect positions
  firstWrong: number; // position of first wrong token (Infinity if none)
  meanPTrue: number; // mean confidence on the truth (sort tie-breaker)
}

/** Per-epoch statistics recorded at each rollover. */
export interface EpochStat {
  /** Epoch number (matches epochHistory.x). */
  x: number;
  /** Wall-clock duration of the epoch in seconds (includes any pauses). */
  seconds: number;
  /** Training draws rejected for colliding with the test set. */
  hits: number;
}

/** Everything needed to continue a run exactly where it left off
 *  (sample-boundary granularity; weights are saved separately). */
export interface LoopSnapshotData {
  iterHistory: LossPoint[];
  epochHistory: LossPoint[];
  iteration: number;
  epoch: number;
  cursor: number;
  /** Legacy (pre-0.0.23 fixed-train-set saves); ignored on restore. */
  order?: number[];
  epochIterStart: number;
  rngState: number;
  /** Optional (added in 0.0.22); older saves simply lack it. */
  timingHistory?: TimingPoint[];
  /** Optional (added in 0.0.29). */
  epochStats?: EpochStat[];
  /** Optional (added in 0.0.30): last post-epoch detailed test eval. */
  lastTestEval?: TestEval[];
  lastTestEvalEpoch?: number;
}

export class TrainingLoop {
  iteration = 0;
  epoch = 0;
  /** Position within the current epoch (0..trainPerEpoch-1). */
  private cursor = 0;
  /** Per-iteration and per-epoch loss history (read by the loss panel). */
  readonly iterHistory: LossPoint[] = [];
  readonly epochHistory: LossPoint[] = [];
  /** Throughput history (samples/sec), pushed by the app's tick sampler. */
  readonly timingHistory: TimingPoint[] = [];
  /** Per-epoch stats (duration, test-set hits), recorded at each rollover. */
  readonly epochStats: EpochStat[] = [];
  /** Wall-clock start of the current epoch. */
  private epochStartT = performance.now();
  /** Index into iterHistory where the current epoch began (for epoch means). */
  private epochIterStart = 0;
  /** Test-set hits (rejected draws) while sampling the current epoch. */
  private rejCounter = { rejections: 0 };
  /** Test-set hits during the most recently completed epoch (-1 = none yet). */
  lastEpochRejections = -1;
  /** Per-sample results of the most recent post-epoch test eval (null = none). */
  lastTestEval: TestEval[] | null = null;
  lastTestEvalEpoch = -1;
  /** In-progress per-stage walkthrough (null between samples). */
  staged: StagedSample | null = null;

  constructor(
    private readonly model: TransformerModel,
    private readonly optim: SGD,
    private readonly data: Dataset,
    private readonly rng: Rng = new Rng((Math.random() * 2 ** 32) >>> 0),
    private readonly trainPerEpoch: number = 100,
  ) {}

  /** Position within the current epoch (for progress display). */
  get cursorPos(): number {
    return this.cursor;
  }

  /** Number of on-the-fly training samples that make up one epoch. */
  get trainSize(): number {
    return this.trainPerEpoch;
  }

  /**
   * Draw the next training sample: a fresh random sample from the generation
   * distribution, rejected against the test set. Driven by the serializable
   * RNG, so a restored save reproduces the exact same training stream.
   */
  private nextSample(): Example {
    return generateTrainExample(this.data, this.rng, this.iteration, this.rejCounter);
  }

  /** Test-set hits while sampling the epoch in progress. */
  get rejectionsThisEpoch(): number {
    return this.rejCounter.rejections;
  }

  /** Capture continuation state (histories, counters, RNG). */
  serialize(): LoopSnapshotData {
    return {
      iterHistory: this.iterHistory.map((p) => ({ ...p })),
      epochHistory: this.epochHistory.map((p) => ({ ...p })),
      timingHistory: this.timingHistory.map((p) => ({ ...p })),
      epochStats: this.epochStats.map((p) => ({ ...p })),
      iteration: this.iteration,
      epoch: this.epoch,
      cursor: this.cursor,
      epochIterStart: this.epochIterStart,
      rngState: this.rng.state,
      lastTestEval: this.lastTestEval ?? undefined,
      lastTestEvalEpoch: this.lastTestEvalEpoch,
    };
  }

  /**
   * Adopt another loop's histories and counters (used when only the dataset
   * is regenerated/resized: the model keeps training, the curves keep
   * growing). The new epoch starts fresh (cursor 0, new shuffle) since the
   * train set changed.
   */
  carryOver(prev: TrainingLoop): void {
    this.iterHistory.length = 0;
    for (const p of prev.iterHistory) this.iterHistory.push(p);
    this.epochHistory.length = 0;
    for (const p of prev.epochHistory) this.epochHistory.push(p);
    this.timingHistory.length = 0;
    for (const p of prev.timingHistory) this.timingHistory.push(p);
    this.epochStats.length = 0;
    for (const p of prev.epochStats) this.epochStats.push(p);
    this.iteration = prev.iteration;
    this.epoch = prev.epoch;
    this.epochIterStart = prev.epochIterStart;
    this.rng.state = prev.rng.state;
    this.rejCounter.rejections = prev.rejCounter.rejections;
    this.lastEpochRejections = prev.lastEpochRejections;
    // The test set may have changed — stale per-index marks would lie.
    this.lastTestEval = null;
    this.lastTestEvalEpoch = -1;
  }

  /** Restore a serialized continuation state (after weights are loaded). */
  restore(h: LoopSnapshotData): void {
    this.iterHistory.length = 0;
    for (const p of h.iterHistory) this.iterHistory.push({ ...p });
    this.epochHistory.length = 0;
    for (const p of h.epochHistory) this.epochHistory.push({ ...p });
    this.timingHistory.length = 0;
    for (const p of h.timingHistory ?? []) this.timingHistory.push({ ...p });
    this.epochStats.length = 0;
    for (const p of h.epochStats ?? []) this.epochStats.push({ ...p });
    this.lastTestEval = h.lastTestEval ?? null;
    this.lastTestEvalEpoch = h.lastTestEvalEpoch ?? -1;
    this.iteration = h.iteration;
    this.epoch = h.epoch;
    this.cursor = h.cursor;
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
   * Detailed evaluation over the full test set: per-sample predictions,
   * correctness and confidence, plus the mean loss (numerically equal to the
   * autograd loss). Used at each epoch rollover.
   */
  private evalTestDetailed(): { mean: number | null; evals: TestEval[] } {
    const set = this.data.test;
    if (set.length === 0) return { mean: null, evals: [] };
    const classification = isClassification(this.data.task);
    const evals: TestEval[] = [];
    let lossSum = 0;

    for (const ex of set) {
      const logits = this.model.forward(ex.input).logits;
      const pred: number[] = [];
      const correct: boolean[] = [];
      const pTrue: number[] = [];

      if (classification) {
        const p = 1 / (1 + Math.exp(-logits[0][0].data)); // P(label = 1)
        const label = p >= 0.5 ? 1 : 0;
        const truth = ex.output[0];
        const pt = truth === 1 ? p : 1 - p;
        pred.push(label);
        correct.push(label === truth);
        pTrue.push(pt);
        lossSum += -Math.log(Math.max(pt, 1e-12));
      } else {
        let posLoss = 0;
        for (let i = 0; i < logits.length; i++) {
          const row = logits[i].map((v) => v.data);
          const m = Math.max(...row);
          const exps = row.map((z) => Math.exp(z - m));
          const sum = exps.reduce((a, b) => a + b, 0);
          const probs = exps.map((e) => e / sum);
          let arg = 0;
          for (let k = 1; k < probs.length; k++) if (probs[k] > probs[arg]) arg = k;
          const truth = ex.output[i];
          pred.push(arg);
          correct.push(arg === truth);
          pTrue.push(probs[truth]);
          posLoss += -Math.log(Math.max(probs[truth], 1e-12));
        }
        lossSum += posLoss / logits.length;
      }

      let wrong = 0;
      let firstWrong = Infinity;
      for (let i = 0; i < correct.length; i++) {
        if (!correct[i]) {
          wrong++;
          if (firstWrong === Infinity) firstWrong = i;
        }
      }
      const meanPTrue = pTrue.reduce((a, b) => a + b, 0) / pTrue.length;
      evals.push({ index: ex.index, pred, correct, pTrue, wrong, firstWrong, meanPTrue });
    }

    return { mean: lossSum / set.length, evals };
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

    if (this.cursor >= this.trainPerEpoch) {
      this.cursor = 0;
      this.epoch++;
      this.lastEpochRejections = this.rejCounter.rejections;
      const now = performance.now();
      this.epochStats.push({
        x: this.epoch,
        seconds: (now - this.epochStartT) / 1000,
        hits: this.rejCounter.rejections,
      });
      this.epochStartT = now;
      this.rejCounter.rejections = 0;
      const pts = this.iterHistory.slice(this.epochIterStart);
      const mean =
        pts.reduce((a, p) => a + p.trainLoss, 0) / Math.max(1, pts.length);
      // Detailed test eval drives both the epoch's test-loss point and the
      // per-sample marks shown in the dataset panel.
      const detailed = this.evalTestDetailed();
      this.lastTestEval = detailed.evals.length ? detailed.evals : null;
      this.lastTestEvalEpoch = this.epoch;
      this.epochHistory.push({
        x: this.epoch,
        trainLoss: mean,
        testLoss: detailed.mean,
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
    const sample = this.nextSample();
    const result = this.model.forward(sample.input);
    const lossValue = this.lossFrom(result.logits, sample);
    this.model.zeroGrad();
    lossValue.backward();
    this.optim.step();
    if (snapshot || this.cursor === this.trainPerEpoch - 1) {
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
   * Train one full epoch (trainPerEpoch fresh samples). The epoch-level loss
   * point is recorded by the rollover logic in finishSample.
   */
  stepEpoch(): void {
    const startEpoch = this.epoch;
    let guard = 0;
    const maxSteps = this.trainPerEpoch + 1;
    while (this.epoch === startEpoch && guard < maxSteps) {
      this.stepIteration();
      guard++;
    }
  }
}
