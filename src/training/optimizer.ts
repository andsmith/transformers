/**
 * SGD. `SGD` operates on the scalar `ParamStore` (kept for the gradient-parity
 * oracle); `FastSGD` drives the app's typed-array {@link FastModel}.
 */

import { ParamStore } from "../model/params";
import type { FastModel } from "../model/fast";

export interface OptimizerConfig {
  learningRate: number;
}

/** SGD over a {@link FastModel}: w -= lr·g, then zero grads. */
export class FastSGD {
  constructor(
    private readonly model: FastModel,
    private cfg: OptimizerConfig,
  ) {}

  get learningRate(): number {
    return this.cfg.learningRate;
  }

  setLearningRate(lr: number): void {
    this.cfg = { ...this.cfg, learningRate: lr };
  }

  step(): void {
    this.model.step(this.cfg.learningRate);
  }

  zeroGrad(): void {
    this.model.zeroGrad();
  }
}

export class SGD {
  constructor(
    private readonly store: ParamStore,
    private cfg: OptimizerConfig,
  ) {}

  get learningRate(): number {
    return this.cfg.learningRate;
  }

  setLearningRate(lr: number): void {
    this.cfg = { ...this.cfg, learningRate: lr };
  }

  /** Apply one gradient-descent step: p.data -= lr * p.grad. */
  step(): void {
    const lr = this.cfg.learningRate;
    for (const v of this.store.all()) v.data -= lr * v.grad;
  }

  zeroGrad(): void {
    this.store.zeroGrad();
  }
}
