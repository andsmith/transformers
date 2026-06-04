/**
 * Plain SGD over a {@link ParamStore}.
 *
 * STUB: the gradient step is implemented (it is trivial and harmless), but it
 * is only meaningful once the model's forward/backward path is real. Momentum
 * and other optimizers come later.
 */

import { ParamStore } from "../model/params";

export interface OptimizerConfig {
  learningRate: number;
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
