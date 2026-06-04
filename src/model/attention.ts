/**
 * Single self-attention head.
 *
 * STUB: parameters (Q/K/V projections) are registered so the optimizer and viz
 * can see them, and the forward signature is fixed, but the scaled
 * dot-product computation is left for the next milestone. For now `forward`
 * returns the input unchanged (identity) so shapes line up and the rest of the
 * scaffolding can be exercised.
 */

import { Value } from "../engine/value";
import { ParamStore, heUniform } from "./params";

export interface AttentionConfig {
  embedDim: number;
  /** Dimensionality of the head (== embedDim for a single full-width head). */
  headDim: number;
}

/** Per-head intermediates we will expose to the visualization. */
export interface AttentionTrace {
  /** seqLen x seqLen attention weights (after softmax). */
  weights: number[][];
}

export class AttentionHead {
  readonly cfg: AttentionConfig;
  private readonly wq: Value[][];
  private readonly wk: Value[][];
  private readonly wv: Value[][];

  constructor(store: ParamStore, cfg: AttentionConfig, rng: () => number) {
    this.cfg = cfg;
    const { embedDim, headDim } = cfg;
    this.wq = store.matrix("attn_Wq", headDim, embedDim, heUniform(rng, embedDim));
    this.wk = store.matrix("attn_Wk", headDim, embedDim, heUniform(rng, embedDim));
    this.wv = store.matrix("attn_Wv", headDim, embedDim, heUniform(rng, embedDim));
  }

  /**
   * Compute attended outputs for a sequence of embeddings.
   * @param x seqLen x embedDim
   * @returns seqLen x headDim outputs plus an optional trace for the viz.
   *
   * TODO(next milestone): project to Q/K/V, scaled dot-product attention,
   * softmax over keys, weighted sum of V. Currently identity passthrough.
   */
  forward(x: Value[][]): { out: Value[][]; trace: AttentionTrace } {
    void this.wq;
    void this.wk;
    void this.wv;
    const n = x.length;
    const weights = Array.from({ length: n }, () => Array.from({ length: n }, () => 0));
    return { out: x, trace: { weights } };
  }
}
