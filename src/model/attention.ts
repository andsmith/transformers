/**
 * Single self-attention head, vanilla 2017 style: three learned full-size
 * projections W_Q/W_K/W_V (headDim x embedDim, headDim == embedDim here),
 * scaled dot-product scores, row softmax, weighted sum of V.
 *
 * Every intermediate is kept as Values in the trace so the visualization can
 * read both forward activations and (after backward()) their gradients.
 */

import { Value } from "../engine/value";
import { addVec, dot, matVec, scaleVec, softmax } from "../engine/ops";
import { ParamStore, heUniform } from "./params";

export interface AttentionConfig {
  embedDim: number;
  /** Dimensionality of the head (== embedDim for a single full-width head). */
  headDim: number;
}

/** Per-head intermediates exposed to the visualization. */
export interface AttentionTrace {
  /** seqLen x headDim projections. */
  q: Value[][];
  k: Value[][];
  v: Value[][];
  /** seqLen x seqLen scaled dot-product scores (pre-softmax). */
  scores: Value[][];
  /** seqLen x seqLen attention weights (post-softmax, rows sum to 1). */
  attnW: Value[][];
  /** seqLen x headDim weighted sum of V — the attention output. */
  out: Value[][];
}

export class AttentionHead {
  readonly cfg: AttentionConfig;
  readonly wq: Value[][];
  readonly wk: Value[][];
  readonly wv: Value[][];

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
   */
  forward(x: Value[][]): AttentionTrace {
    const n = x.length;
    const invSqrtD = 1 / Math.sqrt(this.cfg.headDim);

    const q = x.map((xi) => matVec(this.wq, xi));
    const k = x.map((xi) => matVec(this.wk, xi));
    const v = x.map((xi) => matVec(this.wv, xi));

    // scores[i][j] = q_i · k_j / sqrt(d)
    const scores: Value[][] = [];
    for (let i = 0; i < n; i++) {
      const row: Value[] = [];
      for (let j = 0; j < n; j++) row.push(dot(q[i], k[j]).mul(invSqrtD));
      scores.push(row);
    }

    const attnW = scores.map((row) => softmax(row));

    // out[i] = sum_j attnW[i][j] * v[j]
    const out: Value[][] = [];
    for (let i = 0; i < n; i++) {
      let acc = scaleVec(v[0], attnW[i][0]);
      for (let j = 1; j < n; j++) acc = addVec(acc, scaleVec(v[j], attnW[i][j]));
      out.push(acc);
    }

    return { q, k, v, scores, attnW, out };
  }
}
