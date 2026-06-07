/**
 * The toy transformer: token+positional embedding -> single attention head ->
 * residual -> output projection. Output width is `|V|` per position for the
 * transduction tasks, or `1` (mean-pooled) for the parens classification task.
 *
 * `forward` returns the logits plus a complete trace of every intermediate as
 * Values, so the visualization can read activations now and gradients after
 * `loss.backward()`.
 */

import { Value } from "../engine/value";
import { addVec, matVec, scaleVec } from "../engine/ops";
import { ParamStore, heUniform } from "./params";
import { Embeddings, type PEScheme } from "./embeddings";
import { AttentionHead, type AttentionTrace } from "./attention";
import { isClassification, type Task } from "../tasks/types";

export interface ModelConfig {
  task: Task;
  vocabSize: number;
  embedDim: number;
  /** Fixed identity token embedding (d_tok = vocabSize). */
  tokenOneHot: boolean;
  peScheme: PEScheme;
  /** 1 = single output projection; 2 = hidden layer relu(W_ff·y) first. */
  numOutputLayers: 1 | 2;
  maxLen: number;
}

/** Token-content dims (one-hot forces |V|). */
export function tokenDims(cfg: { tokenOneHot: boolean; vocabSize: number; embedDim: number }): number {
  return cfg.tokenOneHot ? cfg.vocabSize : cfg.embedDim;
}

/** Full model width: token dims + a dedicated block for one-hot positions. */
export function modelDims(cfg: {
  tokenOneHot: boolean;
  vocabSize: number;
  embedDim: number;
  peScheme: PEScheme;
  maxLen: number;
}): number {
  return tokenDims(cfg) + (cfg.peScheme === "onehot" ? cfg.maxLen : 0);
}

export interface ForwardTrace {
  /** seqLen x vocabSize one-hot input encoding (plain numbers, viz only). */
  oneHot: number[][];
  /** seqLen x d token-table rows for the input ids (refs into the table). */
  tok: Value[][];
  /** seqLen x d positional rows for positions 0..seqLen-1. */
  pos: Value[][];
  /** seqLen x d embedding sum x = tok + pos. */
  x: Value[][];
  /** Attention intermediates (q/k/v, scores, softmax weights, output). */
  attention: AttentionTrace;
  /** seqLen x d residual output y = x + attn(x). */
  y: Value[][];
  /** Mean-pooled y (classification only, else null). */
  pooled: Value[] | null;
  /** Hidden FF activations relu(W_ff·y) when numOutputLayers === 2, else
   *  null. (Classification: a single pooled row.) */
  hidden: Value[][] | null;
  /** Output logits (transduction: seqLen x |V|; classification: 1 x 1). */
  logits: Value[][];
}

export interface ForwardResult {
  logits: Value[][];
  trace: ForwardTrace;
}

export class TransformerModel {
  readonly cfg: ModelConfig;
  readonly store: ParamStore;
  readonly embeddings: Embeddings;
  readonly attention: AttentionHead;
  /** Hidden FF layer (d x d), only when numOutputLayers === 2. */
  readonly wFF: Value[][] | null;
  /** Output projection: outputUnits x embedDim. */
  readonly wOut: Value[][];
  readonly outputUnits: number;

  /** Full embedding width (token dims + one-hot position block when used). */
  readonly dim: number;

  constructor(cfg: ModelConfig, rng: () => number) {
    this.cfg = cfg;
    this.store = new ParamStore();
    this.outputUnits = isClassification(cfg.task) ? 1 : cfg.vocabSize;
    const dim = modelDims(cfg);
    this.dim = dim;

    this.embeddings = new Embeddings(
      this.store,
      {
        vocabSize: cfg.vocabSize,
        embedDim: dim,
        dTok: tokenDims(cfg),
        tokenOneHot: cfg.tokenOneHot,
        peScheme: cfg.peScheme,
        maxLen: cfg.maxLen,
      },
      rng,
    );

    this.attention = new AttentionHead(
      this.store,
      { embedDim: dim, headDim: dim },
      rng,
    );

    this.wFF =
      cfg.numOutputLayers === 2
        ? this.store.matrix("ff_W1", dim, dim, heUniform(rng, dim))
        : null;

    this.wOut = this.store.matrix(
      "out_proj",
      this.outputUnits,
      dim,
      heUniform(rng, dim),
    );
  }

  /**
   * Zero gradients on all trainable params AND any fixed tables (sinusoidal /
   * one-hot / zero) — those are part of the graph but not in the store, so
   * without this their displayed ∇ would accumulate across samples.
   */
  zeroGrad(): void {
    this.store.zeroGrad();
    for (const row of this.embeddings.posTable) {
      for (const v of row) v.grad = 0;
    }
    for (const row of this.embeddings.tokenTable) {
      for (const v of row) v.grad = 0;
    }
  }

  /** Run the model on a token sequence, capturing the full trace. */
  forward(tokenIds: number[]): ForwardResult {
    const n = tokenIds.length;

    // One-hot encoding (display only — the table lookup is its matmul).
    const oneHot = tokenIds.map((id) => {
      const row = new Array<number>(this.cfg.vocabSize).fill(0);
      row[id] = 1;
      return row;
    });

    const { tok, pos, sum: x } = this.embeddings.lookup(tokenIds);
    const attention = this.attention.forward(x);

    // Residual connection.
    const y = x.map((xi, i) => addVec(xi, attention.out[i]));

    // Optional hidden FF layer: h = relu(W_ff · v).
    const ff = (vec: Value[]): Value[] =>
      matVec(this.wFF!, vec).map((v) => v.relu());

    let pooled: Value[] | null = null;
    let hidden: Value[][] | null = null;
    let logits: Value[][];
    if (isClassification(this.cfg.task)) {
      // Mean-pool positions, then (optionally) the FF layer, then project.
      const inv = 1 / n;
      let acc = scaleVec(y[0], inv);
      for (let i = 1; i < n; i++) acc = addVec(acc, scaleVec(y[i], inv));
      pooled = acc;
      let pre = pooled;
      if (this.wFF) {
        pre = ff(pooled);
        hidden = [pre];
      }
      logits = [matVec(this.wOut, pre)];
    } else {
      let pre = y;
      if (this.wFF) {
        hidden = y.map(ff);
        pre = hidden;
      }
      logits = pre.map((h) => matVec(this.wOut, h));
    }

    return {
      logits,
      trace: { oneHot, tok, pos, x, attention, y, pooled, hidden, logits },
    };
  }
}
