/**
 * The toy transformer: token+positional embedding -> single attention head ->
 * output layer(s). Output width is `|V|` per position for the transduction
 * tasks, or `1` for the parens classification task.
 *
 * STUB: assembles the sub-modules and registers all parameters (so the
 * optimizer and visualization see a complete model), but `forward` returns
 * correctly-shaped zero logits for now. The real forward path lands in the
 * next milestone.
 */

import { Value } from "../engine/value";
import { ParamStore, heUniform } from "./params";
import { Embeddings, type PEScheme } from "./embeddings";
import { AttentionHead, type AttentionTrace } from "./attention";
import { isClassification, type Task } from "../tasks/types";

export interface ModelConfig {
  task: Task;
  vocabSize: number;
  embedDim: number;
  peScheme: PEScheme;
  /** 1 = single output projection, 2 = one hidden layer + projection. */
  numOutputLayers: 1 | 2;
  maxLen: number;
}

export interface ForwardTrace {
  /** seqLen x embedDim input embeddings. */
  embeddings: Value[][];
  attention: AttentionTrace;
}

export interface ForwardResult {
  /**
   * Logits. Transduction: seqLen x vocabSize. Classification: 1 x 1.
   */
  logits: Value[][];
  trace: ForwardTrace;
}

export class TransformerModel {
  readonly cfg: ModelConfig;
  readonly store: ParamStore;
  private readonly embeddings: Embeddings;
  private readonly attention: AttentionHead;
  /** Output projection: outputUnits x embedDim. */
  private readonly wOut: Value[][];
  readonly outputUnits: number;

  constructor(cfg: ModelConfig, rng: () => number) {
    this.cfg = cfg;
    this.store = new ParamStore();
    this.outputUnits = isClassification(cfg.task) ? 1 : cfg.vocabSize;

    this.embeddings = new Embeddings(
      this.store,
      {
        vocabSize: cfg.vocabSize,
        embedDim: cfg.embedDim,
        peScheme: cfg.peScheme,
        maxLen: cfg.maxLen,
      },
      rng,
    );

    this.attention = new AttentionHead(
      this.store,
      { embedDim: cfg.embedDim, headDim: cfg.embedDim },
      rng,
    );

    // TODO(next milestone): if numOutputLayers === 2, register a hidden layer.
    this.wOut = this.store.matrix(
      "out_proj",
      this.outputUnits,
      cfg.embedDim,
      heUniform(rng, cfg.embedDim),
    );
  }

  /**
   * Run the model on a token sequence.
   * TODO(next milestone): real forward (embed -> attention -> projection).
   * Currently returns zero logits of the correct shape.
   */
  forward(tokenIds: number[]): ForwardResult {
    void this.wOut;
    const embeddings = this.embeddings.embed(tokenIds);
    const { trace: attnTrace } = this.attention.forward(embeddings);

    const positions = isClassification(this.cfg.task) ? 1 : tokenIds.length;
    const logits: Value[][] = Array.from({ length: positions }, () =>
      Array.from({ length: this.outputUnits }, () => new Value(0)),
    );

    return { logits, trace: { embeddings, attention: attnTrace } };
  }
}
