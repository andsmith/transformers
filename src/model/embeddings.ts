/**
 * Token + positional embeddings.
 *
 * The token table is always trainable. The positional table is either the
 * fixed sinusoidal encoding (Vaswani et al., 2017) or a trainable matrix,
 * depending on the configured scheme.
 */

import { Value } from "../engine/value";
import { addVec } from "../engine/ops";
import { ParamStore, heUniform } from "./params";

export type PEScheme = "sinusoidal" | "learned";

export interface EmbeddingConfig {
  vocabSize: number;
  embedDim: number;
  peScheme: PEScheme;
  /** Maximum sequence length we precompute positional encodings for. */
  maxLen: number;
}

/** Per-sample embedding intermediates, exposed for the visualization. */
export interface EmbeddingLookup {
  /** seqLen x embedDim — the token-table rows for the input ids (by reference,
   *  so gradients accumulate into the table). */
  tok: Value[][];
  /** seqLen x embedDim — the positional rows for positions 0..seqLen-1. */
  pos: Value[][];
  /** seqLen x embedDim — tok + pos, the transformer's input x. */
  sum: Value[][];
}

export class Embeddings {
  readonly cfg: EmbeddingConfig;
  /** Token embedding table: vocabSize x embedDim (trainable). */
  readonly tokenTable: Value[][];
  /** Positional encodings: maxLen x embedDim. Fixed (sinusoidal) or learned. */
  readonly posTable: Value[][];

  constructor(store: ParamStore, cfg: EmbeddingConfig, rng: () => number) {
    this.cfg = cfg;
    this.tokenTable = store.matrix(
      "tok_emb",
      cfg.vocabSize,
      cfg.embedDim,
      heUniform(rng, cfg.embedDim),
    );

    if (cfg.peScheme === "learned") {
      this.posTable = store.matrix(
        "pos_emb",
        cfg.maxLen,
        cfg.embedDim,
        heUniform(rng, cfg.embedDim),
      );
    } else {
      this.posTable = sinusoidalTable(cfg.maxLen, cfg.embedDim);
    }
  }

  /** Look up token + positional embeddings for a sequence. */
  lookup(tokenIds: number[]): EmbeddingLookup {
    const tok = tokenIds.map((id) => this.tokenTable[id]);
    const pos = tokenIds.map((_, p) => this.posTable[p]);
    const sum = tok.map((row, i) => addVec(row, pos[i]));
    return { tok, pos, sum };
  }

  /** Token + positional embedding per position (just the summed x). */
  embed(tokenIds: number[]): Value[][] {
    return this.lookup(tokenIds).sum;
  }
}

/** Fixed sinusoidal positional encodings (Vaswani et al., 2017), as Values. */
export function sinusoidalTable(maxLen: number, dim: number): Value[][] {
  const table: Value[][] = [];
  for (let pos = 0; pos < maxLen; pos++) {
    const row: Value[] = [];
    for (let i = 0; i < dim; i++) {
      const k = Math.floor(i / 2);
      const denom = Math.pow(10000, (2 * k) / dim);
      const angle = pos / denom;
      row.push(new Value(i % 2 === 0 ? Math.sin(angle) : Math.cos(angle)));
    }
    table.push(row);
  }
  return table;
}
