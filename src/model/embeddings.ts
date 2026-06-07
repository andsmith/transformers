/**
 * Token + positional embeddings.
 *
 * The token table is trainable, or a fixed identity ("one-hot" pedagogical
 * mode, d_tok = |V|). The positional table is the fixed sinusoidal encoding
 * (Vaswani et al., 2017), a trainable matrix, a fixed identity in a dedicated
 * trailing block of `maxLen` coordinates ("one-hot"), or all zeros ("none").
 */

import { Value } from "../engine/value";
import { addVec } from "../engine/ops";
import { ParamStore, heUniform } from "./params";

export type PEScheme = "sinusoidal" | "learned" | "onehot" | "none";

export interface EmbeddingConfig {
  vocabSize: number;
  /** FULL embedding width (token dims + one-hot position block when used). */
  embedDim: number;
  /** Token-content dims (= vocabSize when tokenOneHot; the one-hot position
   *  identity block starts at this column offset). */
  dTok: number;
  /** Fixed identity token table instead of a trainable one. */
  tokenOneHot: boolean;
  peScheme: PEScheme;
  /** Maximum sequence length we precompute positional encodings for. */
  maxLen: number;
}

/** A fixed (non-trainable) Value matrix — not registered in the store. */
function fixedMatrix(
  rows: number,
  cols: number,
  at: (r: number, c: number) => number,
): Value[][] {
  const table: Value[][] = [];
  for (let r = 0; r < rows; r++) {
    const row: Value[] = [];
    for (let c = 0; c < cols; c++) row.push(new Value(at(r, c)));
    table.push(row);
  }
  return table;
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
    this.tokenTable = cfg.tokenOneHot
      ? fixedMatrix(cfg.vocabSize, cfg.embedDim, (r, c) => (c === r ? 1 : 0))
      : store.matrix("tok_emb", cfg.vocabSize, cfg.embedDim, heUniform(rng, cfg.embedDim));

    switch (cfg.peScheme) {
      case "learned":
        this.posTable = store.matrix(
          "pos_emb",
          cfg.maxLen,
          cfg.embedDim,
          heUniform(rng, cfg.embedDim),
        );
        break;
      case "sinusoidal":
        this.posTable = sinusoidalTable(cfg.maxLen, cfg.embedDim);
        break;
      case "onehot":
        // Identity in a dedicated trailing block: position i flags coordinate
        // dTok + i, fully disentangled from the token content dims.
        this.posTable = fixedMatrix(cfg.maxLen, cfg.embedDim, (r, c) =>
          c === cfg.dTok + r ? 1 : 0,
        );
        break;
      case "none":
        this.posTable = fixedMatrix(cfg.maxLen, cfg.embedDim, () => 0);
        break;
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
