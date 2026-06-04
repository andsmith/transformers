/**
 * Token + positional embeddings.
 *
 * STUB: the structure, parameter registration, and sinusoidal table are in
 * place, but this is intentionally minimal scaffolding — the full forward path
 * is wired up alongside attention/transformer in the next milestone.
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

export class Embeddings {
  readonly cfg: EmbeddingConfig;
  /** Token embedding table: vocabSize x embedDim (trainable). */
  private readonly tokenTable: Value[][];
  /** Positional encodings: maxLen x embedDim. Fixed (sinusoidal) or learned. */
  private readonly posTable: Value[][];

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

  /** Look up token + positional embedding for each position: seqLen x embedDim. */
  embed(tokenIds: number[]): Value[][] {
    return tokenIds.map((id, pos) => addVec(this.tokenTable[id], this.posTable[pos]));
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
