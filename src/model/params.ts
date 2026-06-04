/**
 * Parameter registry. Every trainable weight in the model is created through a
 * {@link ParamStore} so that a single place can enumerate them for the
 * optimizer (gradient steps) and for the visualization (drawing weight values
 * and gradients as Hinton diagrams).
 */

import { Value } from "../engine/value";

/** A named, trainable tensor (matrix). Vectors are stored as a single row. */
export interface Param {
  name: string;
  rows: number;
  cols: number;
  /** rows x cols grid of leaf Values. */
  values: Value[][];
}

export type Initializer = () => number;

/** Small-random initializer scaled by fan-in (1/sqrt(cols)). */
export function heUniform(rng: () => number, fanIn: number): Initializer {
  const scale = 1 / Math.sqrt(Math.max(1, fanIn));
  return () => (rng() * 2 - 1) * scale;
}

export class ParamStore {
  readonly params: Param[] = [];

  /** Create and register a rows x cols parameter matrix. */
  matrix(name: string, rows: number, cols: number, init: Initializer): Value[][] {
    const values: Value[][] = [];
    for (let r = 0; r < rows; r++) {
      const row: Value[] = [];
      for (let c = 0; c < cols; c++) {
        const v = new Value(init(), [], "", `${name}[${r}][${c}]`);
        row.push(v);
      }
      values.push(row);
    }
    this.params.push({ name, rows, cols, values });
    return values;
  }

  /** Create and register a length-n parameter vector (stored as one row). */
  vector(name: string, n: number, init: Initializer): Value[] {
    return this.matrix(name, 1, n, init)[0];
  }

  /** Flat list of every leaf Value, for the optimizer to iterate. */
  all(): Value[] {
    const out: Value[] = [];
    for (const p of this.params) for (const row of p.values) out.push(...row);
    return out;
  }

  /** Zero every parameter's gradient (call before each backward pass). */
  zeroGrad(): void {
    for (const v of this.all()) v.grad = 0;
  }
}
