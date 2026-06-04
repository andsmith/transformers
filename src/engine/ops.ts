/**
 * Vector/matrix helpers built on top of the scalar {@link Value} autodiff.
 *
 * Everything here operates on `Value[]` (vectors) and `Value[][]` (matrices,
 * row-major) so that gradients flow through to the underlying scalar graph.
 * Sizes in this app are tiny (vocab <= 10, short strings), so the scalar
 * overhead is irrelevant and we get full per-element gradient visibility.
 */

import { Value } from "./value";

/** Dot product of two equal-length vectors. */
export function dot(a: Value[], b: Value[]): Value {
  if (a.length !== b.length) {
    throw new Error(`dot: length mismatch ${a.length} vs ${b.length}`);
  }
  let acc = a[0].mul(b[0]);
  for (let i = 1; i < a.length; i++) acc = acc.add(a[i].mul(b[i]));
  return acc;
}

/** Matrix (rows x cols) times vector (cols) -> vector (rows). */
export function matVec(m: Value[][], v: Value[]): Value[] {
  return m.map((row) => dot(row, v));
}

/** Elementwise add of two equal-length vectors. */
export function addVec(a: Value[], b: Value[]): Value[] {
  if (a.length !== b.length) {
    throw new Error(`addVec: length mismatch ${a.length} vs ${b.length}`);
  }
  return a.map((x, i) => x.add(b[i]));
}

/** Scale every element of a vector by a constant or Value. */
export function scaleVec(v: Value[], s: Value | number): Value[] {
  return v.map((x) => x.mul(s));
}

/**
 * Numerically-stable softmax over a vector of logits.
 * Subtracts the (constant) max before exponentiating.
 */
export function softmax(logits: Value[]): Value[] {
  let maxData = logits[0].data;
  for (const l of logits) if (l.data > maxData) maxData = l.data;

  const exps = logits.map((l) => l.sub(maxData).exp());
  let sum = exps[0];
  for (let i = 1; i < exps.length; i++) sum = sum.add(exps[i]);
  return exps.map((e) => e.div(sum));
}

/**
 * Cross-entropy between a target class index and a vector of logits.
 * Returns -log(softmax(logits)[target]).
 */
export function crossEntropy(logits: Value[], target: number): Value {
  const probs = softmax(logits);
  return probs[target].log().neg();
}

/** Mean-squared error between a scalar prediction and a target number. */
export function mse(pred: Value, target: number): Value {
  return pred.sub(target).pow(2);
}

/** Sum a list of Values (e.g. to combine per-position losses). */
export function sum(values: Value[]): Value {
  let acc = values[0];
  for (let i = 1; i < values.length; i++) acc = acc.add(values[i]);
  return acc;
}
