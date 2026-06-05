/**
 * Headless smoke test (not part of the app build):
 *   npx esbuild scripts/smoke.ts --bundle --platform=node --format=esm --outfile=scripts/smoke.mjs && node scripts/smoke.mjs
 *
 * Covers: learning on the on-the-fly training stream, the staging machinery,
 * multi-step modes, save/load round-trip (bit-identical continuation), the
 * data-only carryOver path, rejection sampling against the test set, and the
 * sample-space math.
 */

import {
  generateTestSet,
  generateTrainExample,
  sampleKey,
  sampleSpaceSize,
} from "../src/tasks/datasets";
import { TransformerModel } from "../src/model/transformer";
import { SGD } from "../src/training/optimizer";
import { TrainingLoop, PIPELINE_STAGES } from "../src/training/loop";
import { Rng } from "../src/util/rng";

const TRAIN_PER_EPOCH = 48;

function build(task: "copy" | "parens", layers: 1 | 2 = 1) {
  const dataset = generateTestSet({
    task,
    vocabSize: 4,
    count: 12,
    seed: 7,
    minLen: 3,
    maxLen: 6,
  });
  const model = new TransformerModel(
    {
      task,
      vocabSize: 4,
      embedDim: 8,
      peScheme: "sinusoidal",
      numOutputLayers: layers,
      maxLen: 6,
    },
    // simple deterministic rng
    (() => {
      let s = 42;
      return () => ((s = (s * 16807) % 2147483647) / 2147483647);
    })(),
  );
  const optim = new SGD(model.store, { learningRate: 0.1 });
  return {
    dataset,
    model,
    optim,
    loop: new TrainingLoop(model, optim, dataset, new Rng(99), TRAIN_PER_EPOCH),
  };
}

// --- 1. learning check (copy, on-the-fly samples) ---
{
  const { loop } = build("copy");
  for (let i = 0; i < 300; i++) loop.stepIteration();
  const first = loop.iterHistory.slice(0, 20).reduce((a, p) => a + p.trainLoss, 0) / 20;
  const last = loop.iterHistory.slice(-20).reduce((a, p) => a + p.trainLoss, 0) / 20;
  console.log(`copy: first20 avg loss=${first.toFixed(3)}  last20 avg loss=${last.toFixed(3)}`);
  if (!(last < first * 0.8)) throw new Error("copy task did not learn (loss not decreasing)");
}

// --- 1b. classification path doesn't throw ---
{
  const { loop } = build("parens");
  for (let i = 0; i < 200; i++) loop.stepIteration();
  const first = loop.iterHistory.slice(0, 20).reduce((a, p) => a + p.trainLoss, 0) / 20;
  const last = loop.iterHistory.slice(-20).reduce((a, p) => a + p.trainLoss, 0) / 20;
  console.log(`parens: first20 avg loss=${first.toFixed(3)}  last20 avg loss=${last.toFixed(3)}`);
}

// --- 1c. two-layer FF path learns and traces hidden activations ---
{
  const { loop, model } = build("copy", 2);
  for (let i = 0; i < 300; i++) loop.stepIteration();
  const first = loop.iterHistory.slice(0, 20).reduce((a, p) => a + p.trainLoss, 0) / 20;
  const last = loop.iterHistory.slice(-20).reduce((a, p) => a + p.trainLoss, 0) / 20;
  console.log(`copy-2layer: first20 avg loss=${first.toFixed(3)}  last20 avg loss=${last.toFixed(3)}`);
  if (!(last < first * 0.8)) throw new Error("2-layer copy did not learn");
  if (!model.wFF) throw new Error("wFF should be registered for 2 layers");
  if (!loop.staged?.trace.hidden) throw new Error("trace.hidden missing for 2 layers");
}

// --- 2. multi-step modes: snapshots, epoch rollover, stepEpoch ---
{
  const { loop } = build("copy");
  loop.stepIteration();
  if (loop.staged?.phase !== "complete") throw new Error("stepIteration should leave a 'complete' snapshot");
  loop.stepEpoch();
  if (loop.epoch !== 1) throw new Error(`stepEpoch should land on epoch 1, got ${loop.epoch}`);
  if (loop.epochHistory.length !== 1) throw new Error(`expected 1 epoch point, got ${loop.epochHistory.length}`);
  if (loop.iteration !== TRAIN_PER_EPOCH) throw new Error("epoch should process trainPerEpoch samples");
  loop.stepLayer();
  if (loop.staged?.phase !== "forward" || loop.staged.stage !== 0) {
    throw new Error("stepLayer after a complete snapshot should start at forward/0");
  }
  console.log("multi-step: snapshot/epoch-rollover/stepEpoch OK");
}

// --- 2b. staging sequence ---
{
  const { loop } = build("copy");
  const N = PIPELINE_STAGES.length; // 8
  const seen: string[] = [];
  for (let i = 0; i < 2 * N; i++) {
    loop.stepLayer();
    const st = loop.staged!;
    seen.push(`${st.phase[0]}${st.stage}`);
  }
  const expect16 = [
    ...Array.from({ length: N }, (_, i) => `f${i}`),
    ...Array.from({ length: N }, (_, i) => `b${N - 1 - i}`),
  ];
  if (seen.join() !== expect16.join()) {
    throw new Error(`staging sequence wrong:\n got ${seen.join()}\n exp ${expect16.join()}`);
  }
  if (loop.iteration !== 0) throw new Error("iteration advanced too early");
  loop.stepLayer();
  if (loop.iteration !== 1) throw new Error(`iteration should be 1, got ${loop.iteration}`);
  console.log("staging: forward 0..7, backward 7..0, finalize+restart OK");
}

// --- 3. save/load round-trip: identical continuation (weights + history + rng) ---
{
  const a = build("copy");
  for (let i = 0; i < 100; i++) a.loop.stepIteration();

  const weights: Record<string, number[][]> = {};
  for (const p of a.model.store.params) {
    weights[p.name] = p.values.map((row) => row.map((v) => v.data));
  }
  const hist = a.loop.serialize();

  const b = build("copy");
  for (const p of b.model.store.params) {
    const w = weights[p.name];
    if (!w) throw new Error(`missing weights for ${p.name}`);
    for (let r = 0; r < p.rows; r++) {
      for (let c = 0; c < p.cols; c++) p.values[r][c].data = w[r][c];
    }
  }
  b.loop.restore(hist);

  for (let i = 0; i < 50; i++) {
    a.loop.stepIteration();
    b.loop.stepIteration();
  }
  const la = a.loop.iterHistory.slice(-50).map((p) => p.trainLoss);
  const lb = b.loop.iterHistory.slice(-50).map((p) => p.trainLoss);
  for (let i = 0; i < 50; i++) {
    if (la[i] !== lb[i]) {
      throw new Error(`round-trip diverged at step ${i}: ${la[i]} vs ${lb[i]}`);
    }
  }
  console.log("save/load round-trip: 50 continued steps identical OK (incl. generation stream)");
}

// --- 3b. data-only rebuild keeps model + history ---
{
  const a = build("copy");
  for (let i = 0; i < 30; i++) a.loop.stepIteration();
  const newData = generateTestSet({
    task: "copy",
    vocabSize: 4,
    count: 20,
    seed: 8,
    minLen: 3,
    maxLen: 6,
  });
  const loop2 = new TrainingLoop(a.model, a.optim, newData, new Rng(123), 60);
  loop2.carryOver(a.loop);
  if (loop2.iteration !== 30) throw new Error("carryOver lost iteration count");
  if (loop2.iterHistory.length !== 30) throw new Error("carryOver lost history");
  loop2.stepIteration();
  if (loop2.iteration !== 31) throw new Error("loop did not continue after carryOver");
  console.log("data-only rebuild: history carried over OK");
}

// --- 4. rejection sampling: train draws never collide with the test set ---
{
  const ds = generateTestSet({
    task: "copy",
    vocabSize: 3,
    count: 10,
    seed: 5,
    minLen: 3,
    maxLen: 4,
  });
  // space = 3^3 + 3^4 = 108; test = 10 — collisions WILL be drawn and must be rejected.
  const rng = new Rng(77);
  for (let i = 0; i < 500; i++) {
    const ex = generateTrainExample(ds, rng, i);
    if (ds.testKeys.has(sampleKey(ex.input))) {
      throw new Error(`train sample ${i} collides with the test set`);
    }
    if (ex.input.length < 3 || ex.input.length > 4) {
      throw new Error(`train sample length ${ex.input.length} outside [3,4]`);
    }
    if (ex.index !== i) throw new Error("train sample index should be the display index");
  }
  console.log("rejection sampling: 500 train draws disjoint from test, lengths OK");
}

// --- 5. sample-space math + test-set dedup on a tiny space ---
{
  if (sampleSpaceSize(2, 3, 3, true) !== 8) throw new Error("space(2,len=3 fixed) should be 8");
  if (sampleSpaceSize(4, 3, 6, false) !== 64 + 256 + 1024 + 4096) {
    throw new Error("space(4,3..6) wrong");
  }
  const tiny = generateTestSet({
    task: "copy",
    vocabSize: 2,
    count: 8,
    seed: 3,
    minLen: 3,
    maxLen: 3,
  });
  const keys = new Set(tiny.test.map((e) => sampleKey(e.input)));
  if (keys.size !== tiny.test.length) throw new Error("test set has duplicates");
  if (tiny.test.length > 8) throw new Error("test set exceeded the sample space");
  console.log(`sample space + dedup OK (tiny space produced ${tiny.test.length}/8 distinct)`);
}

console.log("SMOKE OK");
