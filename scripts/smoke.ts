/**
 * Headless smoke test (not part of the app build):
 *   npx esbuild scripts/smoke.ts --bundle --platform=node --format=esm --outfile=scripts/smoke.mjs && node scripts/smoke.mjs
 *
 * 1. Trains the copy task for a few hundred iterations and asserts the loss
 *    drops (validates forward/backward end-to-end).
 * 2. Exercises stepLayer staging: phases/stage indices and per-sample step
 *    count must follow forward 0..7 then backward 7..0 then next sample.
 */

import { generateDataset } from "../src/tasks/datasets";
import { TransformerModel } from "../src/model/transformer";
import { SGD } from "../src/training/optimizer";
import { TrainingLoop, PIPELINE_STAGES } from "../src/training/loop";
import { Rng } from "../src/util/rng";

function build(task: "copy" | "parens", layers: 1 | 2 = 1) {
  const dataset = generateDataset({
    task,
    vocabSize: 4,
    count: 60,
    testFraction: 0.2,
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
    loop: new TrainingLoop(model, optim, dataset, new Rng(99)),
  };
}

// --- 1. learning check (copy) ---
{
  const { loop } = build("copy");
  for (let i = 0; i < 300; i++) loop.stepIteration();
  const first = loop.iterHistory.slice(0, 20).reduce((a, p) => a + p.trainLoss, 0) / 20;
  const last = loop.iterHistory.slice(-20).reduce((a, p) => a + p.trainLoss, 0) / 20;
  console.log(`copy: first20 avg loss=${first.toFixed(3)}  last20 avg loss=${last.toFixed(3)}`);
  if (!(last < first * 0.8)) throw new Error("copy task did not learn (loss not decreasing)");
}

// --- 1b. classification path doesn't throw and learns a bit ---
{
  const { loop } = build("parens");
  for (let i = 0; i < 200; i++) loop.stepIteration();
  const first = loop.iterHistory.slice(0, 20).reduce((a, p) => a + p.trainLoss, 0) / 20;
  const last = loop.iterHistory.slice(-20).reduce((a, p) => a + p.trainLoss, 0) / 20;
  console.log(`parens: first20 avg loss=${first.toFixed(3)}  last20 avg loss=${last.toFixed(3)}`);
}

// --- 1b2. two-layer FF path learns and traces hidden activations ---
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

// --- 1c. multi-step modes: snapshots, epoch rollover points, stepEpoch ---
{
  const { loop, dataset } = build("copy");
  loop.stepIteration();
  if (loop.staged?.phase !== "complete") throw new Error("stepIteration should leave a 'complete' snapshot");
  loop.stepEpoch();
  if (loop.epoch !== 1) throw new Error(`stepEpoch should land on epoch 1, got ${loop.epoch}`);
  if (loop.epochHistory.length !== 1) throw new Error(`expected 1 epoch point, got ${loop.epochHistory.length}`);
  if (loop.iteration !== dataset.train.length) throw new Error("epoch should process the whole train set");
  // A 'complete' snapshot then a layer step must start a fresh walkthrough.
  loop.stepLayer();
  if (loop.staged?.phase !== "forward" || loop.staged.stage !== 0) {
    throw new Error("stepLayer after a complete snapshot should start at forward/0");
  }
  console.log(`multi-step: snapshot/epoch-rollover/stepEpoch OK (epoch pts=${loop.epochHistory.length})`);
}

// --- 2. staging machinery ---
{
  const { loop } = build("copy");
  const N = PIPELINE_STAGES.length; // 8
  const seen: string[] = [];
  // First click starts the sample at forward/0; 2N-1 more clicks should end
  // the backward sweep at stage 0; the next click finishes + starts sample 2.
  for (let i = 0; i < 2 * N; i++) {
    loop.stepLayer();
    const st = loop.staged!;
    seen.push(`${st.phase[0]}${st.stage}`);
  }
  // forward f0..f7, then backward b7..b0 (2N clicks total)
  const expect16 = [
    ...Array.from({ length: N }, (_, i) => `f${i}`),
    ...Array.from({ length: N }, (_, i) => `b${N - 1 - i}`),
  ];
  if (seen.join() !== expect16.join()) {
    throw new Error(`staging sequence wrong:\n got ${seen.join()}\n exp ${expect16.join()}`);
  }
  if (loop.iteration !== 0) throw new Error("iteration advanced too early");
  loop.stepLayer(); // finalize + start next sample
  if (loop.iteration !== 1) throw new Error(`iteration should be 1, got ${loop.iteration}`);
  const st = loop.staged!;
  if (st.phase !== "forward" || st.stage !== 0) throw new Error("next sample did not start at forward/0");
  console.log("staging: forward 0..7, backward 7..0, finalize+restart OK");
}

// --- 3. save/load round-trip: weights + history + rng restore must continue
// the EXACT same run (identical loss sequence) as the original.
{
  const a = build("copy");
  for (let i = 0; i < 100; i++) a.loop.stepIteration();

  // "Save": capture weights + loop snapshot (what persist.buildSave stores).
  const weights: Record<string, number[][]> = {};
  for (const p of a.model.store.params) {
    weights[p.name] = p.values.map((row) => row.map((v) => v.data));
  }
  const hist = a.loop.serialize();

  // "Load": fresh build with the same config/seed, then restore.
  const b = build("copy");
  for (const p of b.model.store.params) {
    const w = weights[p.name];
    if (!w) throw new Error(`missing weights for ${p.name}`);
    for (let r = 0; r < p.rows; r++) {
      for (let c = 0; c < p.cols; c++) p.values[r][c].data = w[r][c];
    }
  }
  b.loop.restore(hist);

  // Continue both 50 steps — losses must match exactly.
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
  if (b.loop.iteration !== a.loop.iteration) throw new Error("iteration mismatch after restore");
  console.log("save/load round-trip: 50 continued steps identical OK");
}

// --- 3b. data-only rebuild keeps model + history (dataset size changes) ---
{
  const a = build("copy");
  for (let i = 0; i < 30; i++) a.loop.stepIteration();
  const newData = generateDataset({
    task: "copy",
    vocabSize: 4,
    count: 120, // different size
    testFraction: 0.1,
    seed: 8,
    minLen: 3,
    maxLen: 6,
  });
  const loop2 = new TrainingLoop(a.model, a.optim, newData, new Rng(123));
  loop2.carryOver(a.loop);
  if (loop2.iteration !== 30) throw new Error("carryOver lost iteration count");
  if (loop2.iterHistory.length !== 30) throw new Error("carryOver lost history");
  loop2.stepIteration();
  if (loop2.iteration !== 31) throw new Error("loop did not continue after carryOver");
  console.log("data-only rebuild: history carried over OK");
}

// --- 4. min sequence length honored ---
{
  const ds = generateDataset({
    task: "copy",
    vocabSize: 4,
    count: 200,
    testFraction: 0,
    seed: 5,
    minLen: 4,
    maxLen: 6,
  });
  for (const ex of ds.examples) {
    if (ex.input.length < 4 || ex.input.length > 6) {
      throw new Error(`length ${ex.input.length} outside [4,6]`);
    }
  }
  console.log("min/max sequence length honored OK");
}

console.log("SMOKE OK");
