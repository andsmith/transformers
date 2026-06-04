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

function build(task: "copy" | "parens") {
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
      numOutputLayers: 1,
      maxLen: 6,
    },
    // simple deterministic rng
    (() => {
      let s = 42;
      return () => ((s = (s * 16807) % 2147483647) / 2147483647);
    })(),
  );
  const optim = new SGD(model.store, { learningRate: 0.1 });
  return { dataset, model, optim, loop: new TrainingLoop(model, optim, dataset) };
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

console.log("SMOKE OK");
