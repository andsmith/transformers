/**
 * App bootstrap. Owns the single AppState, implements the AppContext the panels
 * call into, and runs the animation-frame loop that drives continuous training
 * and canvas redraws.
 */

import "./styles.css";
import {
  createInitialState,
  rebuild,
  type AppContext,
  type AppState,
} from "./state";
import { mountTopPanel, type PanelHandle } from "./ui/top-panel";
import { mountDatasetPanel } from "./ui/dataset-panel";
import { mountLossPanel } from "./ui/loss-panel";
import { mountNetworkView } from "./ui/network-view";
import { mountRunControls } from "./ui/run-controls";
import { mountSplitters } from "./ui/splitters";
import { VERSION } from "./version";

/** Patch keys that require tearing down and rebuilding model + dataset. */
const REBUILD_KEYS = new Set<keyof AppState>([
  "task",
  "numSymbols",
  "embedDim",
  "peScheme",
  "numOutputLayers",
  "numExamples",
  "maxSeqLen",
  "fixedLength",
  "trainTestSplit",
]);

/** How many train iterations to run per frame while in continuous "run" mode. */
const STEPS_PER_FRAME = 5;

window.addEventListener("DOMContentLoaded", () => {
  const root = document.getElementById("app");
  if (!root) return;

  document.title = `Transformer Playground - Version ${VERSION}`;

  // --- grid scaffold ---
  root.classList.add("grid");
  const topHost = section("top");
  const centerHost = section("center");
  const datasetHost = section("dataset");
  const lossHost = section("loss");
  root.append(topHost, centerHost, datasetHost, lossHost);
  mountSplitters(root);

  function section(area: string): HTMLElement {
    const el = document.createElement("div");
    el.style.gridArea = area;
    el.dataset.area = area;
    return el;
  }

  const state = createInitialState();

  let top: PanelHandle;
  let dataset: PanelHandle;
  let loss: PanelHandle;
  let network: PanelHandle;
  let run: PanelHandle;

  function refreshAll(): void {
    top.update();
    dataset.update();
    loss.update();
    network.update();
    run.update();
  }

  function doRebuild(): void {
    const built = rebuild(state);
    state.dataset = built.dataset;
    state.model = built.model;
    state.optim = built.optim;
    state.loop = built.loop;
  }

  const ctx: AppContext = {
    state,
    apply(patch) {
      Object.assign(state, patch);
      if ("learningRate" in patch) state.optim.setLearningRate(state.learningRate);
      const needsRebuild = Object.keys(patch).some((k) =>
        REBUILD_KEYS.has(k as keyof AppState),
      );
      if (needsRebuild) {
        state.running = false;
        doRebuild();
      }
      refreshAll();
    },
    regenerate() {
      state.seed = (state.seed + 1) >>> 0;
      state.running = false;
      doRebuild();
      refreshAll();
    },
    step() {
      if (state.stepGranularity === "run") {
        state.running = !state.running;
        run.update();
        return;
      }
      state.running = false;
      if (state.stepGranularity === "layer") state.loop.stepLayer();
      else if (state.stepGranularity === "iteration") state.loop.stepIteration();
      else if (state.stepGranularity === "epoch") state.loop.stepEpoch();
      refreshAll();
    },
  };

  top = mountTopPanel(topHost, ctx);
  dataset = mountDatasetPanel(datasetHost, ctx);
  loss = mountLossPanel(lossHost, ctx);
  network = mountNetworkView(centerHost, ctx);
  run = mountRunControls(centerHost, ctx); // overlay on the network view

  // --- animation loop ---
  const tick = () => {
    if (state.running && state.stepGranularity === "run") {
      for (let i = 0; i < STEPS_PER_FRAME; i++) state.loop.stepIteration();
      run.update(); // refresh counters/button
    }
    loss.update();
    network.update();
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
});
