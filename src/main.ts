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

/** Per-frame time budget (ms) for continuous "run" mode — scalar-autograd
 *  iterations vary a lot in cost with seqLen/embedDim, so budget time rather
 *  than a fixed step count to keep the UI responsive. */
const RUN_FRAME_BUDGET_MS = 24;
const RUN_MAX_STEPS_PER_FRAME = 50;

window.addEventListener("DOMContentLoaded", () => {
  const root = document.getElementById("app");
  if (!root) return;

  document.title = `Transformer Playground - Version ${VERSION}`;

  // --- grid scaffold ---
  root.classList.add("grid");
  // Left column: Run panel stacked above the dataset panel.
  const leftHost = document.createElement("div");
  leftHost.className = "left-col";
  const runHost = document.createElement("div");
  const datasetHost = document.createElement("div");
  leftHost.append(runHost, datasetHost);

  const topHost = section("top");
  const centerHost = section("center");
  const lossHost = section("loss");
  root.append(leftHost, topHost, centerHost, lossHost);
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

  // Remembered loss-row height so expanding restores the user's chosen size.
  let savedRowBottom = "210px";

  const ctx: AppContext = {
    state,
    apply(patch) {
      Object.assign(state, patch);
      if ("learningRate" in patch) state.optim.setLearningRate(state.learningRate);
      // Leaving "run" granularity stops a continuous run.
      if ("stepGranularity" in patch && patch.stepGranularity !== "run") {
        state.running = false;
      }
      // Collapsing panels resizes their grid rows (main owns the grid).
      if ("topCollapsed" in patch) {
        // auto-size: collapsed -> slim title bar; expanded -> content height.
        root.style.setProperty("--row-top", "auto");
      }
      if ("lossCollapsed" in patch) {
        if (patch.lossCollapsed) {
          const cur = root.style.getPropertyValue("--row-bottom").trim();
          if (cur && cur !== "34px") savedRowBottom = cur;
          root.style.setProperty("--row-bottom", "34px");
        } else {
          root.style.setProperty("--row-bottom", savedRowBottom);
        }
      }
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
  run = mountRunControls(runHost, ctx); // top of the left column

  // --- animation loop ---
  const tick = () => {
    if (state.running && state.stepGranularity === "run") {
      const t0 = performance.now();
      let steps = 0;
      while (
        performance.now() - t0 < RUN_FRAME_BUDGET_MS &&
        steps < RUN_MAX_STEPS_PER_FRAME
      ) {
        state.loop.stepIteration();
        steps++;
      }
      run.update(); // refresh counters/button
    }
    loss.update();
    network.update();
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
});
