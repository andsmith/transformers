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
import { mountStatusPanel } from "./ui/status-panel";
import { mountHistoryPanel } from "./ui/history-panel";
import { mountSplitters } from "./ui/splitters";
import { applySave, type SaveFile } from "./persist";
import { VERSION } from "./version";

/** Patch keys that require tearing down and rebuilding model + dataset. */
const REBUILD_KEYS = new Set<keyof AppState>([
  "task",
  "numSymbols",
  "embedDim",
  "peScheme",
  "numOutputLayers",
  "numExamples",
  "minSeqLen",
  "maxSeqLen",
  "fixedLength",
  "trainTestSplit",
]);

/** Per-frame time budget (ms) for continuous "run" mode — scalar-autograd
 *  iterations vary a lot in cost with seqLen/embedDim, so budget time rather
 *  than a fixed step count to keep the UI responsive. */
const RUN_FRAME_BUDGET_MS = 24;
const RUN_MAX_STEPS_PER_FRAME = 50;
/** "Run by epochs" trains harder per frame and redraws only per epoch. */
const EPOCHS_FRAME_BUDGET_MS = 40;

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
  // Bottom row: Status | History | Loss.
  const bottomHost = section("loss");
  bottomHost.className = "bottom-row";
  const statusHost = document.createElement("div");
  const historyHost = document.createElement("div");
  const lossHost = document.createElement("div");
  lossHost.className = "loss-host";
  bottomHost.append(statusHost, historyHost, lossHost);
  root.append(leftHost, topHost, centerHost, bottomHost);
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
  let status: PanelHandle;
  let history: PanelHandle;

  function refreshAll(): void {
    top.update();
    dataset.update();
    loss.update();
    network.update();
    run.update();
    status.update();
    history.update();
    // Collapsing the loss row hides its companions too.
    statusHost.style.display = state.lossCollapsed ? "none" : "";
    historyHost.style.display = state.lossCollapsed ? "none" : "";
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
      // Leaving a continuous granularity stops the run.
      if (
        "stepGranularity" in patch &&
        patch.stepGranularity !== "run" &&
        patch.stepGranularity !== "epochs"
      ) {
        state.running = false;
      }
      if ("uiFontPx" in patch) {
        root.style.setProperty("--ui-font-px", `${state.uiFontPx}px`);
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
      // Deterministic mode reuses the entered seed (reproducible); random mode
      // draws a fresh one each time (and displays it).
      if (state.randomSeed) state.seed = (Math.random() * 2 ** 32) >>> 0;
      state.running = false;
      doRebuild();
      refreshAll();
    },
    step() {
      if (state.stepGranularity === "run" || state.stepGranularity === "epochs") {
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
    loadSave(json) {
      let parsed: SaveFile;
      try {
        parsed = JSON.parse(json) as SaveFile;
      } catch {
        return "File is not valid JSON.";
      }
      const err = applySave(parsed, state, doRebuild);
      refreshAll();
      return err;
    },
  };

  top = mountTopPanel(topHost, ctx);
  dataset = mountDatasetPanel(datasetHost, ctx);
  loss = mountLossPanel(lossHost, ctx);
  network = mountNetworkView(centerHost, ctx);
  run = mountRunControls(runHost, ctx); // top of the left column
  status = mountStatusPanel(statusHost, ctx);
  history = mountHistoryPanel(historyHost, ctx);

  // --- animation loop ---
  let lastDrawnEpoch = -1;
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
      loss.update();
      network.update();
      status.update();
      history.update();
    } else if (state.running && state.stepGranularity === "epochs") {
      // Fastest mode: bigger budget, no per-sample snapshots (except each
      // epoch's last sample), and the UI redraws only at epoch boundaries.
      const t0 = performance.now();
      while (performance.now() - t0 < EPOCHS_FRAME_BUDGET_MS) {
        state.loop.stepIteration(false);
      }
      if (state.loop.epoch !== lastDrawnEpoch) {
        lastDrawnEpoch = state.loop.epoch;
        run.update();
        loss.update();
        network.update();
        status.update();
        history.update();
      }
    } else {
      loss.update();
      network.update();
      status.update();
      history.update();
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
});
