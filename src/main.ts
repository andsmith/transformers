/**
 * App bootstrap. Owns the single AppState, implements the AppContext the panels
 * call into, and runs the animation-frame loop that drives continuous training
 * and canvas redraws.
 */

import "./styles.css";
import {
  createInitialState,
  rebuild,
  rebuildDataset,
  effectiveTrainPerEpoch,
  maxNestingDepth,
  testSetMax,
  type AppContext,
  type AppState,
} from "./state";
import { TrainingLoop } from "./training/loop";
import { maxDelims } from "./tasks/grammar";
import { Rng } from "./util/rng";
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

/** Dataset-GENERATION / model keys: changing these resets training entirely. */
const REBUILD_KEYS = new Set<keyof AppState>([
  "task",
  "numSymbols",
  "embedDim",
  "peScheme",
  "numOutputLayers",
  "minSeqLen",
  "maxSeqLen",
  "fixedLength",
  "uniformLen",
  "parensMaxDepth",
  "parensNoMixedNesting",
  "parensDelims",
  "grokFilters",
]);

/** Dataset-SIZE keys: regenerate the test set / change the epoch length but
 *  keep the model and its training history. */
const DATA_ONLY_KEYS = new Set<keyof AppState>(["trainPerEpoch", "testSetSize"]);

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
  // Left column: Run, Dataset (controls), Test Set — each frame separated by a
  // draggable divider.
  const leftHost = document.createElement("div");
  leftHost.className = "left-col";
  const runHost = document.createElement("div");
  const datasetGenHost = document.createElement("div");
  const testHost = document.createElement("div");
  const split1 = document.createElement("div");
  const split2 = document.createElement("div");
  split1.className = "left-splitter";
  split2.className = "left-splitter";
  leftHost.append(runHost, split1, datasetGenHost, split2, testHost);
  mountLeftSplitter(leftHost, runHost, split1);
  mountLeftSplitter(leftHost, datasetGenHost, split2);

  const topHost = section("top");
  const centerHost = section("center");
  // Bottom row: Status | History | Loss, with draggable vertical dividers.
  const bottomHost = section("loss");
  bottomHost.className = "bottom-row";
  const statusHost = document.createElement("div");
  const historyHost = document.createElement("div");
  const lossHost = document.createElement("div");
  lossHost.className = "loss-host";
  const vsplit1 = document.createElement("div");
  const vsplit2 = document.createElement("div");
  vsplit1.className = "bottom-splitter";
  vsplit2.className = "bottom-splitter";
  bottomHost.append(statusHost, vsplit1, historyHost, vsplit2, lossHost);
  mountBottomSplitter(bottomHost, statusHost, vsplit1);
  mountBottomSplitter(bottomHost, historyHost, vsplit2);
  root.append(leftHost, topHost, centerHost, bottomHost);
  mountSplitters(root);

  function section(area: string): HTMLElement {
    const el = document.createElement("div");
    el.style.gridArea = area;
    el.dataset.area = area;
    return el;
  }

  /** Drag the divider between the Run and Dataset frames to resize them. */
  function mountLeftSplitter(col: HTMLElement, run: HTMLElement, handle: HTMLElement): void {
    let active = false;
    let startY = 0;
    let startH = 0;
    handle.addEventListener("pointerdown", (e) => {
      active = true;
      startY = e.clientY;
      startH = run.offsetHeight;
      handle.setPointerCapture(e.pointerId);
      handle.classList.add("dragging");
      document.body.classList.add("resizing");
      e.preventDefault();
    });
    handle.addEventListener("pointermove", (e) => {
      if (!active) return;
      const h = Math.max(
        60,
        Math.min(col.clientHeight - 120, startH + (e.clientY - startY)),
      );
      run.style.flex = "0 0 auto";
      run.style.height = `${h}px`;
    });
    const end = (e: PointerEvent) => {
      if (!active) return;
      active = false;
      handle.releasePointerCapture?.(e.pointerId);
      handle.classList.remove("dragging");
      document.body.classList.remove("resizing");
    };
    handle.addEventListener("pointerup", end);
    handle.addEventListener("pointercancel", end);
  }

  /** Drag a vertical divider to resize the panel to its left (bottom row). */
  function mountBottomSplitter(row: HTMLElement, panel: HTMLElement, handle: HTMLElement): void {
    let active = false;
    let startX = 0;
    let startW = 0;
    handle.addEventListener("pointerdown", (e) => {
      active = true;
      startX = e.clientX;
      startW = panel.offsetWidth;
      handle.setPointerCapture(e.pointerId);
      handle.classList.add("dragging");
      document.body.classList.add("resizing");
      e.preventDefault();
    });
    handle.addEventListener("pointermove", (e) => {
      if (!active) return;
      const w = Math.max(
        80,
        Math.min(row.clientWidth - 220, startW + (e.clientX - startX)),
      );
      panel.style.flex = "0 0 auto";
      panel.style.width = `${w}px`;
      panel.style.aspectRatio = "auto"; // history panel: free width/height
    });
    const end = (e: PointerEvent) => {
      if (!active) return;
      active = false;
      handle.releasePointerCapture?.(e.pointerId);
      handle.classList.remove("dragging");
      document.body.classList.remove("resizing");
    };
    handle.addEventListener("pointerup", end);
    handle.addEventListener("pointercancel", end);
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
    // Collapsing the loss row hides its companions (and their dividers) too.
    const hide = state.lossCollapsed ? "none" : "";
    statusHost.style.display = hide;
    historyHost.style.display = hide;
    vsplit1.style.display = hide;
    vsplit2.style.display = hide;
  }

  function doRebuild(): void {
    const built = rebuild(state);
    state.dataset = built.dataset;
    state.model = built.model;
    state.optim = built.optim;
    state.loop = built.loop;
  }

  /** Regenerate just the test set / epoch length; the model keeps its weights
   *  and the loss curves keep growing (dataset-size changes and Regenerate). */
  function doRebuildDataOnly(): void {
    state.dataset = rebuildDataset(state);
    const loop = new TrainingLoop(
      state.model,
      state.optim,
      state.dataset,
      new Rng(state.seed ^ 0x51ed270b),
      effectiveTrainPerEpoch(state, state.dataset.test.length),
    );
    loop.carryOver(state.loop);
    state.loop = loop;
  }

  /** One Step click's worth of computation, at the chosen granularity. */
  function doOneStep(): void {
    const g = state.stepGranularity;
    if (g === "layer") state.loop.stepLayer();
    else if (g === "iteration" || g === "run") state.loop.stepIteration();
    else state.loop.stepEpoch(); // "epoch" and "epochs"
  }

  // Remembered loss-row height so expanding restores the user's chosen size.
  let savedRowBottom = "210px";

  const ctx: AppContext = {
    state,
    apply(patch) {
      Object.assign(state, patch);
      if ("learningRate" in patch) state.optim.setLearningRate(state.learningRate);
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
      const keys = Object.keys(patch);
      if (keys.some((k) => REBUILD_KEYS.has(k as keyof AppState))) {
        // Generation/model change: full reset. Keep dependent values in range.
        state.parensMaxDepth = Math.min(
          state.parensMaxDepth,
          maxNestingDepth(state.maxSeqLen),
        );
        state.parensDelims = Math.max(
          1,
          Math.min(state.parensDelims, maxDelims(state.numSymbols)),
        );
        state.testSetSize = Math.min(state.testSetSize, testSetMax(state));
        state.running = false;
        doRebuild();
      } else if (keys.some((k) => DATA_ONLY_KEYS.has(k as keyof AppState))) {
        // Size-only change: new data, same model, history continues.
        doRebuildDataOnly();
      }
      refreshAll();
    },
    regenerate() {
      // Deterministic mode reuses the entered seed (reproducible); random mode
      // draws a fresh one each time (and displays it). The model keeps
      // training — only the data is re-rolled.
      if (state.randomSeed) state.seed = (Math.random() * 2 ** 32) >>> 0;
      doRebuildDataOnly();
      refreshAll();
    },
    step() {
      // A manual Step pauses Tick mode and advances one step of the chosen size.
      state.running = false;
      doOneStep();
      refreshAll();
    },
    reset() {
      // Fresh model weights + empty training history; the dataset is
      // regenerated from the unchanged seed, so the data stays the same.
      state.running = false;
      doRebuild();
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
  dataset = mountDatasetPanel(datasetGenHost, testHost, ctx);
  loss = mountLossPanel(lossHost, ctx);
  network = mountNetworkView(centerHost, ctx);
  run = mountRunControls(runHost, ctx); // top of the left column
  status = mountStatusPanel(statusHost, ctx);
  history = mountHistoryPanel(historyHost, ctx);

  // --- animation loop / Tick mode ---
  // While running, Go simulates clicking Step at the Speed slider's rate;
  // at the slider's max position each granularity uses its unthrottled path.
  let lastDrawnEpoch = -1;
  let tickAccum = 0;
  let lastFrameT = performance.now();
  // Throughput sampler: record samples/sec into the loop's timing history at
  // most once per second (skipped when no training happened in the window).
  let spsLastT = performance.now();
  let spsLastIter = 0;

  const tick = () => {
    const now = performance.now();
    const dt = now - lastFrameT;
    lastFrameT = now;

    if (now - spsLastT >= 1000) {
      const delta = state.loop.iteration - spsLastIter;
      if (delta > 0) {
        state.loop.timingHistory.push({
          x: state.loop.iteration,
          sps: (delta * 1000) / (now - spsLastT),
        });
      }
      spsLastT = now;
      spsLastIter = state.loop.iteration;
    }

    let redraw = true;
    if (state.running) {
      const gran = state.stepGranularity;
      if (gran === "run") {
        // Continuous iterations, unthrottled (Speed does not apply).
        const t0 = performance.now();
        let steps = 0;
        while (
          performance.now() - t0 < RUN_FRAME_BUDGET_MS &&
          steps < RUN_MAX_STEPS_PER_FRAME
        ) {
          state.loop.stepIteration();
          steps++;
        }
        run.update();
      } else if (gran === "epochs") {
        // Fastest: no per-sample snapshots; UI refreshed only per epoch.
        const t0 = performance.now();
        while (performance.now() - t0 < EPOCHS_FRAME_BUDGET_MS) {
          state.loop.stepIteration(false);
        }
        redraw = state.loop.epoch !== lastDrawnEpoch;
        if (redraw) {
          lastDrawnEpoch = state.loop.epoch;
          run.update();
        }
      } else if (state.speed >= 100) {
        // Unthrottled "max" per step granularity.
        if (gran === "layer") {
          state.loop.stepLayer(); // one pipeline stage per frame
        } else if (gran === "iteration") {
          const t0 = performance.now();
          let steps = 0;
          while (
            performance.now() - t0 < RUN_FRAME_BUDGET_MS &&
            steps < RUN_MAX_STEPS_PER_FRAME
          ) {
            state.loop.stepIteration();
            steps++;
          }
        } else {
          state.loop.stepEpoch(); // one full epoch per frame
        }
        run.update();
      } else {
        // Timed ticks: 0 -> 0.5 Hz ... 99 -> 5 Hz (log scale).
        const hz = 0.5 * Math.pow(10, state.speed / 99);
        const interval = 1000 / hz;
        tickAccum += dt;
        if (tickAccum >= interval) {
          tickAccum = Math.min(tickAccum - interval, interval); // no catch-up bursts
          doOneStep();
          run.update();
        }
      }
    } else {
      tickAccum = 0;
    }

    if (redraw) {
      loss.update();
      network.update();
      status.update();
      history.update();
      // Keep the test-set marks and sort/out buttons current during runs
      // (the list rebuild is memoized, so this is cheap between epochs).
      dataset.update();
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
});
