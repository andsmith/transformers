/**
 * The "Run" panel at the top of the left column, split 2/3 | 1/3:
 * left — step-granularity dropdown, Step/Go button, counters, constant-size
 * option; right — Save/Load of the experiment at three levels of detail
 * (experiment config / + model weights / + training history). The only dialog
 * is a tiny filename modal for saving; loading uses the native file picker.
 */

import type { AppContext } from "../state";
import type { StepGranularity } from "../training/loop";
import { buildSave } from "../persist";
import {
  makeButton,
  makeCheckbox,
  makeDropdown,
  makeSlider,
  type Checkbox,
  type Dropdown,
  type Slider,
} from "./controls";
import type { PanelHandle } from "./top-panel";

export function mountRunControls(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "run-panel");
  host.innerHTML = "";

  // ---------- left 2/3: run controls ----------
  const left = document.createElement("div");
  left.className = "run-left";

  // Title row: big dark "Run", Step (one click = one step), Go/Stop (Tick
  // mode = simulated Step clicks at the Speed rate), and the Speed slider.
  const stepBtn = makeButton("Step", () => ctx.step());
  stepBtn.classList.add("btn-compact");
  const goBtn = makeButton("Go", () => ctx.apply({ running: !ctx.state.running }));
  goBtn.classList.add("btn-compact");
  const speedSlider: Slider = makeSlider({
    label: "Speed",
    min: 0,
    max: 100,
    step: 1,
    inline: true,
    value: ctx.state.speed,
    format: (v) => (v >= 100 ? "max" : `${(0.5 * Math.pow(10, v / 99)).toFixed(1)} Hz`),
    onInput: (v) => ctx.apply({ speed: v }),
  });
  speedSlider.el.classList.add("speed-slider");
  speedSlider.el.title =
    "Tick rate while running: 0.5–5 Hz, or unthrottled at the right end";

  const title = document.createElement("div");
  title.className = "run-title";
  title.textContent = "Run";
  const titleRow = document.createElement("div");
  titleRow.className = "run-title-row";
  titleRow.append(title, stepBtn, goBtn, speedSlider.el);

  const granDropdown: Dropdown<StepGranularity> = makeDropdown(
    [
      { value: "layer", label: "step 1 layer" },
      { value: "iteration", label: "step 1 iteration" },
      { value: "epoch", label: "step 1 epoch" },
      { value: "run", label: "Run continuously" },
      { value: "epochs", label: "Run by epochs (fast)" },
    ],
    ctx.state.stepGranularity,
    (g) => ctx.apply({ stepGranularity: g }),
  );
  granDropdown.el.classList.add("full-width");

  const counters = document.createElement("div");
  counters.className = "hint";

  const constSizeCheck: Checkbox = makeCheckbox(
    "Constant-size viz elements",
    ctx.state.vizConstantSize,
    (c) => ctx.apply({ vizConstantSize: c }),
  );
  constSizeCheck.el.title =
    "Size visualization elements for the maximum sequence length so the " +
    "layout doesn't shift between samples (cell aspect ratio stretches).";

  left.append(titleRow, granDropdown.el, counters, constSizeCheck.el);

  // ---------- right 1/3: save / load ----------
  const right = document.createElement("div");
  right.className = "run-right";

  const slTitle = document.createElement("div");
  slTitle.className = "fieldset-title";
  slTitle.textContent = "Model State";

  let saveWeights = false;
  let saveHistory = false;
  const histCheck: Checkbox = makeCheckbox("Training history", false, (c) => {
    saveHistory = c;
  });
  const histInput = histCheck.el.querySelector("input")!;
  const weightsCheck: Checkbox = makeCheckbox("Model weights", false, (c) => {
    saveWeights = c;
    if (!c) {
      saveHistory = false;
      histCheck.set(false);
    }
    histInput.disabled = !c;
  });
  histInput.disabled = true;
  weightsCheck.el.title = "Include trained weights (resume training later)";
  histCheck.el.title =
    "Also include loss history and progress — restores the exact app state";

  const saveBtn = makeButton("Save", () => openSaveModal());
  const loadBtn = makeButton("Load", () => fileInput.click());
  const slRow = document.createElement("div");
  slRow.className = "run-row";
  slRow.append(saveBtn, loadBtn);

  const slError = document.createElement("div");
  slError.className = "hint sl-error";

  const resetBtn = makeButton("Reset", () => ctx.reset());
  resetBtn.classList.add("btn-secondary", "btn-compact");
  resetBtn.title = "Re-initialize model weights and clear training history";

  right.append(slTitle, slRow, weightsCheck.el, histCheck.el, slError, resetBtn);

  // Hidden native file picker for Load.
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = ".json,application/json";
  fileInput.style.display = "none";
  fileInput.addEventListener("change", () => {
    const f = fileInput.files?.[0];
    if (!f) return;
    f.text().then((text) => {
      const err = ctx.loadSave(text);
      slError.textContent = err ?? "";
      fileInput.value = ""; // allow re-loading the same file
    });
  });
  right.appendChild(fileInput);

  // Filename modal (the only dialog: filename + Save/Cancel).
  function openSaveModal(): void {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";
    const box = document.createElement("div");
    box.className = "modal-box";

    const cap = document.createElement("div");
    cap.className = "fieldset-title";
    const level = saveWeights ? (saveHistory ? "experiment + weights + history" : "experiment + weights") : "experiment only";
    cap.textContent = `Save (${level})`;

    const nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.className = "modal-filename";
    nameInput.value = `transformer-${ctx.state.task}.json`;

    const btnRow = document.createElement("div");
    btnRow.className = "run-row modal-btns";
    const doSave = makeButton("Save", () => {
      const file = buildSave(ctx.state, { weights: saveWeights, history: saveHistory });
      const name = nameInput.value.trim() || "transformer.json";
      const blob = new Blob([JSON.stringify(file)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = name.endsWith(".json") ? name : `${name}.json`;
      a.click();
      URL.revokeObjectURL(a.href);
      overlay.remove();
    });
    const cancel = makeButton("Cancel", () => overlay.remove());
    cancel.classList.add("btn-secondary");
    btnRow.append(doSave, cancel);

    box.append(cap, nameInput, btnRow);
    overlay.appendChild(box);
    overlay.addEventListener("pointerdown", (e) => {
      if (e.target === overlay) overlay.remove();
    });
    document.body.appendChild(overlay);
    nameInput.focus();
    nameInput.select();
  }

  host.append(left, right);

  const speedInput = speedSlider.el.querySelector("input")!;

  function update(): void {
    const s = ctx.state;
    granDropdown.set(s.stepGranularity);
    goBtn.textContent = s.running ? "Stop" : "Go";
    goBtn.classList.toggle("btn-stop", s.running);
    goBtn.classList.toggle("btn-go", !s.running);
    speedSlider.set(s.speed);
    // The continuous modes are unthrottled — Speed doesn't apply.
    const continuous = s.stepGranularity === "run" || s.stepGranularity === "epochs";
    speedInput.disabled = continuous;
    speedSlider.el.classList.toggle("disabled", continuous);
    counters.textContent = `iter ${s.loop.iteration} · epoch ${s.loop.epoch}`;
    constSizeCheck.set(s.vizConstantSize);
  }

  update();
  return { update };
}
