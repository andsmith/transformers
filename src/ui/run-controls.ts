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
  type Checkbox,
  type Dropdown,
} from "./controls";
import type { PanelHandle } from "./top-panel";

export function mountRunControls(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "run-panel");
  host.innerHTML = "";

  // ---------- left 2/3: run controls ----------
  const left = document.createElement("div");
  left.className = "run-left";

  // Title row: big dark "Run" with the Step/Go button beside it (above the
  // dropdown, saving horizontal space).
  const stepBtn = makeButton("Step", () => ctx.step());
  const resetBtn = makeButton("Reset", () => ctx.reset());
  resetBtn.classList.add("btn-secondary", "btn-compact");
  resetBtn.title = "Re-initialize model weights and clear training history";
  const title = document.createElement("div");
  title.className = "run-title";
  title.textContent = "Run";
  const titleRow = document.createElement("div");
  titleRow.className = "run-title-row";
  titleRow.append(title, stepBtn, resetBtn);

  const granDropdown: Dropdown<StepGranularity> = makeDropdown(
    [
      { value: "layer", label: "1 layer" },
      { value: "iteration", label: "1 iteration" },
      { value: "epoch", label: "1 epoch" },
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
  slTitle.textContent = "Save / Load";

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

  right.append(slTitle, slRow, weightsCheck.el, histCheck.el, slError);

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

  function update(): void {
    const s = ctx.state;
    granDropdown.set(s.stepGranularity);
    const continuous = s.stepGranularity === "run" || s.stepGranularity === "epochs";
    stepBtn.textContent = continuous ? (s.running ? "Stop" : "Go") : "Step";
    counters.textContent = `iter ${s.loop.iteration} · epoch ${s.loop.epoch}`;
    constSizeCheck.set(s.vizConstantSize);
  }

  update();
  return { update };
}
