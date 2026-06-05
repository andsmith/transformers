/**
 * Dataset panel: display mode, vocabulary / sequence-length / size / split
 * controls, a regenerate button, a train/test view selector, and an indexed
 * list of every sample in the selected split (input → output for transduction,
 * input + balanced/unbalanced tag for classification).
 */

import type { AppContext, DatasetView, DisplayMode } from "../state";
import type { Example } from "../tasks/types";
import { isClassification } from "../tasks/types";
import { datasetSummary, MAX_SEQ_LEN_LIMIT } from "../tasks/datasets";
import { tokenChar, tokenColor, MAX_VOCAB } from "../tasks/grammar";
import {
  makeButton,
  makeCheckbox,
  makeDropdown,
  makeRadioGroup,
  makeSlider,
  type Checkbox,
  type Dropdown,
  type RadioGroup,
  type Slider,
} from "./controls";
import type { PanelHandle } from "./top-panel";

export function mountDatasetPanel(host: HTMLElement, ctx: AppContext): PanelHandle {
  host.classList.add("panel", "dataset-panel");
  host.innerHTML = "";

  const header = document.createElement("div");
  header.className = "panel-header";
  header.textContent = "Dataset";
  host.appendChild(header);

  const summary = document.createElement("div");
  summary.className = "hint";
  host.appendChild(summary);

  // --- controls ---
  const displayRadios: RadioGroup<DisplayMode> = makeRadioGroup(
    [
      { value: "chars", label: "Characters" },
      { value: "squares", label: "Colored squares" },
    ],
    ctx.state.display,
    (d) => ctx.apply({ display: d }),
  );

  const vocabSlider: Slider = makeSlider({
    label: "Vocabulary size",
    min: 2,
    max: MAX_VOCAB,
    step: 1,
    value: ctx.state.numSymbols,
    onInput: (v) => ctx.apply({ numSymbols: v }),
  });

  // Live preview of the vocabulary (chars or colored squares).
  const vocabPreview = document.createElement("div");
  vocabPreview.className = "labeled";
  const vocabCap = document.createElement("div");
  vocabCap.className = "caption";
  vocabCap.textContent = "Vocabulary";
  const vocabTokens = document.createElement("div");
  vocabTokens.className = "vocab-tokens";
  vocabPreview.append(vocabCap, vocabTokens);

  const vocabRow = document.createElement("div");
  vocabRow.className = "control-pair";
  vocabRow.append(vocabSlider.el, vocabPreview);

  const maxLenSlider: Slider = makeSlider({
    label: "Max sequence length",
    min: 2,
    max: MAX_SEQ_LEN_LIMIT,
    step: 1,
    value: ctx.state.maxSeqLen,
    onInput: (v) => ctx.apply({ maxSeqLen: v }),
  });
  const fixedLenCheck: Checkbox = makeCheckbox(
    "All max length",
    ctx.state.fixedLength,
    (c) => ctx.apply({ fixedLength: c }),
  );
  fixedLenCheck.el.title = "Generate every sequence at exactly the maximum length";

  const lenRow = document.createElement("div");
  lenRow.className = "control-pair";
  lenRow.append(maxLenSlider.el, fixedLenCheck.el);
  const countSlider: Slider = makeSlider({
    label: "Examples",
    min: 10,
    max: 5000,
    step: 10,
    value: ctx.state.numExamples,
    onInput: (v) => ctx.apply({ numExamples: v }),
  });
  const splitSlider: Slider = makeSlider({
    label: "Train / test split",
    min: 0,
    max: 50,
    step: 1,
    value: Math.round(ctx.state.trainTestSplit * 100),
    format: (v) => `${v}% test`,
    onInput: (v) => ctx.apply({ trainTestSplit: v / 100 }),
  });
  const regenBtn = makeButton("Regenerate", () => ctx.regenerate());

  const viewDropdown: Dropdown<DatasetView> = makeDropdown(
    [
      { value: "train", label: "Train set" },
      { value: "test", label: "Test set" },
    ],
    ctx.state.datasetView,
    (v) => ctx.apply({ datasetView: v }),
  );
  const viewRow = document.createElement("div");
  viewRow.className = "labeled";
  const viewCap = document.createElement("div");
  viewCap.className = "caption";
  viewCap.textContent = "Viewing";
  viewRow.append(viewCap, viewDropdown.el);

  const controls = document.createElement("div");
  controls.className = "dataset-controls";
  controls.append(
    displayRadios.el,
    vocabRow,
    lenRow,
    countSlider.el,
    splitSlider.el,
    regenBtn,
    viewRow,
  );
  host.appendChild(controls);

  // --- examples ---
  const examplesEl = document.createElement("div");
  examplesEl.className = "examples";
  host.appendChild(examplesEl);

  function renderToken(id: number): HTMLElement {
    const s = ctx.state;
    if (s.display === "squares") {
      const sq = document.createElement("span");
      sq.className = "tok square";
      sq.style.backgroundColor = tokenColor(id, s.numSymbols);
      sq.title = tokenChar(s.task, id, s.numSymbols);
      return sq;
    }
    const span = document.createElement("span");
    span.className = "tok char";
    span.textContent = tokenChar(s.task, id, s.numSymbols);
    return span;
  }

  function renderSeq(ids: number[]): HTMLElement {
    const row = document.createElement("span");
    row.className = "seq";
    for (const id of ids) row.appendChild(renderToken(id));
    return row;
  }

  /** Re-render the vocabulary preview (all token ids, current display mode). */
  function renderVocab(): void {
    vocabTokens.innerHTML = "";
    for (let id = 0; id < ctx.state.numSymbols; id++) {
      vocabTokens.appendChild(renderToken(id));
    }
  }

  /** Render every sample in `list` as an indexed row (uses each sample's
   *  stable global index), always sorted by index — training visits samples
   *  in a per-epoch shuffled order, but the list stays stable. Transduction
   *  shows input → output; classification shows input plus a
   *  balanced/unbalanced tag. */
  function renderList(list: Example[]): void {
    const classification = isClassification(ctx.state.task);
    const frag = document.createDocumentFragment();

    // Column header: "Input → Output".
    const head = document.createElement("div");
    head.className = "examples-header";
    const headIdx = document.createElement("span");
    headIdx.className = "ex-index";
    const headIn = document.createElement("span");
    headIn.textContent = "Input";
    head.append(headIdx, headIn);
    const headOut = document.createElement("span");
    headOut.textContent = "Output";
    if (classification) {
      headOut.className = "head-right";
      head.append(headOut);
    } else {
      const headArrow = document.createElement("span");
      headArrow.className = "arrow";
      headArrow.textContent = "→";
      head.append(headArrow, headOut);
    }
    frag.appendChild(head);

    const sorted = [...list].sort((a, b) => a.index - b.index);
    for (const ex of sorted) {
      const rowEl = document.createElement("div");
      rowEl.className = "example-row";

      const idx = document.createElement("span");
      idx.className = "ex-index";
      idx.textContent = `#${ex.index}`;
      rowEl.append(idx, renderSeq(ex.input));

      if (classification) {
        const balanced = ex.output[0] === 1;
        const tag = document.createElement("span");
        tag.className = `class-tag ${balanced ? "balanced" : "unbalanced"}`;
        tag.textContent = balanced ? "balanced" : "unbalanced";
        rowEl.append(tag);
      } else {
        const arrow = document.createElement("span");
        arrow.className = "arrow";
        arrow.textContent = "→";
        rowEl.append(arrow, renderSeq(ex.output));
      }
      frag.appendChild(rowEl);
    }
    examplesEl.innerHTML = "";
    examplesEl.appendChild(frag);
  }

  // The example list can be large (up to 5000 rows), so only rebuild it when
  // something that affects it actually changes.
  let lastDataset = ctx.state.dataset;
  let lastListKey = "";

  function update(): void {
    const s = ctx.state;
    summary.textContent = datasetSummary(s.dataset);
    displayRadios.set(s.display);
    vocabSlider.set(s.numSymbols);
    maxLenSlider.set(s.maxSeqLen);
    fixedLenCheck.set(s.fixedLength);
    countSlider.set(s.numExamples);
    splitSlider.set(Math.round(s.trainTestSplit * 100));
    viewDropdown.set(s.datasetView);
    renderVocab();

    const listKey = `${s.datasetView}|${s.display}`;
    if (s.dataset !== lastDataset || listKey !== lastListKey) {
      renderList(s.datasetView === "test" ? s.dataset.test : s.dataset.train);
      lastDataset = s.dataset;
      lastListKey = listKey;
    }
  }

  update();
  return { update };
}
