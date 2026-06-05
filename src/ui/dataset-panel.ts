/**
 * Dataset panel. "Dataset Generation" (display, vocabulary, sequence length —
 * changes reset training) and "Dataset Size" (train samples/epoch, test set
 * size, regenerate/seed — changes keep the model training). Training samples
 * are drawn on the fly; the list below shows the FIXED test set (the only
 * samples with stable ids).
 */

import type { AppContext, DisplayMode } from "../state";
import { testSetMax, TRAIN_PER_EPOCH_MAX } from "../state";
import type { Example } from "../tasks/types";
import { isClassification } from "../tasks/types";
import { MAX_SEQ_LEN_LIMIT, sampleSpaceSize, SPACE_HUGE } from "../tasks/datasets";
import { tokenChar, tokenColor, MAX_VOCAB } from "../tasks/grammar";
import {
  makeButton,
  makeCheckbox,
  makeFieldset,
  makeNumberInput,
  makeRadioGroup,
  makeRangeSlider,
  makeSlider,
  type Checkbox,
  type NumberInput,
  type RadioGroup,
  type RangeSlider,
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

  // Dual slider: min and max sequence length on one axis.
  const lenSlider: RangeSlider = makeRangeSlider({
    label: "Sequence length",
    min: 2,
    max: MAX_SEQ_LEN_LIMIT,
    step: 1,
    lo: ctx.state.minSeqLen,
    hi: ctx.state.maxSeqLen,
    onInput: (lo, hi) => ctx.apply({ minSeqLen: lo, maxSeqLen: hi }),
  });
  const fixedLenCheck: Checkbox = makeCheckbox(
    "All max length",
    ctx.state.fixedLength,
    (c) => ctx.apply({ fixedLength: c }),
  );
  fixedLenCheck.el.title = "Generate every sequence at exactly the maximum length";

  const lenRow = document.createElement("div");
  lenRow.className = "control-pair";
  lenRow.append(lenSlider.el, fixedLenCheck.el);
  // On-the-fly training: how many fresh samples make up one epoch.
  const trainSlider: Slider = makeSlider({
    label: "Train samples/epoch",
    min: 10,
    max: TRAIN_PER_EPOCH_MAX,
    step: 1,
    inline: true,
    value: ctx.state.trainPerEpoch,
    onInput: (v) => ctx.apply({ trainPerEpoch: v }),
  });
  // Fixed held-out test set (max dynamically clamped to 20% of the space).
  const testSlider: Slider = makeSlider({
    label: "Test set size",
    min: 0,
    max: 500,
    step: 1,
    inline: true,
    value: ctx.state.testSetSize,
    onInput: (v) => ctx.apply({ testSetSize: v }),
  });
  const spaceHint = document.createElement("div");
  spaceHint.className = "control-note";
  // Regenerate (compact) + seed entry + random-seed checkbox, one row.
  const regenBtn = makeButton("Regenerate", () => ctx.regenerate());
  regenBtn.classList.add("btn-compact");
  const seedInput: NumberInput = makeNumberInput(
    ctx.state.seed,
    (v) => ctx.apply({ seed: v }),
    "Random seed (reproducibility) — applied on Regenerate",
  );
  const randomCheck: Checkbox = makeCheckbox(
    "random",
    ctx.state.randomSeed,
    (c) => ctx.apply({ randomSeed: c }),
  );
  randomCheck.el.title = "Draw a fresh random seed on every Regenerate";
  const regenRow = document.createElement("div");
  regenRow.className = "regen-row";
  regenRow.append(regenBtn, seedInput.el, randomCheck.el);

  // --- two sub-boxes: Generation (resets training) | Size (does not) ---
  const genBox = makeFieldset("Dataset Generation");
  genBox.classList.add("sub-box");
  genBox.append(displayRadios.el, vocabRow, lenRow);

  const sizeBox = makeFieldset("Dataset Size");
  sizeBox.classList.add("sub-box");
  sizeBox.append(trainSlider.el, testSlider.el, spaceHint, regenRow);

  const controls = document.createElement("div");
  controls.className = "dataset-controls";
  controls.append(genBox, sizeBox);
  host.appendChild(controls);

  // --- examples (the fixed test set — the only samples with stable ids) ---
  const listCap = document.createElement("div");
  listCap.className = "caption";
  listCap.textContent = "Test set (held out from training)";
  host.appendChild(listCap);
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

  /** Render every test sample as an indexed row, sorted by id. Transduction
   *  shows input → output; classification shows input plus a
   *  balanced/unbalanced tag. */
  function renderList(list: Example[]): void {
    const classification = isClassification(ctx.state.task);
    const frag = document.createDocumentFragment();

    const sorted = [...list].sort((a, b) => a.index - b.index);

    // Fixed input-column width (max sequence in the list) so every arrow
    // lines up vertically, in both chars and squares modes.
    const maxInLen = sorted.reduce((m, e) => Math.max(m, e.input.length), 1);
    const perTok = ctx.state.display === "squares" ? 16 : 14; // token + gap px
    const inColW = `${maxInLen * perTok}px`;

    // Column header: "Input → Output".
    const head = document.createElement("div");
    head.className = "examples-header";
    const headIdx = document.createElement("span");
    headIdx.className = "ex-index";
    const headIn = document.createElement("span");
    headIn.textContent = "Input";
    headIn.style.minWidth = inColW;
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

    for (const ex of sorted) {
      const rowEl = document.createElement("div");
      rowEl.className = "example-row";

      const idx = document.createElement("span");
      idx.className = "ex-index";
      idx.textContent = `#${ex.index}`;
      const inSeq = renderSeq(ex.input);
      inSeq.style.minWidth = inColW;
      rowEl.append(idx, inSeq);

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
    displayRadios.set(s.display);
    vocabSlider.set(s.numSymbols);
    lenSlider.set(s.minSeqLen, s.maxSeqLen);
    fixedLenCheck.set(s.fixedLength);
    trainSlider.set(s.trainPerEpoch);

    // Dynamic test-set cap: 20% of the theoretical sample space (≤500).
    const maxTest = testSetMax(s);
    testSlider.setMax(maxTest);
    testSlider.set(s.testSetSize);
    const space = sampleSpaceSize(s.numSymbols, s.minSeqLen, s.maxSeqLen, s.fixedLength);
    const fmtSpace =
      space >= SPACE_HUGE
        ? "≥10¹⁵"
        : space >= 1e6
          ? space.toExponential(1)
          : String(Math.round(space));
    // What share of the whole space the test set occupies.
    const pct = (s.dataset.test.length / space) * 100;
    const fmtPct =
      pct >= 0.01 || pct === 0 ? `${pct.toFixed(2)}%` : `${pct.toExponential(1)}%`;
    spaceHint.textContent =
      `sample space ≈ ${fmtSpace}, test set ${fmtPct}` +
      (maxTest < 500 ? ` (capped at ${maxTest})` : "");

    seedInput.set(s.seed);
    seedInput.el.disabled = s.randomSeed;
    randomCheck.set(s.randomSeed);
    renderVocab();

    const listKey = s.display;
    if (s.dataset !== lastDataset || listKey !== lastListKey) {
      renderList(s.dataset.test);
      lastDataset = s.dataset;
      lastListKey = listKey;
    }
  }

  update();
  return { update };
}
