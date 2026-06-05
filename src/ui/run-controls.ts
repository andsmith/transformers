/**
 * The "Run" panel at the top of the left column: the step-granularity dropdown,
 * the Step/Go button, the iteration/epoch counters, and misc view options that
 * affect stepping (constant-size visualization).
 */

import type { AppContext } from "../state";
import type { StepGranularity } from "../training/loop";
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

  const title = document.createElement("div");
  title.className = "fieldset-title";
  title.textContent = "Run";

  const granDropdown: Dropdown<StepGranularity> = makeDropdown(
    [
      { value: "layer", label: "1 layer" },
      { value: "iteration", label: "1 iteration" },
      { value: "epoch", label: "1 epoch" },
      { value: "run", label: "Run continuously" },
    ],
    ctx.state.stepGranularity,
    (g) => ctx.apply({ stepGranularity: g }),
  );
  const stepBtn = makeButton("Step", () => ctx.step());

  const row = document.createElement("div");
  row.className = "run-row";
  row.append(granDropdown.el, stepBtn);

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

  host.append(title, row, counters, constSizeCheck.el);

  function update(): void {
    const s = ctx.state;
    granDropdown.set(s.stepGranularity);
    stepBtn.textContent =
      s.stepGranularity === "run" ? (s.running ? "Stop" : "Go") : "Step";
    counters.textContent = `iter ${s.loop.iteration} · epoch ${s.loop.epoch}`;
    constSizeCheck.set(s.vizConstantSize);
  }

  update();
  return { update };
}
