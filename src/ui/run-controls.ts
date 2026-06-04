/**
 * The "Run" control frame, overlaid on the top-right corner of the network
 * visualization: the step-granularity dropdown, the Step/Go button, and the
 * iteration/epoch counters.
 */

import type { AppContext } from "../state";
import type { StepGranularity } from "../training/loop";
import { makeButton, makeDropdown, type Dropdown } from "./controls";
import type { PanelHandle } from "./top-panel";

export function mountRunControls(host: HTMLElement, ctx: AppContext): PanelHandle {
  const overlay = document.createElement("div");
  overlay.className = "run-overlay";

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

  overlay.append(title, row, counters);
  host.appendChild(overlay);

  function update(): void {
    const s = ctx.state;
    granDropdown.set(s.stepGranularity);
    stepBtn.textContent =
      s.stepGranularity === "run" ? (s.running ? "Stop" : "Go") : "Step";
    counters.textContent = `iter ${s.loop.iteration} · epoch ${s.loop.epoch}`;
  }

  update();
  return { update };
}
