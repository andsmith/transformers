/**
 * Draggable splitters between the page panels. Each sits in one of the grid's
 * gutter tracks and, on drag, rewrites the grid's track-size CSS variables
 * (`--col-right`, `--row-top`, `--row-bottom`). The canvas panels re-measure
 * themselves every animation frame, so they follow the new sizes automatically.
 */

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function makeSplitter(kind: "vertical" | "horizontal", area: string): HTMLElement {
  const el = document.createElement("div");
  el.className = `splitter ${kind}`;
  el.style.gridArea = area;
  el.setAttribute("role", "separator");
  el.setAttribute(
    "aria-orientation",
    kind === "vertical" ? "vertical" : "horizontal",
  );
  return el;
}

/** Wire pointer drag on `el`, calling `onMove` with each move event. */
function dragHandle(el: HTMLElement, onMove: (e: PointerEvent) => void): void {
  let active = false;
  el.addEventListener("pointerdown", (e) => {
    active = true;
    el.setPointerCapture(e.pointerId);
    el.classList.add("dragging");
    document.body.classList.add("resizing");
    e.preventDefault();
  });
  el.addEventListener("pointermove", (e) => {
    if (active) onMove(e);
  });
  const end = (e: PointerEvent) => {
    if (!active) return;
    active = false;
    el.releasePointerCapture?.(e.pointerId);
    el.classList.remove("dragging");
    document.body.classList.remove("resizing");
  };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
}

/** Add the three panel splitters to the grid container. */
export function mountSplitters(grid: HTMLElement): void {
  const hsplitT = makeSplitter("horizontal", "hsplit-t"); // top | middle
  const vsplit = makeSplitter("vertical", "vsplit"); // center | dataset
  const hsplitB = makeSplitter("horizontal", "hsplit-b"); // middle | loss
  grid.append(hsplitT, vsplit, hsplitB);

  const padOf = (side: string) =>
    parseFloat(getComputedStyle(grid).getPropertyValue(side)) || 0;

  // Vertical: resize the left (dataset) column from the pointer's distance to
  // the grid's left edge.
  dragHandle(vsplit, (e) => {
    const rect = grid.getBoundingClientRect();
    const w = e.clientX - rect.left - padOf("padding-left");
    grid.style.setProperty("--col-left", `${clamp(w, 220, rect.width - 360)}px`);
  });

  // Horizontal (bottom): resize the loss row from the pointer's distance to the
  // grid's bottom edge.
  dragHandle(hsplitB, (e) => {
    const rect = grid.getBoundingClientRect();
    const h = rect.bottom - padOf("padding-bottom") - e.clientY;
    grid.style.setProperty(
      "--row-bottom",
      `${clamp(h, 120, rect.height * 0.6)}px`,
    );
  });

  // Horizontal (top): resize the top row from the pointer's distance to the
  // grid's top edge.
  dragHandle(hsplitT, (e) => {
    const rect = grid.getBoundingClientRect();
    const h = e.clientY - rect.top - padOf("padding-top");
    grid.style.setProperty("--row-top", `${clamp(h, 96, rect.height * 0.5)}px`);
  });
}
