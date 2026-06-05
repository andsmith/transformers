/**
 * Tiny DOM control builders. Each returns the element plus, where useful, a
 * handle to push new values in (so panels can sync controls to AppState during
 * their `update()`).
 */

export interface RadioOption<T extends string> {
  value: T;
  label: string;
  title?: string;
  disabled?: boolean;
}

export interface RadioGroup<T extends string> {
  el: HTMLElement;
  set(value: T): void;
}

/** A horizontal set of radio-style buttons (single selection). */
export function makeRadioGroup<T extends string>(
  options: RadioOption<T>[],
  selected: T,
  onChange: (value: T) => void,
): RadioGroup<T> {
  const el = document.createElement("div");
  el.className = "radio-group";
  const buttons = new Map<T, HTMLButtonElement>();

  for (const opt of options) {
    const btn = document.createElement("button");
    btn.className = "radio-btn";
    btn.textContent = opt.label;
    if (opt.title) btn.title = opt.title;
    btn.disabled = !!opt.disabled;
    btn.classList.toggle("selected", opt.value === selected);
    btn.addEventListener("click", () => onChange(opt.value));
    buttons.set(opt.value, btn);
    el.appendChild(btn);
  }

  return {
    el,
    set(value: T) {
      for (const [v, btn] of buttons) btn.classList.toggle("selected", v === value);
    },
  };
}

export interface RadioCardOption<T extends string> {
  value: T;
  title: string;
  description: string;
}

/**
 * A single-selection group where each option is a card showing a bold title
 * and a short description (used for task selection).
 */
export function makeRadioCards<T extends string>(
  options: RadioCardOption<T>[],
  selected: T,
  onChange: (value: T) => void,
): RadioGroup<T> {
  const el = document.createElement("div");
  el.className = "radio-cards";
  const cards = new Map<T, HTMLButtonElement>();

  for (const opt of options) {
    const card = document.createElement("button");
    card.className = "radio-card";
    card.classList.toggle("selected", opt.value === selected);

    const title = document.createElement("div");
    title.className = "radio-card-title";
    title.textContent = opt.title;

    const desc = document.createElement("div");
    desc.className = "radio-card-desc";
    desc.textContent = opt.description;

    card.append(title, desc);
    card.addEventListener("click", () => onChange(opt.value));
    cards.set(opt.value, card);
    el.appendChild(card);
  }

  return {
    el,
    set(value: T) {
      for (const [v, card] of cards) card.classList.toggle("selected", v === value);
    },
  };
}

export interface SliderOpts {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  /** Format the live value readout (default: String). */
  format?: (v: number) => string;
  /** Single-row layout: label, slider, value side by side. */
  inline?: boolean;
  onInput: (value: number) => void;
}

export interface Slider {
  el: HTMLElement;
  set(value: number): void;
  /** Change the slider's maximum (e.g. dynamic clamps). */
  setMax(max: number): void;
}

/** A labelled range slider with a live value readout. */
export function makeSlider(opts: SliderOpts): Slider {
  const fmt = opts.format ?? ((v: number) => String(v));
  const el = document.createElement("label");
  el.className = opts.inline ? "slider inline" : "slider";

  const name = document.createElement("span");
  name.textContent = opts.label;
  const readout = document.createElement("span");
  readout.className = "slider-value";
  readout.textContent = fmt(opts.value);

  const input = document.createElement("input");
  input.type = "range";
  input.min = String(opts.min);
  input.max = String(opts.max);
  input.step = String(opts.step);
  input.value = String(opts.value);
  input.addEventListener("input", () => {
    const v = Number(input.value);
    readout.textContent = fmt(v);
    opts.onInput(v);
  });

  if (opts.inline) {
    // Title and value on either end, slider between.
    el.append(name, input, readout);
  } else {
    const head = document.createElement("div");
    head.className = "slider-head";
    head.append(name, readout);
    el.append(head, input);
  }
  return {
    el,
    set(value: number) {
      // Don't fight the user while they're dragging this slider.
      if (document.activeElement === input) return;
      input.value = String(value);
      readout.textContent = fmt(value);
    },
    setMax(max: number) {
      if (input.max !== String(max)) input.max = String(max);
    },
  };
}

export interface RangeSliderOpts {
  label: string;
  min: number;
  max: number;
  step: number;
  lo: number;
  hi: number;
  format?: (lo: number, hi: number) => string;
  onInput: (lo: number, hi: number) => void;
}

export interface RangeSlider {
  el: HTMLElement;
  set(lo: number, hi: number): void;
}

/**
 * A dual-thumb slider for picking a [lo, hi] range on one axis. Custom track +
 * two pointer-dragged thumbs (no native dual-range input exists).
 */
export function makeRangeSlider(opts: RangeSliderOpts): RangeSlider {
  const fmt = opts.format ?? ((lo: number, hi: number) => `${lo}–${hi}`);
  let lo = opts.lo;
  let hi = opts.hi;

  const el = document.createElement("div");
  el.className = "slider range-slider";

  const head = document.createElement("div");
  head.className = "slider-head";
  const name = document.createElement("span");
  name.textContent = opts.label;
  const readout = document.createElement("span");
  readout.className = "slider-value";
  head.append(name, readout);

  const track = document.createElement("div");
  track.className = "range-track";
  const fill = document.createElement("div");
  fill.className = "range-fill";
  const thumbLo = document.createElement("div");
  thumbLo.className = "range-thumb";
  const thumbHi = document.createElement("div");
  thumbHi.className = "range-thumb";
  track.append(fill, thumbLo, thumbHi);

  el.append(head, track);

  const frac = (v: number) => (v - opts.min) / Math.max(1e-9, opts.max - opts.min);

  function render(): void {
    readout.textContent = fmt(lo, hi);
    fill.style.left = `${frac(lo) * 100}%`;
    fill.style.width = `${(frac(hi) - frac(lo)) * 100}%`;
    thumbLo.style.left = `${frac(lo) * 100}%`;
    thumbHi.style.left = `${frac(hi) * 100}%`;
  }

  function valueAt(clientX: number): number {
    const r = track.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (clientX - r.left) / Math.max(1, r.width)));
    const raw = opts.min + t * (opts.max - opts.min);
    return Math.round(raw / opts.step) * opts.step;
  }

  let active: "lo" | "hi" | null = null;

  const attach = (thumb: HTMLElement, which: "lo" | "hi") => {
    thumb.addEventListener("pointerdown", (e) => {
      active = which;
      thumb.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    thumb.addEventListener("pointermove", (e) => {
      if (active !== which) return;
      const v = valueAt(e.clientX);
      if (which === "lo") lo = Math.max(opts.min, Math.min(v, hi));
      else hi = Math.min(opts.max, Math.max(v, lo));
      render();
      opts.onInput(lo, hi);
    });
    const end = () => {
      active = null;
    };
    thumb.addEventListener("pointerup", end);
    thumb.addEventListener("pointercancel", end);
  };
  attach(thumbLo, "lo");
  attach(thumbHi, "hi");

  // Clicking the track moves the nearest thumb.
  track.addEventListener("pointerdown", (e) => {
    if (e.target !== track && e.target !== fill) return;
    const v = valueAt(e.clientX);
    if (Math.abs(v - lo) <= Math.abs(v - hi)) lo = Math.max(opts.min, Math.min(v, hi));
    else hi = Math.min(opts.max, Math.max(v, lo));
    render();
    opts.onInput(lo, hi);
  });

  render();
  return {
    el,
    set(newLo: number, newHi: number) {
      if (active !== null) return; // don't fight an in-progress drag
      lo = newLo;
      hi = newHi;
      render();
    },
  };
}

export interface NumberInput {
  el: HTMLInputElement;
  set(value: number): void;
}

/** A compact integer input. */
export function makeNumberInput(
  value: number,
  onChange: (value: number) => void,
  title = "",
): NumberInput {
  const el = document.createElement("input");
  el.type = "number";
  el.className = "num-input";
  el.step = "1";
  el.value = String(value);
  if (title) el.title = title;
  el.addEventListener("change", () => {
    const v = Math.floor(Number(el.value));
    if (Number.isFinite(v)) onChange(v >>> 0);
  });
  return {
    el,
    set(v: number) {
      if (document.activeElement === el) return;
      el.value = String(v);
    },
  };
}

export interface DropdownOption<T extends string> {
  value: T;
  label: string;
  disabled?: boolean;
}

export interface Dropdown<T extends string> {
  el: HTMLSelectElement;
  set(value: T): void;
}

export function makeDropdown<T extends string>(
  options: DropdownOption<T>[],
  selected: T,
  onChange: (value: T) => void,
): Dropdown<T> {
  const el = document.createElement("select");
  el.className = "dropdown";
  for (const opt of options) {
    const o = document.createElement("option");
    o.value = opt.value;
    o.textContent = opt.label;
    o.disabled = !!opt.disabled;
    el.appendChild(o);
  }
  el.value = selected;
  el.addEventListener("change", () => onChange(el.value as T));
  return {
    el,
    set(value: T) {
      if (document.activeElement === el) return;
      el.value = value;
    },
  };
}

export interface Checkbox {
  el: HTMLElement;
  set(checked: boolean): void;
}

/** A labelled checkbox. */
export function makeCheckbox(
  label: string,
  checked: boolean,
  onChange: (checked: boolean) => void,
): Checkbox {
  const el = document.createElement("label");
  el.className = "checkbox";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = checked;
  input.addEventListener("change", () => onChange(input.checked));
  const span = document.createElement("span");
  span.textContent = label;
  el.append(input, span);
  return {
    el,
    set(value: boolean) {
      if (document.activeElement === input) return;
      input.checked = value;
    },
  };
}

export function makeButton(label: string, onClick: () => void): HTMLButtonElement {
  const btn = document.createElement("button");
  btn.className = "btn";
  btn.textContent = label;
  btn.addEventListener("click", onClick);
  return btn;
}

/** A titled grouping box for a cluster of related controls. */
export function makeFieldset(title: string): HTMLElement {
  const el = document.createElement("div");
  el.className = "fieldset";
  const legend = document.createElement("div");
  legend.className = "fieldset-title";
  legend.textContent = title;
  el.appendChild(legend);
  return el;
}
