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
