# Transformer Playground

An interactive, in-browser visual demo of how a small transformer learns. You
pick a toy task, set up the architecture, and step through the computation —
watching activations (as color / Hinton diagrams) and backpropagation update the
model's weights, one layer / iteration / epoch at a time.

> **Status:** scaffolding. The page layout, dataset generation, and the scalar
> autograd engine are in place; the model forward pass, training, and the center
> network visualization are stubbed and being filled in next.

Live (project page): **https://andsmith.github.io/transformers/**
Target vanity URL: `andsmith.net/transformers` (see [Hosting](#hosting)).

## Toy tasks

All tasks operate on a tiny grammar: a vocabulary of 2–10 symbols forming short
strings. Three are sequence-to-sequence transductions; one is a classification.

| Task | Kind | Output | Description |
| --- | --- | --- | --- |
| **Copy** | transduction | `|V|`/pos | Reproduce the input sequence. |
| **Reverse** | transduction | `|V|`/pos | Output the input reversed. |
| **Sort** | transduction | `|V|`/pos | Output the input tokens sorted ascending. |
| **Parens** | classification | `1` | Are the delimiters balanced? Some symbols are matched `()` pairs, others are distractors that are ignored. |

Datasets are generated from a seeded PRNG, so **Regenerate** is reproducible. The
parens generator deliberately mixes constructed-balanced strings with random
ones to keep the two classes roughly even.

## Model architecture

Deliberately minimal, mirroring the 2017 "Attention Is All You Need" setup at
toy scale:

- **Token embedding** — trainable `|V| × d` lookup table.
- **Positional embedding** — fixed **sinusoidal** encoding (2017-style); a
  **learned** (unconstrained weights) option is also selectable.
- **Self-attention** — a single head (Q/K/V projections, scaled dot-product,
  softmax).
- **Output layer(s)** — one (optionally two) linear layers with `|V|` units
  (transduction, per position) or `1` unit (classification).

### Roadmap (now → later)

- **Now:** fixed sinusoidal PE, single attention head, 1–2 output layers, SGD.
- **Later:** learned PE everywhere, multi-head attention, deeper stacks,
  LayerNorm / residual connections, more optimizers, additional tasks.

## Visualization goals

The center panel is the focus of upcoming work:

- Activation color maps and **Hinton diagrams** for embeddings, attention
  weights, and layer outputs.
- Animated **backpropagation** showing gradients flow and weights update.
- Step granularity: one layer, one iteration (sample), one epoch, or continuous.

Because the model is built on a hand-rolled **scalar autograd** engine
(`src/engine/value.ts`), every activation and gradient is an addressable value
the visualization can read and color directly — no opaque tensor library.

## Tech stack

- TypeScript + [Vite](https://vitejs.dev/), no UI framework.
- HTML5 Canvas for the plots and network view.
- Scalar reverse-mode autodiff (micrograd-style); zero ML-library dependencies.

## Project structure

```
src/
  main.ts            App bootstrap: owns AppState, runs the rAF loop.
  state.ts           AppState + rebuild() (dataset/model/optimizer/loop).
  styles.css         CSS-grid page layout and panel styling.
  engine/            Scalar autograd: value.ts (Value graph), ops.ts (softmax,
                     cross-entropy, matVec, …).
  model/             params.ts, embeddings.ts, attention.ts, transformer.ts.
  tasks/             types.ts, grammar.ts (roles/glyphs/colors), datasets.ts.
  training/          optimizer.ts (SGD), loop.ts (step clock + loss history).
  ui/                controls.ts + top-panel, dataset-panel, loss-panel,
                     network-view.
```

## Local development

```bash
npm install
npm run dev        # Vite dev server, hot reload
npm run build      # tsc --noEmit && vite build  → dist/
npm run preview    # serve the production build locally
```

## Hosting

The app deploys to **GitHub Pages** via `.github/workflows/deploy.yml` (GitHub
Actions, on push to `main`). Vite is configured with `base: "/transformers/"`,
so every asset resolves under the subpath.

**Make it live (one-time):** in the repo, **Settings → Pages → Build and
deployment → Source = "GitHub Actions"**. After the workflow's `deploy` job goes
green, the site is at `https://andsmith.github.io/transformers/`.

### Serving it at `andsmith.net/transformers` (deferred)

A custom **apex** domain (`andsmith.net`) can attach to only one Pages site, and
subpath routing to *other* project repos only works when the apex is owned by a
**`andsmith.github.io` user page**. Today the apex is held by the `whiteboard_web`
project repo, so `andsmith.net/transformers` will not resolve yet.

To enable it later (separate task, no changes needed in *this* repo since we
already build with `base: "/transformers/"`):

1. Create a `andsmith.github.io` user-page repo and point the apex `andsmith.net`
   at it (move the `CNAME` / custom-domain setting off `whiteboard_web`).
2. Give `whiteboard_web` its own subpath base (e.g. `/whiteboard_web/`) so it
   serves at `andsmith.net/whiteboard_web/`.
3. With the apex on the user page, every project repo's Pages site —
   including this one — is automatically reachable at `andsmith.net/<repo>/`.
