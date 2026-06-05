# Minibatches — design notes

Proposal: add a minibatch size (mbs) slider (powers of 2) to the Run box.
Layer-mode stepping does mbs forward passes accumulating loss, then a single
backprop pass. New step granularities: "step 1 minibatch (1-phase)" and
"step 1 minibatch (2-phase)".

Mechanically this slots in cleanly: accumulate the per-sample losses into a
mean, one `backward()`, one `optim.step()` per batch — the "presentation
cursor over a precomputed graph" design in `src/training/loop.ts` survives
intact. The ambiguities below are all about what the viewer *thinks* they're
seeing.

## 1. The "1-phase" spec has two readings

"Tick for 1 forward pass, tick for 1 backward pass" — is one forward tick:

- **(a) one sample's forward pass** → a minibatch takes `mbs + 1` ticks
  (mbs forwards, then 1 backward), or
- **(b) the whole batch's forward phase** → exactly 2 ticks per minibatch
  (forward phase, backward phase)?

Reading (a) matches the layer-mode description (mbs forward sweeps, one
backward sweep); reading (b) matches the word "phase." They differ by a
factor of mbs, and the label "1-phase" vs "2-phase" doesn't disambiguate —
does the number count phases-per-tick or something else? Naming like
"1 fwd pass / tick" vs "1 minibatch / tick" would be clearer.

Under reading (a) the tick is asymmetric: forward ticks process one sample,
the backward tick processes the whole batch. Defensible (that's genuinely how
accumulation works), but it should be deliberate.

## 2. The backward sweep mixes two gradient scopes

The deepest one. During the single R→L backward sweep:

- **Activation gradients** are per-sample (each sample has its own graph).
  Only one sample's can be displayed — presumably the last one forwarded.
  Fine, and exact: `∂L_batch/∂a_k = (1/mbs)·∂L_k/∂a_k`.
- **Weight gradients** (the `∇` view in the weight panels) are the
  *batch-aggregated* gradient — contributions from all mbs samples, including
  ones whose backward the user never "watched."

So the screen shows sample-mbs's activations and activation-grads next to
weight-grads that mostly came from other samples. A viewer will naturally
read the whole sweep as "this sample's backprop," which is wrong for the
weight panels.

Mitigations: a caption like `∇ = ∂(batch mean loss)` during the sweep, and/or
a visible batch indicator (see §5). An alternative design that dissolves the
ambiguity entirely: per-sample backward sweeps with weight-grads visibly
*accumulating* sample by sample — more ticks, but it makes gradient
accumulation itself the thing being taught. The single-sweep version is
faster but semantically blurrier.

## 3. What's an "iteration" now?

Currently iteration = sample = optimizer step, and everything hangs off that:
the `iter` counter, the loss plot x-axis, the every-10-iterations test eval,
epoch means. With minibatches, pick one:

- **iteration = optimizer step (per batch)** — the conventional choice; loss
  point = batch mean loss. But changing mbs mid-run silently rescales the
  x-axis (one unit = mbs samples), and curves taken at different mbs aren't
  comparable. Plotting against *samples seen* instead of iteration number
  fixes comparability.
- **iteration = sample** — keeps the axis stable, but then mbs consecutive
  points share one weight update, and the "loss per iteration" curve gets a
  stairstep-correlation structure that looks like a bug.

Relatedly: the epoch rollover in `finishSample` and the epoch boundary lines
(added in 0.0.10) need to know whether boundaries land on batch edges.

## 4. The partial final batch

If `trainSize % mbs ≠ 0`, the last batch of the epoch is short. Options:

- Train on the partial batch (mean over fewer samples — fine, but the mbs
  slider is then occasionally a lie). **Standard answer.**
- Let batches straddle epoch boundaries — keeps batch size constant but makes
  "epoch" fuzzy and misaligns the epoch boundary lines.
- Padding/dropping the remainder silently skips data — avoid.

Make the batch progress display honest about it ("sample 3/4" on the last
batch).

## 5. Mid-batch sample switching looks like training progress

In layer or 1-phase mode, the view switches from sample 1's pipeline to
sample 2's with *no weight update and no loss point* in between. Today, every
sample-to-sample transition implies an update, so users have learned that
association. An explicit batch progress indicator is needed — "minibatch
sample 3/8, accumulating" — next to the iter/epoch counters, otherwise the
forwards read as discarded work or silent iterations.

## 6. Loss readout semantics during accumulation

While stepping through sample k of a batch: does the loss display show
sample k's own loss, the running accumulated mean, or both? Each alone is
misleading (the per-sample loss never gets plotted; the running mean changes
for reasons not visible on screen). Showing both — "sample loss 1.32 ·
batch mean 0.97 (3/8)" — is probably the honest version.

## 7. Sum vs mean (and the LR slider)

Accumulate as **mean**, not sum — otherwise the effective learning rate
scales with the mbs slider and the two sliders become covertly coupled. With
mean loss, mbs=1 reduces exactly to today's behavior, which is also the best
regression test.

---

The pattern across all of these: minibatching splits "one sample" away from
"one update," and every UI element that currently assumes they're the same
thing (counters, loss plot, sample transitions, the ∇ view) needs to declare
which one it's talking about.
