# Predicting grokking at the boundary (modular addition)

Can we predict whether a given run of a network trained on (a + b) mod p will grok, from its
initialization and its train/test split? This project follows a literature review (kept separately, not in
this repo). In short: per-hyperparameter delay laws exist (~20% error), nobody predicts single runs, and several
practical settings (float32, Adam eps, step budget) can move the boundary you measure.

## Plan (from the report, in order; each step decides whether the next is worth running)

| step | question | status |
|---|---|---|
| 1. Numerics audit | Does float32 softmax collapse change whether/when the p=97 MLP groks, depending on weight decay and Adam eps? Which loss/dtype should later steps use? | done: use `margin32`, eps 1e-8 (see "Step 1 results") |
| 2. Locate the boundary | Where is P(grok) ≈ 0.5 as a function of train fraction, and does it move with the step budget? | not started |
| 3. Crossed variance decomposition | How much outcome variance comes from the split, the init, their interaction, and numerics (thread count, device, one-bit flips)? | not started |
| 4. Step-0 predictors | Do statistics of the init and the split predict the outcome on held-out splits/inits? | not started |
| 5. f_c(p) scaling | Which of the competing critical-fraction scalings does each measurement track? | not started |
| 6. Frequency-count invariants | Does weight decay pick Θ(log p) frequencies as the corrected tensor math predicts? | not started |

## Code

- `grokpred/data.py`: train/test splits ("masks"), same convention as `grok.py`, the single-model script this project started from (not in this repo).
- `grokpred/model.py`: `BatchedMLP`, R independent copies of grok.py's MLP (one hidden ReLU layer, no biases)
  stored as stacked tensors and trained in one pass.
- `grokpred/optim.py`: `BatchedAdamW`, AdamW with a separate lr / weight decay / eps per replica.
- `grokpred/losses.py`: cross-entropy three ways (`ce32`, `ce64`, `margin32`) and the softmax-collapse diagnostics.
- `grokpred/train.py`: full-batch training loop with logging (accuracy, loss, margins, collapse share, L2 and L∞ norms).
- `audit.py`: step 1 grid. `analyze_audit.py`: step 1 tables and figures.
- `kaggle_launch.py`: packs the code into a private Kaggle notebook (T4 x2, internet off) and pushes it
  (`KAGGLE_USERNAME=<you> python kaggle_launch.py --push`).
- `tests/test_equivalence.py`: the batched code against grok.py. `python -m pytest -q tests`; the three
  comparisons with grok.py run only when `$GROK_PY` points to that script, and are skipped otherwise.

Checks done (2026-11-21):
- Same splits and same initial weights as grok.py for a given seed.
- Trained side by side in float64, batched replicas match grok.py + `torch.optim.AdamW` to ~1e-14 after
  200 steps, so the update rule is identical.
- In float32 the two differ by one rounding unit (~1e-8) after step 1, because the gradient is summed in a
  different order, and that difference grows to ~1e-3 by step 200 (lr 1e-2, p=11). Training this model is
  chaotic at the level of rounding errors. That matters for step 3: two runs that should be "identical"
  are not, unless device, thread count and kernel order are fixed.
- 4-replica validation on CPU, p=97, 8000 steps (`runs/validate/`): wd=1, eps=1e-8 groks at step 4,400
  (grok.py's runs of the same setting: ~5,000).

## Step 1: numerics audit

Baseline: p=97, width 512, 40% train, full batch, AdamW lr 1e-3, betas (0.9, 0.98), PyTorch default init.
Replicas: weight decay {0, 0.1, 1} × eps {1e-6, 1e-8, 1e-12} × seeds {0, 1, 2, 3}; the seed sets both
init and split. 100k steps. Variants (see `audit.py`):

| variant | loss | params | role |
|---|---|---|---|
| ce32 | standard cross-entropy | float32 | what grok.py does |
| ce32_ulp | same | float32 | chaos control: one weight moved by one float step |
| margin32 | softplus(−margin) | float32 | same loss, no float32 cliff |
| ce64 | cross-entropy on float64 logits | float32 | cliff moved to margin ~33–37 |
| ref64 | softplus(−margin) | float64 | reference with no cliff below margin ~700 (eps 1e-8 only, 12 replicas) |

Ran as a private Kaggle notebook, `grok-boundary-audit-v2` (T4 ×2). Results download to
`runs/kaggle_v2/`; then `python analyze_audit.py runs/kaggle_v2/runs/audit`. The final weights
(`*.final.pt`, ~95 MB) are not in the repo; the per-step logs (`*.npz`) are.

An independent review of the code (2026-11-21) found the core correct and changed the audit design:
- float64 cross-entropy also collapses (margin ~33–37), so the clean reference is float64 + margin loss (`ref64`);
- float32 collapse starts at margin ~15.4–15.8 on CPU/MPS, below the 16.64 bound, and depends on the
  device's summation order, so all variants of one comparison must run on the same device;
- rounding noise alone moves trajectories, so `ce32_ulp` measures that floor and variants are compared
  as distributions, not replica by replica;
- medians count runs that never grokked as +∞ instead of dropping them.

### Step 1 results (2026-11-21)

Kaggle T4, torch 2.11, 100k steps, all five variants completed with no non-finite losses. Full table:
`runs/kaggle_v2/runs/audit/SUMMARY.md`; figures `curves.png`, `collapse.png`, `margin.png` in the same folder.

Grokked runs out of 4, and median step of first test accuracy ≥ 0.95:

| wd | eps | ce32 | margin32 | ce64 | ref64 (float64) |
|---|---|---|---|---|---|
| 0 | all three | 0/4 | 0/4 | 0/4 | 0/4 |
| 0.1 | 1e-12 | **0/4** (test 0.27–0.41 at 100k, still rising) | 4/4, 29,250 | 4/4, 29,400 | not run |
| 0.1 | 1e-8 | 4/4, 33,950 | 4/4, 34,250 | 4/4, 34,250 | 4/4, 34,150 |
| 0.1 | 1e-6 | 4/4, 39,150 | 4/4, 38,950 | 4/4, 38,950 | not run |
| 1 | 1e-12 | 4/4, 3,550 (final test 0.991) | 4/4, 4,100 | 4/4, 4,050 | not run |
| 1 | 1e-8 | 4/4, 4,450 | 4/4, 4,450 | 4/4, 4,450 | 4/4, 4,450 |
| 1 | 1e-6 | 4/4, 5,150 | 4/4, 5,150 | 4/4, 5,200 | not run |

The chaos control `ce32_ulp` matches `ce32` in every cell (grokking steps within 0–300).

1. **Float32 collapse matters only when eps is small.** At wd 0.1, eps 1e-12, standard float32 cross-entropy did
   not grok in 100k steps (4 seeds, and all 4 one-ulp copies), while the same loss without the cliff grokked at
   ~29k on every seed. The ce32 runs' median train margin sat at the cliff (15.2–15.6) the whole time, with 1–18%
   of train examples collapsed at any logged step, and test accuracy crept upward. At wd 1, eps 1e-12, ce32
   grokked ~13% earlier but plateaued at 0.991. At eps ≥ 1e-8 with wd > 0, all variants agree to within the
   chaos-control spread.
2. **Adam's eps sets the training margin.** Under the margin loss the median train margin settles at
   m* ≈ ln(1/eps) + c(wd), with c(0.1) = −3.9 at all three eps (within 0.03) and c(1) between −6.5 and −6.8.
   At wd 0 it never settles: at 100k it is still rising, at ln(1/eps) + 1.2 to 1.9. Sketch of why: once a
   coordinate's gradient is below eps, AdamW's step is about −lr(g/eps + wd·θ). That stops moving when
   |g| ≈ eps·wd·|θ|, and |g| shrinks like e^(−m), so m* ≈ ln(1/eps) − ln(wd) + const. The sketch predicts slope 1 in
   ln(1/eps) (observed: 1.00 at wd 0.1) and −2.3 per decade of wd (observed: −2.7, from two wd values). Float32
   cross-entropy breaks exactly when m* passes the cliff at ~15.4: at wd 0.1 that means eps below ~4e-9. The
   default eps 1e-8 at wd 0.1 gives m* = 14.5, with the largest margins at 16.1–16.7, just at the edge.
3. **wd 0 does not grok under any variant**, float64 included. In this setup the wd-0 failure is not a float32
   artifact. Prieto et al.'s StableMax rescue changes the loss itself, which this audit does not test.
4. **Numerics noise is small next to the seed effect, and the seed effect repeats.** At wd 1, eps 1e-8, all five
   variants give the same grokking step for each seed: 4400, 4700, 4400, 4500. At wd 0.1, the one-ulp control
   moves the step by 0–300 against a between-seed sd of ~550. Under margin32, seed 1 is the slowest of the
   four seeds in all six conditions where runs grokked. If each condition ordered the seeds at random, that
   would happen with probability 4·(1/4)^6 ≈ 0.001. Here the seed sets both init and split, so step 3 has to separate the two.
5. **Cost on a T4:** margin32 is 7% slower than ce32; float64 (ref64) is ~6.8× slower per replica.

Decisions for later steps:
- Use `margin32` (float32 weights, margin loss). It matched the float64 reference everywhere it was run, and it
  has no cliff at any eps.
- Fix eps at 1e-8 (grok.py's value) and treat it as a real hyperparameter, not a numerics detail. At wd 0.1 it
  moves grokking from 29k (1e-12) to 34k (1e-8) to 39k (1e-6) steps.
- Find the 0.95 crossing more precisely than every 100 steps, by logging more often near the transition or by
  interpolating. At wd 1, seed differences are only 1–3 log intervals.

## Conventions

- "Grokked" = test accuracy ≥ 0.95 at some logged step. A run that has not grokked by the step budget is
  right-censored, not "never groks".
- Time is in optimizer steps (full batch, so steps = epochs).
- Every run records device, torch version, thread count, TF32 setting and the sha256 of the source files.
