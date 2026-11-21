"""Step 1: numerics audit of the p=97 ReLU MLP baseline.

Question: is whether/when this model groks affected by float32 softmax collapse, and does that depend
on weight decay and Adam's eps? The answer decides which loss and dtype every later experiment uses.

Baseline (same as ../modaddition-vs-sparse-parity/grok.py defaults): p=97, width 512, no biases,
40% of pairs for training, full batch, AdamW lr 1e-3, betas (0.9, 0.98), PyTorch default init.

Variants (each launch trains all of a variant's replicas together):
  variant   loss    parameters  what it tests
  ce32      ce      float32     what grok.py does
  ce32_ulp  ce      float32     same, but one W_in entry moved by one float step: the chaos control.
                                Differences between ce32 and ce32_ulp are what rounding noise alone does.
  margin32  margin  float32     same loss written so the gradient has no float32 cliff
  ce64      ce64    float32     loss computed in float64 (cliff moves to margin ~33-37)
  ref64     margin  float64     everything float64 + margin loss: no cliff below margin ~700.
                                Subset (eps 1e-8 only) because the T4 is slow at float64.
Replicas per variant: weight decay {0, 0.1, 1} x eps {1e-6, 1e-8, 1e-12} x seeds {0, 1, 2, 3}.
The seed sets both the init and the train split. Pairing across variants is only meaningful up to
the ce32 vs ce32_ulp spread, so compare variants by distributions, not replica by replica.

Usage:
  python audit.py --variant ce32 --steps 100000 [--device cuda] [--out runs/audit]
  python audit.py --variant ce32 --steps 300 --smoke     # quick check: 4 replicas
"""
import argparse
import itertools

from grokpred.train import train_batch

VARIANTS = {
    "ce32": dict(loss="ce", dtype="float32"),
    "ce32_ulp": dict(loss="ce", dtype="float32", ulp=True),
    "margin32": dict(loss="margin", dtype="float32"),
    "ce64": dict(loss="ce64", dtype="float32"),
    "ref64": dict(loss="margin", dtype="float64", subset="eps1e-8"),
}
WDS = [0.0, 0.1, 1.0]
EPSS = [1e-6, 1e-8, 1e-12]
SEEDS = [0, 1, 2, 3]


def replicas_for(variant: str, smoke: bool = False):
    v = VARIANTS[variant]
    grid = list(itertools.product(WDS, EPSS, SEEDS))
    if v.get("subset") == "eps1e-8":
        grid = [g for g in grid if g[1] == 1e-8]
    if smoke:
        grid = [(0.0, 1e-8, 0), (1.0, 1e-8, 0), (1.0, 1e-12, 1), (0.1, 1e-6, 2)]
    reps = []
    for i, (wd, eps, s) in enumerate(grid):
        r = dict(init_seed=s, mask_seed=s, lr=1e-3, wd=wd, eps=eps)
        if v.get("ulp"):
            r["ulp_flip"] = 1000 + i
        reps.append(r)
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--device", default="auto", help="auto = cuda if available, else cpu")
    ap.add_argument("--out", default="runs/audit")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--smoke", action="store_true", help="4 replicas only, for timing and debugging")
    a = ap.parse_args()

    v = VARIANTS[a.variant]
    cfg = dict(p=97, width=512, frac=0.4, steps=a.steps, betas=[0.9, 0.98], init_scale=1.0,
               variant=a.variant, loss=v["loss"], dtype=v["dtype"])
    out = f"{a.out}/{a.variant}" + ("_smoke" if a.smoke else "")
    meta = train_batch(cfg, replicas_for(a.variant, a.smoke), out, device=a.device, log_every=a.log_every)
    print(f"done: {out}  ({meta['seconds']} s on {meta['device']} {meta['device_name']})")


if __name__ == "__main__":
    main()
