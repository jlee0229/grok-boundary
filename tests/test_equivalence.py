"""Checks that the batched code computes exactly what the old single-model code computes.

Run with:  .venv/bin/python -m pytest -q tests
"""
import importlib.util
import os
import pathlib

import pytest
import torch
import torch.nn.functional as F

from grokpred.data import batched_split
from grokpred.losses import margins, per_example_ce, training_loss
from grokpred.model import BatchedMLP
from grokpred.optim import BatchedAdamW

# CPU index-gather backward is only reproducible with deterministic algorithms on.
torch.use_deterministic_algorithms(True)

# grok.py is the single-model script this project started from. It lives outside this repo; point
# $GROK_PY at it to run the comparisons below, otherwise they are skipped.
GROK_PY = pathlib.Path(os.environ.get(
    "GROK_PY", pathlib.Path(__file__).resolve().parents[2] / "modaddition-vs-sparse-parity" / "grok.py"))
needs_grok_py = pytest.mark.skipif(not GROK_PY.exists(), reason=f"reference script not found: {GROK_PY}")


def load_grok_py():
    spec = importlib.util.spec_from_file_location("grok_old", GROK_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def old_model_and_data(grok, p, width, frac, seed):
    """Exactly the setup lines of grok.py's main() for --task modadd --arch mlp."""
    torch.manual_seed(seed)
    D = grok.make_modadd(p, frac, 0.0, seed)
    model = grok.MLP(2 * p, width, 1, p, bias=False, init_scale=1.0)
    return model, D


@needs_grok_py
def test_split_and_init_match_grok_py():
    grok = load_grok_py()
    p, width, frac, seeds = 13, 32, 0.4, [0, 3]
    D = batched_split(p, frac, seeds)
    model = BatchedMLP(p, width, seeds)
    all_logits = model(D["a_tr"], D["b_tr"])          # [R, n, p]
    for r, s in enumerate(seeds):
        old, Dold = old_model_and_data(grok, p, width, frac, s)
        # same training pairs in the same order: decode grok.py's one-hot rows back to (a, b)
        a_old = Dold["Xtr"][:, :p].argmax(1)
        b_old = Dold["Xtr"][:, p:].argmax(1)
        assert torch.equal(a_old, D["a_tr"][r]) and torch.equal(b_old, D["b_tr"][r])
        assert torch.equal(Dold["ytr"][:, 0], D["y_tr"][r])
        # same logits
        assert torch.allclose(all_logits[r], old(Dold["Xtr"]), atol=1e-6, rtol=0)


def test_margin_loss_equals_cross_entropy():
    torch.manual_seed(0)
    logits = (torch.randn(3, 50, 11, dtype=torch.float64) * 5).requires_grad_()
    y = torch.randint(0, 11, (3, 50))
    l_ref = F.cross_entropy(logits.reshape(-1, 11), y.reshape(-1), reduction="none").reshape(3, 50).mean(-1)
    g_ref, = torch.autograd.grad(l_ref.sum(), logits)
    for kind in ["ce", "ce64", "margin"]:
        l = training_loss(kind, logits, y)
        g, = torch.autograd.grad(l.sum(), logits)
        assert torch.allclose(l, l_ref, atol=1e-12) and torch.allclose(g, g_ref, atol=1e-12), kind


def test_margin_loss_keeps_gradient_past_fp32_cliff():
    # One example with margin 30: float32 cross-entropy gives an exactly-zero correct-class gradient,
    # the softplus-of-margin form does not.
    logits = torch.tensor([[[30.0, 0.0, 0.0]]], requires_grad=True)
    y = torch.tensor([[0]])
    g_ce, = torch.autograd.grad(training_loss("ce", logits, y).sum(), logits)
    g_mg, = torch.autograd.grad(training_loss("margin", logits, y).sum(), logits)
    assert g_ce[0, 0, 0] == 0.0
    assert g_mg[0, 0, 0] < 0 and abs(g_mg[0, 0, 0].item() + 2 * torch.exp(torch.tensor(-30.0)).item()) < 1e-16


def _train_both(dtype, steps, atol):
    """Train 3 replicas with different (seed, wd, eps) together, and each one alone with grok.py's MLP
    and torch.optim.AdamW, then compare the weights.

    Why two tolerances: the batched model sums gradients in a different order than grok.py's one-hot
    matmul, so float32 runs differ by ~1 rounding unit (1e-8) after step 1. Training amplifies that
    difference exponentially (we measured 1e-8 -> 1e-3 over 200 steps at lr 1e-2), so float32 can
    only be compared over a few steps. In float64 the same 200 steps agree to ~1e-14, which is the
    real check that the update rule is identical."""
    grok = load_grok_py()
    p, width, frac, lr = 11, 24, 0.5, 1e-2
    reps = [dict(seed=0, wd=0.0, eps=1e-8), dict(seed=1, wd=1.0, eps=1e-8), dict(seed=2, wd=0.3, eps=1e-6)]

    seeds = [r["seed"] for r in reps]
    D = batched_split(p, frac, seeds)
    model = BatchedMLP(p, width, seeds, dtype=dtype)
    opt = BatchedAdamW(model.parameters(), lr=[lr] * 3, weight_decay=[r["wd"] for r in reps],
                       eps=[r["eps"] for r in reps], betas=(0.9, 0.98))
    for _ in range(steps):
        opt.zero_grad()
        training_loss("ce", model(D["a_tr"], D["b_tr"]), D["y_tr"]).sum().backward()
        opt.step()

    for r, rep in enumerate(reps):
        old, Dold = old_model_and_data(grok, p, width, frac, rep["seed"])
        old = old.to(dtype)
        old_opt = torch.optim.AdamW(old.parameters(), lr=lr, weight_decay=rep["wd"], eps=rep["eps"],
                                    betas=(0.9, 0.98), foreach=False)
        for _ in range(steps):
            old_opt.zero_grad()
            grok.per_example_loss(old(Dold["Xtr"].to(dtype)), Dold["ytr"], 1, p).mean().backward()
            old_opt.step()
        assert torch.allclose(model.W_in[r], old.inp.weight.T, atol=atol, rtol=0), f"W_in replica {r}"
        assert torch.allclose(model.W_out[r], old.out.weight.T, atol=atol, rtol=0), f"W_out replica {r}"


@needs_grok_py
def test_batched_training_matches_grok_py_float64():
    _train_both(torch.float64, steps=200, atol=1e-11)


@needs_grok_py
def test_batched_training_matches_grok_py_float32_short():
    _train_both(torch.float32, steps=20, atol=1e-6)


def test_margins_definition():
    logits = torch.tensor([[[2.0, 1.0, 0.0]]], dtype=torch.float64)
    y = torch.tensor([[0]])
    expected = 2.0 - torch.logsumexp(torch.tensor([1.0, 0.0], dtype=torch.float64), 0)
    assert torch.allclose(margins(logits, y)[0, 0], expected)
    assert torch.allclose(per_example_ce(logits, y)[0, 0], F.softplus(-expected))


def test_ulp_flip_changes_exactly_one_entry_by_one_step():
    a = BatchedMLP(11, 16, [0, 0])
    b = BatchedMLP(11, 16, [0, 0], ulp_flips=[None, 7])
    assert torch.equal(a.W_in[0], b.W_in[0]) and torch.equal(a.W_out, b.W_out)
    diff = (a.W_in[1] != b.W_in[1])
    assert diff.sum() == 1
    old, new = a.W_in[1][diff], b.W_in[1][diff]
    assert new > old and torch.equal(torch.nextafter(old, torch.tensor(float("inf"))), new)
