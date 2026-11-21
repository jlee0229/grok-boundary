"""Train a batch of replicas full-batch and log what step 1 (the numerics audit) needs.

A "replica" is one (init seed, mask seed, lr, weight decay, eps[, ulp flip]) combination. All replicas
in one call share p, width, frac, the loss kind and the parameter dtype, and are trained together.

Logged every `log_every` steps, one value per replica:
  train/test accuracy and loss (loss always reported as float64 cross-entropy where the device has
  float64, so the loss kinds are comparable); train margins (min, 10th percentile, median, max);
  deg_frac = share of train examples with margin > 13 (float32 ce gradient off by > ~3%);
  collapse_frac / collapse64_frac = share whose float32 / float64 softmax_y is exactly 1.0;
  L2 and L-infinity norms of each weight matrix.
Logs are written to disk every `save_every` steps as well as at the end, so a crash keeps what ran.
"""
import json
import os
import platform
import time

import numpy as np
import torch

from .data import batched_split
from .losses import DEGRADED_MARGIN, collapse_fraction, margins, per_example_ce, training_loss
from .model import BatchedMLP
from .optim import BatchedAdamW


def pick_device(name: str = "auto"):
    """auto = CUDA if present, else CPU. MPS is only used when asked for explicitly: it has no
    float64 and its scatter-add backward is not deterministic."""
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_deterministic(device):
    """Ask PyTorch for bit-reproducible kernels and plain float32 matmuls (no TF32)."""
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    # MPS lacks deterministic versions of some kernels; there we only warn.
    torch.use_deterministic_algorithms(True, warn_only=(device.type == "mps"))


@torch.no_grad()
def evaluate(model, D):
    tr_logits = model(D["a_tr"], D["b_tr"])
    te_logits = model(D["a_te"], D["b_te"])
    has64 = tr_logits.device.type != "mps"
    hi = torch.float64 if has64 else torch.float32
    m = margins(tr_logits.to(hi), D["y_tr"])                       # [R, n_train]
    q = torch.quantile(m, torch.tensor([0.1, 0.5], dtype=m.dtype, device=m.device), dim=-1)
    out = {
        "train_acc": (tr_logits.argmax(-1) == D["y_tr"]).float().mean(-1),
        "test_acc": (te_logits.argmax(-1) == D["y_te"]).float().mean(-1),
        "train_loss": per_example_ce(tr_logits.to(hi), D["y_tr"]).mean(-1),
        "test_loss": per_example_ce(te_logits.to(hi), D["y_te"]).mean(-1),
        "margin_min": m.min(-1).values,
        "margin_p10": q[0],
        "margin_med": q[1],
        "margin_max": m.max(-1).values,
        "deg_frac": (m > DEGRADED_MARGIN).float().mean(-1),
        "collapse_frac": collapse_fraction(tr_logits, D["y_tr"], torch.float32),
    }
    if has64:
        out["collapse64_frac"] = collapse_fraction(tr_logits, D["y_tr"], torch.float64)
    for name, W in (("in", model.W_in), ("out", model.W_out)):
        flat = W.reshape(W.shape[0], -1)
        out[f"l2_{name}"] = flat.norm(dim=-1)
        out[f"linf_{name}"] = flat.abs().max(-1).values
    return {k: v.cpu().double().numpy() for k, v in out.items()}


def _save(out_path, steps_logged, logs, meta, model=None):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path + ".npz", step=np.array(steps_logged),
                        **{k: np.stack(v) for k, v in logs.items()})
    with open(out_path + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    if model is not None:
        torch.save({"W_in": model.W_in.detach().cpu(), "W_out": model.W_out.detach().cpu()},
                   out_path + ".final.pt")


def train_batch(cfg: dict, replicas: list, out_path: str, device="auto", log_every=100,
                save_every=10_000, verbose=True):
    """cfg: shared settings {p, width, frac, steps, loss, dtype, betas, init_scale}.
    replicas: list of dicts {init_seed, mask_seed, lr, wd, eps[, ulp_flip]}.
    Saves <out_path>.npz (logs, shape [n_logs, R]), <out_path>.json (cfg, replicas, environment) and
    <out_path>.final.pt (final weights)."""
    device = pick_device(device)
    make_deterministic(device)
    dtype = {"float32": torch.float32, "float64": torch.float64}[cfg["dtype"]]
    if device.type == "mps" and (cfg["loss"] == "ce64" or dtype == torch.float64):
        raise ValueError("MPS has no float64: run ce64 / float64 configs on CPU or CUDA")

    D = batched_split(cfg["p"], cfg["frac"], [r["mask_seed"] for r in replicas], device)
    model = BatchedMLP(cfg["p"], cfg["width"], [r["init_seed"] for r in replicas],
                       cfg.get("init_scale", 1.0), dtype,
                       ulp_flips=[r.get("ulp_flip") for r in replicas]).to(device)
    opt = BatchedAdamW(model.parameters(),
                       lr=[r["lr"] for r in replicas], weight_decay=[r["wd"] for r in replicas],
                       eps=[r["eps"] for r in replicas], betas=tuple(cfg.get("betas", (0.9, 0.98))))

    meta = {
        "cfg": cfg, "replicas": replicas, "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch": torch.__version__, "threads": torch.get_num_threads(),
        "tf32": bool(device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32),
        "seconds": None, "nonfinite_loss_at_log_step": None, "complete": False,
    }
    logs, steps_logged = {}, []
    t0 = time.time()
    for step in range(cfg["steps"] + 1):
        is_last = step == cfg["steps"]
        if step % log_every == 0 or is_last:
            row = evaluate(model, D)
            for k, v in row.items():
                logs.setdefault(k, []).append(v)
            steps_logged.append(step)
            if meta["nonfinite_loss_at_log_step"] is None and not np.isfinite(row["train_loss"]).all():
                meta["nonfinite_loss_at_log_step"] = step
            if verbose and step % (log_every * 50) == 0:
                te = row["test_acc"]
                print(f"step {step:>7}  test_acc min/med/max {te.min():.3f}/{np.median(te):.3f}/{te.max():.3f}"
                      f"  collapse med {np.median(row['collapse_frac']):.3f}  {time.time() - t0:.0f}s", flush=True)
            if step > 0 and step % save_every == 0 and not is_last:
                meta["seconds"] = round(time.time() - t0, 1)
                _save(out_path, steps_logged, logs, meta)
        if is_last:
            break
        opt.zero_grad()
        loss = training_loss(cfg["loss"], model(D["a_tr"], D["b_tr"]), D["y_tr"])   # [R]
        loss.sum().backward()
        opt.step()

    meta["seconds"], meta["complete"] = round(time.time() - t0, 1), True
    _save(out_path, steps_logged, logs, meta, model)
    return meta
