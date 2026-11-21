"""Cross-entropy written three ways, plus the margin diagnostics.

All three give the same loss value in exact arithmetic. They differ only in floating-point
behaviour once a training example is fit with a large margin.

"Softmax collapse" (Prieto et al. 2025): define the margin of an example as
    m = z_y - logsumexp_{j != y} z_j            (correct logit vs. a soft max of the others)
The exact gradient of cross-entropy w.r.t. the correct logit is softmax_y - 1 = -1 / (1 + e^m).
Computed the standard way, softmax_y is a float, and 1 - softmax_y loses relative precision like
2^-24 * e^m in float32: ~4e-4 at m = 10, ~10% at m = 15. Once softmax_y rounds to exactly 1.0 the
gradient is exactly 0. That certainly happens for m > 24 ln 2 ~= 16.64, but in practice it starts
earlier (m ~ 15.4-15.8 measured on CPU and MPS), because the softmax denominator adds the tiny
terms to 1.0 a few at a time and each partial sum rounds away. The onset therefore depends on the
kernel's summation order, i.e. on the device. In float64 the same thing happens around m ~ 33-37.

  ce      : what grok.py does: log_softmax in the logits' own dtype.
  ce64    : cast float32 logits to float64 first. Moves the cliff from ~16 to ~33-37. The gradient
            is cast back to float32 at the logits, which is harmless (float32 reaches ~1e-38).
  margin  : cross-entropy = softplus(-m), computed in the logits' dtype. Its correct-class gradient
            is -sigmoid(-m) ~= -e^-m with full relative precision, so there is no cliff until exp
            underflows (m ~ 100 in float32, ~700 in float64) (Barron et al. 2026).
"""
import torch
import torch.nn.functional as F

FP32_CLIFF = 24 * 0.6931471805599453   # ~16.64: above this margin float32 softmax_y is surely 1.0
DEGRADED_MARGIN = 13.0                  # above this, the float32 ce gradient is off by > ~3%


def per_example_ce(logits, y):
    """Standard cross-entropy per example. logits [R, n, p], y [R, n] -> [R, n]."""
    return -logits.log_softmax(-1).gather(-1, y[..., None]).squeeze(-1)


def margins(logits, y):
    """m = z_y - logsumexp over the other classes. logits [R, n, p], y [R, n] -> [R, n]."""
    z_y = logits.gather(-1, y[..., None]).squeeze(-1)
    others = logits.scatter(-1, y[..., None], float("-inf"))   # knock out the correct class
    return z_y - torch.logsumexp(others, dim=-1)


def training_loss(kind: str, logits, y):
    """Mean loss per replica, shape [R]. Sum this over replicas before .backward(): replica r's
    loss only depends on replica r's weights, so each one still gets its own gradient."""
    if kind == "ce":
        return per_example_ce(logits, y).mean(-1)
    if kind == "ce64":
        return per_example_ce(logits.double(), y).mean(-1)
    if kind == "margin":
        return F.softplus(-margins(logits, y)).mean(-1)
    raise ValueError(f"unknown loss kind {kind!r}")


@torch.no_grad()
def collapse_fraction(logits, y, dtype=torch.float32):
    """Share of examples whose softmax_y, computed in `dtype`, is exactly 1.0: the examples whose
    correct-class gradient under standard cross-entropy in that dtype would be exactly zero.
    For runs trained with another loss this is hypothetical ("what ce would do with these logits").
    Shape [R]."""
    sm_y = logits.to(dtype).softmax(-1).gather(-1, y[..., None]).squeeze(-1)
    return (sm_y == 1.0).float().mean(-1)
