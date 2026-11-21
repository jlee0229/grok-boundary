"""Train/test splits for (a + b) mod p.

The training set is a "mask": a subset of the p*p pairs (a, b). Each mask comes from its own seed,
so later experiments can cross mask seeds with init seeds.

Convention (same as ../modaddition-vs-sparse-parity/grok.py): shuffle all p*p pairs with
torch.randperm(seed) and take the first int(frac * p * p) as training data. Because the shuffle does
not depend on frac, the training set at a smaller frac is a subset of the one at a larger frac
("nested masks"), which keeps comparisons across frac paired.
"""
import torch


def split_indices(p: int, frac: float, mask_seed: int):
    """Return (train_idx, test_idx), indices into the flattened p*p grid (index = a * p + b)."""
    g = torch.Generator().manual_seed(mask_seed)
    perm = torch.randperm(p * p, generator=g)
    n_train = int(frac * p * p)
    return perm[:n_train], perm[n_train:]


def batched_split(p: int, frac: float, mask_seeds, device="cpu"):
    """Stack one split per replica.

    Returns a dict of LongTensors, each with replica as dim 0:
      a_tr, b_tr, y_tr: [R, n_train]     a_te, b_te, y_te: [R, n_test]
    Every replica has the same n_train, so the tensors stack without padding.
    """
    cols = {k: [] for k in ["a_tr", "b_tr", "y_tr", "a_te", "b_te", "y_te"]}
    for s in mask_seeds:
        tr, te = split_indices(p, frac, s)
        for name, idx in (("tr", tr), ("te", te)):
            a, b = idx // p, idx % p          # flattened index a * p + b -> (a, b)
            cols[f"a_{name}"].append(a)
            cols[f"b_{name}"].append(b)
            cols[f"y_{name}"].append((a + b) % p)
    return {k: torch.stack(v).to(device) for k, v in cols.items()}
