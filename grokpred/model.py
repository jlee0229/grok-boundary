"""R independent copies of the one-hidden-layer ReLU MLP, trained in one pass.

The single-model version (grok.py's MLP with bias=False) is
    h = relu(W_in @ [onehot(a); onehot(b)]),   logits = W_out @ h.
Multiplying a weight matrix by a one-hot vector just picks out one column, so
    W_in @ [onehot(a); onehot(b)] = (column a of W_in) + (column p + b of W_in).
We use that lookup instead of a matmul: same numbers, far less work.

For many small models we store every weight with an extra leading "replica" dimension:
    W_in : [R, 2p, H]   row a reads input a, row p + b reads input b
    W_out: [R, H, p]
torch.bmm then does R independent matrix multiplies in one call. Each replica only ever touches its
own slice, so the replicas stay fully independent (tests/test_equivalence.py checks this against
grok.py's MLP trained alone).
"""
import torch
import torch.nn as nn


class BatchedMLP(nn.Module):
    def __init__(self, p: int, width: int, init_seeds, init_scale: float = 1.0, dtype=torch.float32,
                 ulp_flips=None):
        """ulp_flips: optional list (one entry per replica) of None or an int seed. With a seed, one
        randomly chosen entry of that replica's W_in is moved by one floating-point step (the
        smallest possible change). Used as a chaos control: how much do outcomes move when nothing
        but rounding-level noise differs?"""
        super().__init__()
        self.p, self.width = p, width
        w_in, w_out = [], []
        for s in init_seeds:
            # Build exactly what grok.py builds for seed s, then copy the weights in. grok.py calls
            # torch.manual_seed(seed) and then creates the two nn.Linear layers; its data split uses
            # a separate generator, so the global RNG reaches the layers in the same state here.
            torch.manual_seed(s)
            lin_in = nn.Linear(2 * p, width, bias=False)   # weight: [H, 2p]
            lin_out = nn.Linear(width, p, bias=False)      # weight: [p, H]
            w_in.append(lin_in.weight.detach().T * init_scale)    # -> [2p, H]
            w_out.append(lin_out.weight.detach().T * init_scale)  # -> [H, p]
        W_in = torch.stack(w_in).to(dtype)
        for r, flip_seed in enumerate(ulp_flips or []):
            if flip_seed is not None:
                g = torch.Generator().manual_seed(flip_seed)
                i = torch.randint(0, 2 * p, (1,), generator=g).item()
                j = torch.randint(0, width, (1,), generator=g).item()
                W_in[r, i, j] = torch.nextafter(W_in[r, i, j], torch.tensor(float("inf"), dtype=dtype))
        self.W_in = nn.Parameter(W_in)
        self.W_out = nn.Parameter(torch.stack(w_out).to(dtype))

    @property
    def n_replicas(self):
        return self.W_in.shape[0]

    def hidden(self, a, b):
        """a, b: [R, n] integer inputs -> hidden activations [R, n, H]."""
        r = torch.arange(self.n_replicas, device=a.device)[:, None]   # [R, 1], broadcasts against [R, n]
        pre = self.W_in[r, a] + self.W_in[r, b + self.p]              # [R, n, H]
        return torch.relu(pre)

    def forward(self, a, b):
        """Logits [R, n, p]."""
        return torch.bmm(self.hidden(a, b), self.W_out)
