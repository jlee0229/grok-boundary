"""AdamW for stacked replicas, with a different lr / weight decay / eps for each replica.

torch.optim.AdamW takes one lr, one weight_decay and one eps for a whole parameter group. Here every
parameter has a leading replica dimension, and we want e.g. replica 0 to use weight decay 0 while
replica 1 uses 1.0. So we reimplement the update with those three as length-R tensors. The update
follows torch.optim.AdamW's single-tensor code line by line; tests/test_equivalence.py checks it
against torch.optim.AdamW run on each replica alone.

One AdamW step, for parameter theta with gradient g (t counts steps from 1):
    theta <- theta * (1 - lr * wd)                 decoupled weight decay
    m     <- beta1 * m + (1 - beta1) * g           first moment
    v     <- beta2 * v + (1 - beta2) * g^2         second moment
    theta <- theta - (lr / (1 - beta1^t)) * m / (sqrt(v) / sqrt(1 - beta2^t) + eps)
"""
import math

import torch


class BatchedAdamW:
    def __init__(self, params, lr, weight_decay, eps, betas=(0.9, 0.98)):
        self.params = list(params)
        # Compute 1 - lr * wd in float64 on the CPU (as precise as torch's Python-float version, and
        # MPS has no float64), then store each per-replica vector once in the parameter's dtype and
        # device, shaped [R, 1, 1, ...] so it broadcasts over the rest of the parameter.
        vec = lambda x: torch.as_tensor(x, dtype=torch.float64)
        lr64, wd64, eps64 = vec(lr), vec(weight_decay), vec(eps)
        self.hyper = []
        for p in self.params:
            shape = (-1,) + (1,) * (p.dim() - 1)
            cast = lambda x: x.view(shape).to(dtype=p.dtype, device=p.device)
            self.hyper.append(dict(lr=cast(lr64), decay=cast(1 - lr64 * wd64), eps=cast(eps64)))
        self.beta1, self.beta2 = betas
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - self.beta1 ** self.t
        bc2_sqrt = math.sqrt(1 - self.beta2 ** self.t)
        for p, m, v, h in zip(self.params, self.m, self.v, self.hyper):
            g = p.grad
            p.mul_(h["decay"])
            m.lerp_(g, 1 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            denom = (v.sqrt() / bc2_sqrt).add_(h["eps"])
            p.sub_((h["lr"] / bc1) * (m / denom))

    def state_dict(self):
        return {"t": self.t, "m": [x.cpu() for x in self.m], "v": [x.cpu() for x in self.v]}
