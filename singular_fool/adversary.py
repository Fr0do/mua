import numpy as np
import torch
from torch.autograd.functional import jvp, vjp
from tqdm.auto import tqdm


def generalized_power_method(size, matvec, rmatvec, n_iters=20, p=np.inf, q=2, truncate=0, device="cpu"):
    def psi(x, q):
        return torch.abs(x) ** (q - 1) * torch.sign(x)

    v = torch.randn(size).to(device)
    if p == np.inf:
        v = torch.clip(v, -1, 1)
    else:
        v = v / torch.norm(v, p=p)
        p_prime = 1.0 / (1.0 - 1.0 / p)
    for i in tqdm(range(n_iters)):
        Av = matvec(v)
        v = rmatvec(psi(Av, q))
        if truncate:
            _, indices = torch.topk(torch.abs(v).view(-1), size - truncate, largest=False)
            v.view(-1)[indices] = 0
        if p == np.inf:
            v = torch.clip(v, -1, 1)
        else:
            v = psi(v, p_prime)
            v = v / torch.norm(v, p=p)
    s = torch.norm(matvec(v), p=q)
    return v, s


def generate_universal_perturbation(batch, layers, n_iters=20, p=10, q=2, k=0):
    m, n = layers(batch).shape, batch[:1].shape  # Bxhidden_dims, 1xCxHxW

    def dnn_matvec(v):
        matvecs = []
        for x in batch:
            matvecs.append(jvp(layers, x.unsqueeze(0), v.view(n))[1])
        return torch.cat(matvecs)

    def dnn_rmatvec(v):
        return vjp(layers, batch, v)[1].sum(0)

    adv, s = generalized_power_method(batch[0].numel(), dnn_matvec, dnn_rmatvec, n_iters, p, q, k, device=batch.device)
    return adv / torch.norm(adv, p=p), s
