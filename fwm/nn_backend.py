"""Tiny MLP regressor (torch, CPU only) with input/output standardisation and masked MSE."""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ.get("FWM_THREADS", "1")))


class MLP:
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, layers: int = 2, seed: int = 0):
        torch.manual_seed(seed)
        self.in_dim, self.out_dim, self.hidden, self.layers = in_dim, out_dim, hidden, layers
        mods, d = [], in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        mods.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*mods).double()
        self.x_mu, self.x_sd = np.zeros(in_dim), np.ones(in_dim)
        self.y_mu, self.y_sd = np.zeros(out_dim), np.ones(out_dim)
        self.train_log = {}

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def fit(self, X, Y, M=None, steps=5000, batch=256, lr=1e-3, seed=0, reset_norm=True):
        """Train (or fine-tune, with reset_norm=False) on arrays X (n,in), Y (n,out), mask M (n,out)."""
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        M = np.ones_like(Y) if M is None else np.asarray(M, float)
        if len(X) == 0 or steps == 0:
            return self
        if reset_norm:
            self.x_mu, self.x_sd = X.mean(0), X.std(0) + 1e-6
            w = M.sum(0) + 1e-9
            self.y_mu = (Y * M).sum(0) / w
            self.y_sd = np.sqrt((((Y - self.y_mu) ** 2) * M).sum(0) / w) + 1e-6
        Xt = torch.tensor((X - self.x_mu) / self.x_sd)
        Yt = torch.tensor((Y - self.y_mu) / self.y_sd)
        Mt = torch.tensor(M)
        g = torch.Generator().manual_seed(seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        n, losses = len(X), []
        for it in range(steps):
            idx = torch.randint(0, n, (min(batch, n),), generator=g)
            pred = self.net(Xt[idx])
            loss = (((pred - Yt[idx]) ** 2) * Mt[idx]).sum() / Mt[idx].sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.train_log = {"steps": steps, "n": n, "final_loss": float(np.mean(losses[-50:]))}
        return self

    def predict(self, x):
        with torch.no_grad():
            xt = torch.tensor((np.asarray(x, float).reshape(1, -1) - self.x_mu) / self.x_sd)
            return self.net(xt).numpy()[0] * self.y_sd + self.y_mu

    def save(self, path):
        torch.save({"state": self.net.state_dict(), "norm": (self.x_mu, self.x_sd, self.y_mu, self.y_sd),
                    "arch": (self.in_dim, self.out_dim, self.hidden, self.layers), "log": self.train_log}, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, weights_only=False)
        m = cls(*d["arch"])
        m.net.load_state_dict(d["state"])
        m.x_mu, m.x_sd, m.y_mu, m.y_sd = d["norm"]
        m.train_log = d["log"]
        return m


def hidden_for_params(target: int, in_dim: int, out_dim: int, layers: int = 2) -> int:
    """Hidden width whose parameter count is closest to `target` (same depth)."""
    best = min(range(4, 1024), key=lambda h: abs(n_params(in_dim, out_dim, h, layers) - target))
    return best


def n_params(in_dim, out_dim, hidden, layers=2):
    n, d = 0, in_dim
    for _ in range(layers):
        n += d * hidden + hidden
        d = hidden
    return n + d * out_dim + out_dim
