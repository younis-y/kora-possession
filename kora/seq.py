"""S3 / Y22: a small temporal convolutional network over the frame features, an extra stage-1 member
beside the gradient-boosted trees.

The trees see each frame through window averages; a short possession is an ordered pattern (a touch by
one team between touches by the other) that averages blur. Dilated, non-causal 1-D convolutions over the
feature columns can read that order (receptive field +-255 frames, about 10 s). Trained on the same
degraded training copies and balance weights as the trees; inputs standardised on the training copies,
NaN set to 0; windows never cross a half. Normalisation is per frame and the loss counts only frames whose
whole receptive field lies inside the training window, so a whole half can be predicted in one pass with
the same statistics as in training.
"""
import numpy as np

from .train import balance_weights

SETTINGS = {"channels": 64, "blocks": 8, "kernel": 3, "dropout": 0.1, "steps": 3000, "batch": 16,
            "length": 2048, "lr": 2e-3, "weight_decay": 1e-2}


def device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def radius(s):
    """Frames of context the network reads on each side."""
    return (s["kernel"] // 2) * (2 ** s["blocks"] - 1)


def network(n_in, s):
    import torch
    from torch import nn

    class FrameNorm(nn.Module):               # LayerNorm over the channels of each frame
        def __init__(self, ch):
            super().__init__()
            self.norm = nn.LayerNorm(ch)

        def forward(self, x):
            return self.norm(x.transpose(1, 2)).transpose(1, 2)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            ch, k = s["channels"], s["kernel"]
            self.inp = nn.Conv1d(n_in, ch, 1)
            self.norms = nn.ModuleList(FrameNorm(ch) for _ in range(s["blocks"]))
            self.blocks = nn.ModuleList(
                nn.Sequential(nn.Conv1d(ch, ch, k, padding=2 ** i * (k // 2), dilation=2 ** i), nn.GELU(),
                              nn.Dropout(s["dropout"]), nn.Conv1d(ch, ch, 1))
                for i in range(s["blocks"]))
            self.out = nn.Conv1d(ch, 3, 1)

        def forward(self, x):                       # x [batch, columns, frames] -> logits [batch, 3, frames]
            h = self.inp(x)
            for norm, block in zip(self.norms, self.blocks):
                h = h + block(norm(h))
            return self.out(h)
    torch.manual_seed(0)
    return TCN()


def standardise(X, mu, sd):
    Z = (np.asarray(X, np.float32) - mu) / sd
    return np.clip(np.nan_to_num(Z, nan=0.0), -10, 10).astype(np.float32)


def fit(train_sets, halves, seed, lam=0.5, settings=None):
    """train_sets: [(X DataFrame, labels)]; halves: the half array of each set (same length).
    Returns a dict model for predict()."""
    import torch
    s = {**SETTINGS, **(settings or {})}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cols = list(train_sets[0][0].columns)
    sample = np.concatenate([x.to_numpy()[::7] for x, _ in train_sets])
    mu = np.nan_to_num(np.nanmean(sample, 0)).astype(np.float32)
    sd = np.nan_to_num(np.nanstd(sample, 0)).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    seqs, starts = [], []                     # every half of every copy, with valid window starts
    for (x, lab), half in zip(train_sets, halves):
        Z, w = standardise(x.to_numpy(), mu, sd), balance_weights(lab, lam).astype(np.float32)
        for h in np.unique(half):
            i = np.flatnonzero(half == h)
            if len(i) > s["length"]:
                seqs.append((torch.from_numpy(Z[i].T.copy()), torch.from_numpy(lab[i].astype(np.int64)),
                             torch.from_numpy(w[i])))
                starts.append(len(i) - s["length"])
    dev = device()
    net = network(len(cols), s).to(dev)
    torch.manual_seed(seed)
    for p in net.parameters():                # seed-dependent initialisation
        if p.dim() > 1:
            torch.nn.init.kaiming_uniform_(p, a=5 ** 0.5)
    opt = torch.optim.AdamW(net.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=s["lr"], total_steps=s["steps"], pct_start=0.1)
    size = np.array(starts, np.float64) + 1
    r = radius(s)
    net.train()
    for _ in range(s["steps"]):
        pick = rng.choice(len(seqs), s["batch"], p=size / size.sum())
        xs, ys, ws = [], [], []
        for j in pick:
            a = int(rng.integers(starts[j] + 1))
            x, y, w = seqs[j]
            xs.append(x[:, a:a + s["length"]]); ys.append(y[a:a + s["length"]]); ws.append(w[a:a + s["length"]])
        x, y, w = torch.stack(xs).to(dev), torch.stack(ys).to(dev), torch.stack(ws).to(dev)
        inside = slice(r, s["length"] - r)
        loss = torch.nn.functional.cross_entropy(net(x)[:, :, inside], y[:, inside], reduction="none")
        loss = (loss * w[:, inside]).sum() / w[:, inside].sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    net.eval()
    return {"net": net.cpu(), "columns": cols, "mu": mu, "sd": sd, "settings": s}


def predict(model, X, half):
    """H/A/D probabilities for every frame of X, each half run as one sequence."""
    import torch
    net = model["net"].to(device())
    Z = standardise(X[model["columns"]].to_numpy(), model["mu"], model["sd"])
    out = np.zeros((len(Z), 3), np.float32)
    with torch.no_grad():
        for h in np.unique(half):
            i = np.flatnonzero(half == h)
            logits = net(torch.from_numpy(Z[i].T.copy())[None].to(device()))[0]
            out[i] = torch.softmax(logits, 0).T.cpu().numpy()
    model["net"].cpu()
    return out
