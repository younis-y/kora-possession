"""Y22 in the submission: fit, save and load the temporal network of kora/seq.py.

kora/seq.py is left unchanged so its cache keys (its bytes) still match kora.nested's folds. The
network is fitted on the GPU when there is one (two fits with the same seed gave identical predictions
on the M4, 7 Nov) and run on the CPU, so kora.run reproduces the file from the committed models.
"""
from contextlib import contextmanager

from . import seq


@contextmanager
def on_cpu():
    import torch
    saved, seq.device = seq.device, lambda: torch.device("cpu")
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        seq.device = saved
        torch.use_deterministic_algorithms(False)


def predict_cpu(model, X, half):
    with on_cpu():
        return seq.predict(model, X, half)


def save(model, path):
    import torch
    torch.save({"state": model["net"].state_dict(), "columns": [str(c) for c in model["columns"]],
                "mu": torch.from_numpy(model["mu"]), "sd": torch.from_numpy(model["sd"]),
                "settings": model["settings"]}, path)


def load(path):
    import torch
    d = torch.load(path, weights_only=True)
    net = seq.network(len(d["columns"]), d["settings"])
    net.load_state_dict(d["state"])
    net.eval()
    return {"net": net, "columns": d["columns"], "mu": d["mu"].numpy(), "sd": d["sd"].numpy(),
            "settings": d["settings"]}
