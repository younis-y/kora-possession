"""Portable temporal-network weights and deterministic CPU inference for production."""
import numpy as np

from . import seq
from .external import MATCHES
from .variant import CONFIG

CPU_THREADS = 1                  # one thread avoids host-dependent parallel reduction schedules


def save(model, path):
    """Save CPU tensors and plain metadata only; the model must include its training seed."""
    import torch
    torch.save({
        "state_dict": {k: v.detach().cpu().clone() for k, v in model["net"].state_dict().items()},
        "columns": [str(c) for c in model["columns"]],
        "mu": torch.from_numpy(np.asarray(model["mu"], np.float32).copy()),
        "sd": torch.from_numpy(np.asarray(model["sd"], np.float32).copy()),
        "settings": dict(model["settings"]), "seed": int(model["seed"]),
        "torch_version": str(torch.__version__),
    }, path)


def load(path):
    """Rebuild the network on CPU without unpickling any model classes."""
    import torch
    model = torch.load(path, map_location="cpu", weights_only=True)
    net = seq.network(len(model["columns"]), model["settings"])
    net.load_state_dict(model.pop("state_dict"))
    model.update(net=net.cpu().float().eval(), mu=model["mu"].numpy(), sd=model["sd"].numpy())
    return model


def predict_cpu(model, X, half):
    """seq.predict's float32 maths, each half one sequence, pinned to one CPU thread.

    Thread count and deterministic-algorithm settings are restored even if prediction fails.
    Single-thread inference avoids varying parallel reductions between hosts; this does not
    promise bit-identical results across CPU architectures or torch versions.
    """
    import torch
    half = np.asarray(half)
    if half.shape != (len(X),):
        raise ValueError("network inference needs a half for every frame")
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(CPU_THREADS)
        torch.use_deterministic_algorithms(True)
        net = model["net"].to(device="cpu", dtype=torch.float32).eval()
        Z = seq.standardise(X[model["columns"]].to_numpy(), np.asarray(model["mu"], np.float32),
                            np.asarray(model["sd"], np.float32))
        out = np.zeros((len(Z), 3), np.float32)
        with torch.inference_mode():
            for h in np.unique(half):
                i = np.flatnonzero(half == h)
                logits = net(torch.from_numpy(Z[i].T.copy())[None])[0]
                out[i] = torch.softmax(logits, 0).T.numpy()
        return out
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)


def fit_final(data, cfg, seed):
    """Fit one seed on all seven matches' training copies, with CONFIG's balance weights.

    data is {match: (training copies, (evaluation X, labels, half))}, as in nested.seq_stage1.
    seq.fit trains on MPS when available; only deployment inference is pinned to CPU.
    """
    train = [(s, data[m][1][2]) for m in MATCHES for s in data[m][0]]
    model = seq.fit([s for s, _ in train], [h for _, h in train], seed,
                    CONFIG.get("balance") or 0.0, cfg.get("settings"))
    model["seed"] = seed
    return model
