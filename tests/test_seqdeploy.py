"""CPU deployment checks with tiny synthetic sequences; no match data or training."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

from kora import ensemble, run, seq, seqdeploy, variant


@unittest.skipIf(torch is None, "torch is optional until the network is built")
class SequenceDeploymentTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.X = pd.DataFrame(rng.normal(size=(47, 4)), columns=["a", "b", "c", "unused"])
        self.X.loc[0, "b"] = np.nan
        self.X.loc[1, "c"] = 1000
        self.half = np.r_[np.ones(23, np.int8), np.full(24, 2, np.int8)]
        s = {**seq.SETTINGS, "channels": 4, "blocks": 2, "dropout": 0.4}
        self.model = {"net": seq.network(3, s), "columns": ["c", "a", "b"],
                      "mu": np.array([0.2, -0.1, 0.3], np.float32),
                      "sd": np.array([0.8, 1.5, 2.0], np.float32), "settings": s, "seed": 2027}

    def test_save_load_round_trip_is_bit_identical_and_weights_only(self):
        expected = seqdeploy.predict_cpu(self.model, self.X, self.half)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seq_2027.pt"
            seqdeploy.save(self.model, path)
            raw = torch.load(path, weights_only=True, map_location="cpu")
            self.assertTrue(all(t.device.type == "cpu" for t in raw["state_dict"].values()))
            restored = seqdeploy.load(path)
        self.assertEqual(restored["columns"], self.model["columns"])
        self.assertEqual(restored["settings"], self.model["settings"])
        self.assertEqual(restored["seed"], 2027)
        self.assertEqual(restored["torch_version"], str(torch.__version__))
        self.assertFalse(restored["net"].training)
        np.testing.assert_array_equal(restored["mu"], self.model["mu"])
        np.testing.assert_array_equal(restored["sd"], self.model["sd"])
        actual = seqdeploy.predict_cpu(restored, self.X, self.half)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_matches_seq_predict_on_cpu(self):
        actual = seqdeploy.predict_cpu(self.model, self.X, self.half)
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(seqdeploy.CPU_THREADS)
            with mock.patch.object(seq, "device", return_value=torch.device("cpu")):
                expected = seq.predict(self.model, self.X, self.half)
        finally:
            torch.set_num_threads(threads)
        np.testing.assert_array_equal(actual, expected)

    def test_repeated_calls_are_identical_and_pin_cpu_eval_float32(self):
        observations = []

        def inspect(net, args):
            observations.append((net.training, args[0].device.type, args[0].dtype,
                                 torch.is_grad_enabled(), torch.get_num_threads(),
                                 torch.are_deterministic_algorithms_enabled()))

        self.model["net"].double().train()
        hook = self.model["net"].register_forward_pre_hook(inspect)
        try:
            with mock.patch.object(seq, "device", side_effect=AssertionError("must not select MPS")):
                first = seqdeploy.predict_cpu(self.model, self.X, self.half)
                second = seqdeploy.predict_cpu(self.model, self.X, self.half)
        finally:
            hook.remove()
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(observations, [(False, "cpu", torch.float32, False, 1, True)] * 4)

    def test_halves_are_independent(self):
        original = seqdeploy.predict_cpu(self.model, self.X, self.half)
        changed = self.X.copy()
        changed.loc[self.half == 2, self.model["columns"]] = -9.0
        actual = seqdeploy.predict_cpu(self.model, changed, self.half)
        np.testing.assert_array_equal(actual[self.half == 1], original[self.half == 1])
        self.assertFalse(np.array_equal(actual[self.half == 2], original[self.half == 2]))
        isolated = seqdeploy.predict_cpu(self.model, self.X.iloc[:23], self.half[:23])
        np.testing.assert_array_equal(isolated, original[:23])

    def test_restores_torch_settings_on_success_and_failure(self):
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.set_num_threads(2)
            torch.use_deterministic_algorithms(False, warn_only=True)
            seqdeploy.predict_cpu(self.model, self.X, self.half)
            self.assertEqual(torch.get_num_threads(), 2)
            self.assertFalse(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
            with mock.patch.object(self.model["net"], "forward", side_effect=RuntimeError("synthetic")):
                with self.assertRaisesRegex(RuntimeError, "synthetic"):
                    seqdeploy.predict_cpu(self.model, self.X, self.half)
            self.assertEqual(torch.get_num_threads(), 2)
            self.assertFalse(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
        finally:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
            torch.set_num_threads(threads)

    def test_missing_half_is_rejected(self):
        for half in (None, self.half[:-1]):
            with self.assertRaisesRegex(ValueError, "half for every frame"):
                seqdeploy.predict_cpu(self.model, self.X, half)

    def synthetic_data(self):
        y = np.arange(len(self.X)) % 3
        return {m: ([(self.X.copy(), y.copy()) for _ in range(4)], (self.X.copy(), y.copy(), self.half.copy()))
                for m in seqdeploy.MATCHES}

    def test_final_fit_uses_every_training_copy_and_its_half(self):
        data = self.synthetic_data()
        sc = variant.VARIANTS["y23_big2"]["seq"]
        with mock.patch.object(seqdeploy, "CONFIG", {"balance": 0.5}), \
                mock.patch.object(seq, "fit", return_value={"settings": sc["settings"]}) as fit:
            model = seqdeploy.fit_final(data, sc, 2028)
        sets, halves, seed, balance, settings = fit.call_args.args
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(len(sets), 28)
        for i, m in enumerate(seqdeploy.MATCHES):
            for j in range(4):
                self.assertIs(sets[4 * i + j], data[m][0][j])
                self.assertIs(halves[4 * i + j], data[m][1][2])
        self.assertEqual((seed, balance, settings), (2028, 0.5, sc["settings"]))
        self.assertEqual(model["seed"], 2028)

    def test_build_blends_cached_single_folds_and_saves_each_seed(self):
        data = self.synthetic_data()
        config = variant.VARIANTS["y23_big2"]
        sc = config["seq"]
        tree = {m: np.tile([0.6, 0.3, 0.1], (len(self.X), 1)) for m in data}
        probabilities = {2027: np.tile(np.array([0.2, 0.5, 0.3], np.float32), (len(self.X), 1)),
                         2028: np.tile(np.array([0.1, 0.2, 0.7], np.float32), (len(self.X), 1))}

        def folds(d, evs, cfg, pairs):
            self.assertIs(d, data)
            self.assertFalse(pairs)
            self.assertEqual(list(evs), [ensemble._train.FEATURE_VERSION])
            for m, X in evs[ensemble._train.FEATURE_VERSION].items():
                self.assertIs(X, data[m][1][0])
            self.assertEqual(cfg, {**sc, "seed": cfg["seed"]})
            return {ensemble._train.FEATURE_VERSION:
                    {(frozenset([m]), m): probabilities[cfg["seed"]] for m in data}}

        settings = {**seq.SETTINGS, **sc["settings"]}
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(ensemble, "CONFIG", config), \
                mock.patch.object(ensemble, "MODELS", Path(d)), \
                mock.patch.object(ensemble.nested, "seq_stage1", side_effect=folds) as cached, \
                mock.patch.object(seqdeploy, "fit_final", return_value={"settings": settings}) as fit, \
                mock.patch.object(seqdeploy, "save") as save:
            blended, metadata = ensemble.add_sequence(tree, data)
            self.assertEqual(cached.call_count, 2)
            self.assertEqual([c.args[2]["seed"] for c in cached.call_args_list], [2027, 2028])
            self.assertEqual([c.args[2] for c in fit.call_args_list], [2027, 2028])
            for call in fit.call_args_list:
                self.assertIs(call.args[0], data)
                self.assertIs(call.args[1], sc)
            self.assertEqual([c.args[1] for c in save.call_args_list],
                             [Path(d) / "seq_2027.pt", Path(d) / "seq_2028.pt"])
        for m in data:
            expected = 0.75 * tree[m] + 0.25 * np.mean(list(probabilities.values()), axis=0)
            np.testing.assert_array_equal(blended[m], expected)
        self.assertEqual(metadata, {"weight": 0.25, "seeds": [2027, 2028], "settings": settings,
                                    "files": ["seq_2027.pt", "seq_2028.pt"],
                                    "torch_version": str(torch.__version__)})

    def test_run_stage1_blends_average_network_with_average_trees(self):
        class Tree:
            def __init__(self, p): self.p = np.array(p)
            def predict_proba(self, x): return np.tile(self.p, (len(x), 1))

        bundle = {"models": [Tree([0.6, 0.3, 0.1]), Tree([0.2, 0.7, 0.1])], "columns": ["a", "b"]}
        networks = {"seq_2027.pt": [0.2, 0.3, 0.5], "seq_2028.pt": [0.6, 0.1, 0.3]}
        cfg = {"seq": {"weight": 0.25, "files": list(networks)}}

        def predict(model, X, half):
            self.assertIs(X, self.X)
            self.assertIs(half, self.half)
            return np.tile(np.asarray(model, np.float32), (len(X), 1))

        with mock.patch.object(seqdeploy, "load", side_effect=lambda p: networks[p.name]) as load, \
                mock.patch.object(seqdeploy, "predict_cpu", side_effect=predict) as net:
            actual = run.stage1(bundle, self.X, cfg, self.half)
        trees = np.mean([m.predict_proba(self.X[bundle["columns"]].to_numpy()) for m in bundle["models"]], axis=0)
        mean_net = np.mean([np.tile(np.asarray(p, np.float32), (len(self.X), 1)) for p in networks.values()], axis=0)
        np.testing.assert_array_equal(actual, 0.75 * trees + 0.25 * mean_net)
        self.assertEqual(load.call_args_list, [mock.call(run.MODELS / name) for name in networks])
        self.assertEqual(net.call_count, 2)

    def test_no_seq_path_is_byte_identical_for_single_and_multiple_trees(self):
        class Tree:
            def __init__(self, p): self.p = np.asarray(p, np.float32)
            def predict_proba(self, x): return np.tile(self.p, (len(x), 1))

        one = Tree([0.2, 0.7, 0.1])
        bundles = [{"model": one, "columns": ["b"]},
                   {"models": [one, Tree([0.6, 0.2, 0.2])], "columns": ["c", "a"]}]
        with mock.patch.object(seqdeploy, "load", side_effect=AssertionError("no network expected")), \
                mock.patch.object(seqdeploy, "predict_cpu", side_effect=AssertionError("no network expected")):
            for bundle in bundles:
                x = self.X[bundle["columns"]].to_numpy()
                legacy = np.mean([m.predict_proba(x) for m in bundle.get("models") or [bundle["model"]]], axis=0)
                for cfg in (None, {}, {"alpha": 1.0}):
                    actual = run.stage1(bundle, self.X, cfg)
                    self.assertEqual(actual.dtype, legacy.dtype)
                    self.assertEqual(actual.tobytes(), legacy.tobytes())


if __name__ == "__main__":
    unittest.main()
