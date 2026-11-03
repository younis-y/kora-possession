"""Organiser rules (3 Nov 2026): the challenge match may be read only by inference and by checks."""
import ast
import unittest
from pathlib import Path

KORA = Path(__file__).resolve().parent.parent / "kora"
ALLOWED = {"run", "checks", "shift", "profile", "data", "target"}     # inference, checks, the loaders
CHALLENGE_NAMES = {"load_target", "tracking", "homography", "smart_tagging", "teams"}


def challenge_imports(path):
    tree = ast.parse(path.read_text())
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("target", "data") and node.level:
            hits += [a.name for a in node.names if a.name in CHALLENGE_NAMES]
        if isinstance(node, ast.ImportFrom) and node.module is None and node.level:
            hits += [a.name for a in node.names if a.name in ("target",)]
    return hits


class ComplianceTests(unittest.TestCase):
    def test_challenge_data_is_read_only_by_inference_and_checks(self):
        offenders = {p.stem: challenge_imports(p) for p in KORA.glob("*.py") if p.stem not in ALLOWED}
        self.assertEqual({k: v for k, v in offenders.items() if v}, {})

    def test_degrade_has_no_numeric_constants(self):
        tree = ast.parse((KORA / "degrade.py").read_text())
        numbers = [n.value for n in ast.walk(tree)
                   if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                   and not isinstance(n.value, bool) and n.value not in (0, 1, 2)]
        self.assertEqual(numbers, [], "degrade.py must take every number from degrade_external.json")


if __name__ == "__main__":
    unittest.main()
