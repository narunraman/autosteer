"""Fillers draw random numbers only through autosteer.filler_tools.rng (pyrng, rng, legacy_np): not through the
stdlib's or numpy's global generators, so each question's numbers come from its own seed."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src/autosteer/filler_tools"
CONSTRUCT = {"Random", "SystemRandom", "default_rng", "Generator", "RandomState", "PCG64", "SeedSequence", "BitGenerator"}


def test_no_global_random_in_fillers():
    found = []
    for f in sorted(ROOT.rglob("*.py")):
        if f.name == "rng.py":
            continue
        t = ast.parse(f.read_text())
        stdlib = any(isinstance(n, ast.Import) and any(a.name == "random" for a in n.names) for n in ast.walk(t))
        for n in ast.walk(t):
            if isinstance(n, ast.ImportFrom) and n.module == "random":
                found.append(f"{f.name}:{n.lineno} from random import ...")
            if isinstance(n, ast.Attribute) and n.attr not in CONSTRUCT:
                v = n.value
                if stdlib and isinstance(v, ast.Name) and v.id == "random":
                    found.append(f"{f.name}:{n.lineno} random.{n.attr}")
                if isinstance(v, ast.Attribute) and v.attr == "random" and isinstance(v.value, ast.Name) and v.value.id == "np":
                    found.append(f"{f.name}:{n.lineno} np.random.{n.attr}")
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "sample" and len(n.args) == 1
                    and not any(k.arg == "random_state" for k in n.keywords)
                    and not (isinstance(n.func.value, ast.Name) and n.func.value.id in ("pyrng", "rng"))):
                found.append(f"{f.name}:{n.lineno} .sample() without random_state")
    assert not found, found
