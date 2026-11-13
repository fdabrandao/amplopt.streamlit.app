"""Validate every implementation of every comparison model.

Runs each tool's implementation in its own subprocess, checks that all of them reach the
model's `expected_objective` (meta.toml), re-solves the AMPL model with every solver in
`ampl_solvers`, and writes `comparisons/validation.json` (objectives + library versions)
which the Streamlit page displays. Not used by the deployed app.

Usage (from the repo root, with the pinned environment in requirements-validate.txt):
    .venv-compare/bin/python apps/syntax_comparison/validate.py
    .venv-compare/bin/python apps/syntax_comparison/validate.py --model production_rules --tool pulp

Set AMPLKEY_UUID to activate an AMPL license (the AMPL demo license limits model size).
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPARISONS = HERE / "comparisons"
OUTPUT = COMPARISONS / "validation.json"

# Executed in a fresh interpreter for each implementation. The implementation's own output
# is captured so the last stdout line is always our JSON result.
RUNNER = r"""
import contextlib, io, json, os, runpy, sys, time
path, solvers = sys.argv[1], [s for s in sys.argv[2].split(",") if s]
log = io.StringIO()
start = time.perf_counter()
with contextlib.redirect_stdout(log):
    g = runpy.run_path(path, run_name="__main__")
result = {"objective": float(g["objective"]), "seconds": time.perf_counter() - start}
if solvers and "ampl" in g:
    ampl, by_solver = g["ampl"], {}
    _, ampl_objective = next(iter(ampl.get_objectives()))  # each model has one objective
    for solver in solvers:
        ampl.option["solver"] = solver
        t = time.perf_counter()
        with contextlib.redirect_stdout(log):
            ampl.solve()
        by_solver[solver] = {
            "objective": float(ampl_objective.value()),
            "status": ampl.solve_result,
            "seconds": time.perf_counter() - t,
        }
    result["ampl_solvers"] = by_solver
print(json.dumps(result))
"""


def load_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def activate_ampl_license():
    uuid = os.environ.get("AMPLKEY_UUID")
    if not uuid:
        print("! AMPLKEY_UUID not set: using the license already installed (may be size-limited)")
        return
    from amplpy import modules

    modules.activate(uuid)


def run_impl(path, timeout, ampl_solvers=()):
    cmd = [sys.executable, "-I", "-W", "ignore", "-c", RUNNER, str(path), ",".join(ampl_solvers)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=path.parent)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout}s"}
    lines = proc.stdout.strip().splitlines()
    if proc.returncode != 0 or not lines:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        return {"error": "\n".join(tail) or f"exit code {proc.returncode}"}
    return json.loads(lines[-1])


def matches(value, expected, rel_tol):
    return abs(value - expected) <= rel_tol * max(1.0, abs(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", help="only validate this model (repeatable)")
    parser.add_argument("--tool", action="append", help="only validate this tool (repeatable)")
    parser.add_argument("--timeout", type=int, default=900, help="seconds per implementation")
    args = parser.parse_args()

    activate_ampl_license()
    tools = load_toml(COMPARISONS / "tools.toml")
    previous = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    results = previous.get("models", {})
    failures = []

    model_dirs = sorted(p.parent for p in COMPARISONS.glob("*/meta.toml"))
    for model_dir in model_dirs:
        model = model_dir.name
        if args.model and model not in args.model:
            continue
        meta = load_toml(model_dir / "meta.toml")
        expected, rel_tol = meta["expected_objective"], meta.get("rel_tol", 1e-6)
        print(f"\n== {model}: expected objective {expected}")
        model_results = results.setdefault(model, {})

        for tool, info in tools.items():
            if args.tool and tool not in args.tool:
                continue
            if tool in meta.get("not_supported", {}):
                model_results[tool] = {"status": "not_supported"}
                print(f"   {tool:9s} not supported: {meta['not_supported'][tool]}")
                continue
            script = model_dir / info["files"][-1]
            if not script.exists():
                failures.append(f"{model}/{tool}: missing {script.name}")
                print(f"   {tool:9s} MISSING {script.name}")
                continue
            solvers = meta.get("ampl_solvers", []) if tool == "ampl" else []
            res = run_impl(script, args.timeout, solvers)
            if "error" in res:
                failures.append(f"{model}/{tool}: {res['error']}")
                model_results[tool] = {"status": "error", "error": res["error"]}
                print(f"   {tool:9s} ERROR\n{res['error']}")
                continue
            ok = matches(res["objective"], expected, rel_tol)
            entry = {"status": "ok" if ok else "mismatch", "objective": res["objective"], "seconds": round(res["seconds"], 2)}
            print(f"   {tool:9s} {'ok      ' if ok else 'MISMATCH'} {res['objective']:.4f}  ({res['seconds']:.1f}s)")
            if not ok:
                failures.append(f"{model}/{tool}: objective {res['objective']} != {expected}")
            for solver, sres in res.get("ampl_solvers", {}).items():
                sok = sres["status"] == "solved" and matches(sres["objective"], expected, rel_tol)
                print(f"     ampl+{solver:8s} {'ok      ' if sok else 'MISMATCH'} {sres['objective']:.4f}  ({sres['seconds']:.1f}s, {sres['status']})")
                if not sok:
                    failures.append(f"{model}/ampl+{solver}: {sres['objective']} ({sres['status']})")
                    entry["status"] = "mismatch"
            if "ampl_solvers" in res:
                entry["solvers"] = {
                    s: {"objective": r["objective"], "status": r["status"], "seconds": round(r["seconds"], 2)}
                    for s, r in res["ampl_solvers"].items()
                }
            model_results[tool] = entry

    packages = sorted({pkg for info in tools.values() for pkg in info["packages"]})
    solver_modules = [f"ampl_module_{s}" for s in ("highs", "gurobi", "cplex", "xpress", "coin", "scip")]
    output = {
        "generated": datetime.date.today().isoformat(),
        "python": platform.python_version(),
        "versions": {pkg: package_version(pkg) for pkg in packages + solver_modules + ["pandas"]},
        "models": results,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWrote {OUTPUT.relative_to(HERE.parent.parent)}")

    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("All implementations match.")


if __name__ == "__main__":
    main()
