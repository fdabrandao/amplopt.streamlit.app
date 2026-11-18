"""Validate every implementation of every comparison model.

Runs each tool's implementation in its own subprocess, checks that all of them reach the
model's `expected_objective` (meta.toml), solves the AMPL model with every solver in
`ampl_solvers` (one subprocess per solver, with a solver time limit; solves that hit it are
recorded as unverified with their optimality gap), and writes `comparisons/validation.json` (objectives + library versions)
which the Streamlit page displays. Not used by the deployed app.

Usage (from the repo root, with the pinned environment in requirements-validate.txt):
    .venv-compare/bin/python apps/syntax_comparison/validate.py
    .venv-compare/bin/python apps/syntax_comparison/validate.py --model production_rules --tool pulp

Set AMPLKEY_UUID to activate an AMPL license (the AMPL demo license limits model size).
"""

import argparse
import datetime
import importlib.util
import json
import os
import platform
import signal
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPARISONS = HERE / "comparisons"
OUTPUT = COMPARISONS / "validation.json"

# Validation-only solver time limit. The code shown on the page has no limit: for each tool,
# validate.py runs the file up to its solve block, applies the limit to the tool's own model object,
# runs the solve block, then reads whether optimality was proven and, if not, the best bound.
# tool -> (code run before the solve block, "proven optimal" expression, "best bound" expression)
PYOMO_HOOK = """
_factory = pyo.SolverFactory
class _TimeLimited:
    def __init__(self, solver):
        self.solver = solver
    def solve(self, model, **kwargs):
        global RESULTS
        if hasattr(self.solver, "config"):  # pyomo.contrib.solver interfaces (highs)
            kwargs.update(timelimit=LIMIT, raise_exception_on_nonoptimal_result=False)
        RESULTS = self.solver.solve(model, **kwargs)
        return RESULTS
pyo.SolverFactory = lambda *args, **kwargs: _TimeLimited(_factory(*args, **kwargs))
"""
HOOKS = {
    "gurobipy": ("m.Params.TimeLimit = LIMIT", "m.Status == GRB.OPTIMAL", "m.ObjBound"),
    "docplex": ("m.parameters.timelimit = LIMIT", "'optimal' in m.solve_details.status", "m.solve_details.best_bound"),
    "xpress": (
        "prob.controls.timelimit = LIMIT",
        "prob.attributes.solstatus == xp.SolStatus.OPTIMAL",
        "prob.attributes.bestbound",
    ),
    "highspy": (
        "h.setOptionValue('time_limit', LIMIT)",
        "h.getModelStatus() == highspy.HighsModelStatus.kOptimal",
        "h.getInfo().mip_dual_bound",
    ),
    "pulp": (
        "import functools; pulp.HiGHS = functools.partial(pulp.HiGHS, timeLimit=LIMIT)",
        "prob.solverModel.getModelStatus() == prob.solverModel.getModelStatus().kOptimal",
        "prob.solverModel.getInfo().mip_dual_bound",
    ),
    "pyomo": (PYOMO_HOOK, "str(RESULTS.solver.termination_condition) == 'optimal'", "RESULTS.problem.lower_bound"),
}
# AMPL solvers that take `timelim` (all MP drivers); all but CBC also return the `bestbound` suffix.
AMPL_TIME_LIMIT = {"highs", "cbc", "scip", "gurobi", "cplex", "xpress"}

# Executed in a fresh interpreter for each implementation (and for each AMPL solver). The
# implementation's own output is captured so the last stdout line is always our JSON result.
RUNNER = r"""
import contextlib, io, json, re, sys, time
path, tool, solver, limit = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
hooks, ampl_time_limit = json.loads(sys.argv[5]), sys.argv[6].split(",")
before, solve = open(path).read().split("# @block solve")
g = {"__file__": path, "__name__": "__main__", "LIMIT": limit}
bound = None
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(before, path, "exec"), g)
    start = time.perf_counter()
    if tool == "ampl":
        ampl = g["ampl"]
        name, objective = next(iter(ampl.get_objectives()))  # each model has one objective
        ampl.option["solver"] = solver
        if solver in ampl_time_limit:
            ampl.option[solver + "_options"] = f"timelim={limit:g}" + ("" if solver == "cbc" else " bestbound=1")
        output = ampl.get_output("solve;")
        value, proven = objective.value(), ampl.solve_result == "solved"
        if not proven:
            try:
                bound = ampl.get_value(name + ".bestbound")
            except Exception:  # CBC reports the gap in its solve message instead
                found = re.search(r"absmipgap=([-+0-9.eE]+)", output)
                bound = value - float(found.group(1)) if found else None
    else:
        setup, proven_expr, bound_expr = hooks[tool]
        exec(setup, g)
        exec(compile(solve, path, "exec"), g)
        value, proven = g["objective"], bool(eval(proven_expr, g))
        if not proven:
            try:
                bound = float(eval(bound_expr, g))
            except Exception:
                bound = None
seconds = time.perf_counter() - start
print(json.dumps({"objective": float(value), "proven": proven, "bound": bound, "seconds": seconds}))
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


def subprocess_env():
    """Put the open-source Ipopt binary from ampl_module_coin on PATH (used by Pyomo's ipopt plugin)."""
    env = dict(os.environ)
    spec = importlib.util.find_spec("ampl_module_coin")
    if spec and spec.submodule_search_locations:
        coin_bin = os.path.join(spec.submodule_search_locations[0], "bin")
        env["PATH"] = coin_bin + os.pathsep + env.get("PATH", "")
    return env


def run(path, tool, solver, time_limit, timeout):
    cmd = [
        sys.executable, "-I", "-W", "ignore", "-c", RUNNER,
        str(path), tool, solver, str(time_limit), json.dumps(HOOKS), ",".join(sorted(AMPL_TIME_LIMIT)),
    ]
    # Own process group, so a timeout also stops the solver processes it started
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=path.parent, env=subprocess_env(), start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return {"error": f"timeout after {timeout}s"}
    lines = stdout.strip().splitlines()
    if proc.returncode != 0 or not lines:
        tail = (stderr or stdout).strip().splitlines()[-5:]
        return {"error": "\n".join(tail) or f"exit code {proc.returncode}"}
    return json.loads(lines[-1])


def matches(value, expected, rel_tol):
    return abs(value - expected) <= rel_tol * max(1.0, abs(expected))


def assess(res, expected, rel_tol):
    """Status of one solve: solved (proven optimal, expected objective), unverified (time limit
    reached before optimality was proven), mismatch or error; plus the details to record."""
    if "error" in res:
        return "error", {"error": res["error"]}
    entry = {"objective": res["objective"], "seconds": round(res["seconds"], 2)}
    if res["proven"]:
        return ("solved" if matches(res["objective"], expected, rel_tol) else "mismatch"), entry
    if res["objective"] < expected - rel_tol * max(1.0, abs(expected)):
        return "mismatch", entry  # better than the "optimal" objective: the expected value is wrong
    if res["bound"] is not None:
        entry["bound"] = res["bound"]
        entry["gap"] = abs(res["objective"] - res["bound"]) / max(1.0, abs(res["objective"]))
    return "unverified", entry


def describe(status, entry):
    if status == "error":
        return "ERROR " + entry["error"].splitlines()[-1]
    text = f"{entry['objective']:.4f}  ({entry['seconds']:.1f}s)"
    if status == "unverified":
        gap = f"gap {entry['gap']:.2%}" if "gap" in entry else "gap unknown"
        return f"UNVERIFIED: time limit, {gap}, best {text}"
    return ("ok       " if status == "solved" else "MISMATCH ") + text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", help="only validate this model (repeatable)")
    parser.add_argument("--tool", action="append", help="only validate this tool (repeatable)")
    parser.add_argument("--time-limit", type=int, default=600, help="solver time limit in seconds (default 600)")
    parser.add_argument("--timeout", type=int, help="hard limit per run, incl. model building (default: time limit + 600)")
    args = parser.parse_args()
    timeout = args.timeout or args.time_limit + 600

    activate_ampl_license()
    tools = load_toml(COMPARISONS / "tools.toml")
    previous = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    results = previous.get("models", {})
    failures, unverified = [], []

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
            limited = meta.get("license_limited", {}).get(tool)
            if tool == "ampl":
                by_solver = {}
                for solver in meta.get("ampl_solvers", ["highs"]):
                    status, entry = assess(run(script, tool, solver, args.time_limit, timeout), expected, rel_tol)
                    by_solver[solver] = {"status": status, **entry}
                    print(f"   ampl+{solver:8s} {describe(status, entry)}")
                    if status in ("mismatch", "error"):
                        failures.append(f"{model}/ampl+{solver}: {describe(status, entry)}")
                    elif status == "unverified":
                        unverified.append(f"{model}/ampl+{solver}")
                first = next((r for r in by_solver.values() if r["status"] == "solved"), None)
                if any(r["status"] in ("mismatch", "error") for r in by_solver.values()):
                    status = "mismatch"
                else:
                    status = "ok" if first else "unverified"
                model_results[tool] = {"status": status, **({"objective": first["objective"]} if first else {}), "solvers": by_solver}
                continue

            status, entry = assess(run(script, tool, "", args.time_limit, timeout), expected, rel_tol)
            if limited and status != "solved":
                model_results[tool] = {"status": "license_limited", "reason": limited}
                print(f"   {tool:9s} license-limited (not counted as a failure): {describe(status, entry)}")
                continue
            print(f"   {tool:9s} {describe(status, entry)}")
            if status in ("mismatch", "error"):
                failures.append(f"{model}/{tool}: {describe(status, entry)}")
            elif status == "unverified":
                unverified.append(f"{model}/{tool}")
            model_results[tool] = {"status": "ok" if status == "solved" else status, **entry}

    existing = {d.name for d in model_dirs}
    results = {name: res for name, res in results.items() if name in existing}  # drop deleted models

    packages = sorted({pkg for info in tools.values() for pkg in info["packages"]})
    solver_modules = [f"ampl_module_{s}" for s in ("highs", "gurobi", "cplex", "xpress", "coin", "scip", "knitro")]
    output = {
        "generated": datetime.date.today().isoformat(),
        "python": platform.python_version(),
        "time_limit": args.time_limit,
        "versions": {pkg: package_version(pkg) for pkg in packages + solver_modules + ["pandas"]},
        "models": dict(sorted(results.items())),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWrote {OUTPUT.relative_to(HERE.parent.parent)}")

    if unverified:
        print(f"\nUNVERIFIED (optimality not proven within {args.time_limit}s, not counted as failures):\n  " + "\n  ".join(unverified))
    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("All implementations match.")


if __name__ == "__main__":
    main()
