# AMPL syntax comparison

The same optimization models written in AMPL and in other Python modeling tools, shown side by
side on the Streamlit page. Only AMPL runs live in the app; the other implementations are static
files that `validate.py` checks offline.

## Layout

```
comparisons/
  tools.toml                  # registry of tools (name, group, files, packages)
  validation.json             # written by validate.py: objectives + library versions
  <model>/
    meta.toml                 # title, summary, expected objective, AMPL solvers, not_supported
    model.mod, ampl_model.py  # AMPL model + data loading/solve with amplpy
    <tool>_model.py           # one file per tool (pyomo, pulp, gurobipy, docplex, xpress, highspy)
    data/*.csv                # shared input data, read by every implementation
validate.py
requirements-validate.txt     # pinned validation environment (not the app's requirements)
```

Each file is split into blocks with `# @block <name>` comments (`setup`, `data`, `sets`, `params`,
`vars`, `objective`, `constraints`, `solve`). The page aligns blocks by name, so their order
inside a file does not matter. Markers are stripped before display.

## Adding a tool

1. Add an entry to `comparisons/tools.toml`.
2. Add `<tool>_model.py` to each model folder, or list the tool under `[not_supported]` in that
   model's `meta.toml` with a one-line reason.
3. Add the pinned package to `requirements-validate.txt` and run `validate.py`.

## Validating

```bash
python -m venv .venv-compare
.venv-compare/bin/pip install -r apps/syntax_comparison/requirements-validate.txt
export AMPLKEY_UUID=<license-uuid>   # the AMPL demo license limits model size
.venv-compare/bin/python apps/syntax_comparison/validate.py
```

Options: `--model <name>` and `--tool <name>` (repeatable), `--timeout <seconds>`.
The script exits with status 1 if any implementation misses the expected objective.
The free pip licenses of gurobipy, cplex and xpress limit model size (CPLEX Community Edition:
1000 variables and 1000 constraints), so comparison instances stay within those limits.
