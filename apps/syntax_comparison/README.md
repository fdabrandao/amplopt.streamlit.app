# AMPL syntax comparison

The same optimization models written in AMPL and in other Python modeling tools, shown side by
side on the Streamlit page. The page only shows code: nothing is solved in the app.
`validate.py` runs every implementation offline and checks that they reach the same optimal objective.

## Layout

```
comparisons/
  tools.toml                  # registry of tools (name, group, files, packages)
  validation.json             # written by validate.py: objectives, gaps, library versions (not shown)
  <model>/
    meta.toml                 # title, summary, headline facts, notes, expected objective, AMPL solvers
    model.mod, ampl_model.py  # AMPL model + data loading/solve with amplpy
    <tool>_model.py           # one file per tool (pyomo, pulp, gurobipy, docplex, xpress, highspy)
    data/*.csv                # shared input data, read by every implementation
app.py                        # Streamlit page (selectors, headline facts, code grid with notes)
catalog.py                    # loads tools/models/blocks/validation.json (no Streamlit)
render.py                     # HTML for the color-banded, aligned code grid
validate.py
requirements-validate.txt     # pinned validation environment (not the app's requirements)
```

The page reads everything from `comparisons/`; nothing is hard-coded per model. URL parameters
`?model=<key>&tool=<tool>&change=<variant>` preselect a comparison, so a demo can be bookmarked.

A model folder whose `meta.toml` sets `variant_of = "<model>"` and `change = "<label>"` is not listed
on its own: it adds one change to that model (e.g. `production_rules_volume_discount` adds volume
discounts). The page lists a model's changes under "Add a new rule" (by `change_name`) and
highlights, per tool, the lines that differ from the base model. Variants are validated like any
other model.

Notes (`[[notes]]` in `meta.toml`) appear as sticky notes just above the code they describe, so
the page works both as a guided demo and on its own; "📝 Show notes" (URL `notes=0`) hides them.
Each note has `at` ("header" or a block name) and `text`; a note without `tools` sits in the AMPL
column, and one with `tools = [...]` sits in the other column when one of those tools is selected.
A rule change (variant) adds its own notes to the base model's.

"Highlight a rule" lists a model's `[[rules]]` (key, label, text). In every implementation, the
lines of a rule are wrapped in `# @rule <key> [<key> ...]` ... `# @rule end` comments (dropped from the
code shown); shared lines, such as on/off variables used by several rules, carry several keys. The
page highlights the picked rule's lines in pink in both columns; it and "Add a new rule" are never
active together.

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

Options: `--model <name>` and `--tool <name>` (repeatable), `--time-limit <seconds>` (default 600).
Each implementation, and the AMPL model with each solver in `ampl_solvers`, runs in its own
subprocess. The time limit is applied by `validate.py` to the tool's own model object right before
its solve block (the code on the page has none). A solve that reaches it without proving optimality
is recorded as `unverified` with the best objective, bound and gap; it is not counted as a failure.
The script exits with status 1 if any implementation misses the expected objective.
The free pip licenses of gurobipy, cplex and xpress limit model size (CPLEX Community Edition:
1000 variables and 1000 constraints), so comparison instances stay within those limits.
