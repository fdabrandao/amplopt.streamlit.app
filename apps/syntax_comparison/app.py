import os
import tempfile
import time
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from ..reformulation_explorer.app import match_submodel, model_summary, read_explorer_model, sections_as_code
from .catalog import COMPARISONS, FULL_BLOCKS, MODEL_BLOCKS, load_models, load_tools, load_validation
from .render import badge, code_panel, info_chip, render_grid

REPO_URL = "https://github.com/fdabrandao/amplopt.streamlit.app"
SOURCE_URL = f"{REPO_URL}/blob/master/apps/syntax_comparison/comparisons"
SOLVER_NAMES = {
    "highs": "HiGHS",
    "gurobi": "Gurobi",
    "cplex": "CPLEX",
    "xpress": "Xpress",
    "cbc": "CBC",
    "scip": "SCIP",
    "ipopt": "Ipopt",
    "knitro": "Knitro",
}
VIEWS = {"Model only": MODEL_BLOCKS, "Model + data loading": FULL_BLOCKS}
MAX_TOOLS = 2


def _fingerprint():
    """Changes whenever a comparison file (or the loader) is edited, refreshing the cache."""
    files = [p for p in COMPARISONS.rglob("*") if p.is_file()] + [Path(__file__).with_name("catalog.py")]
    return max(p.stat().st_mtime_ns for p in files)


@st.cache_resource(max_entries=1)
def _load_catalog(fingerprint):
    tools = load_tools()
    return tools, load_models(tools), load_validation()


def get_catalog():
    return _load_catalog(_fingerprint())


def widen_page():
    # Three code columns need more room than Streamlit's centered layout gives this page.
    st.markdown(
        """<style>
        .block-container, [data-testid="stMainBlockContainer"] { max-width: 1500px; }
        </style>""",
        unsafe_allow_html=True,
    )


def column_header(tool, info, model, validation, block_names):
    result = validation.get("models", {}).get(model.key, {}).get(tool, {})
    versions = validation.get("versions", {})
    html = [
        f'<div style="font-weight:700;font-size:17px;color:#111827;">{escape(info["name"])}</div>',
        f'<div style="color:#6B7280;font-size:12px;margin-bottom:6px;">{escape(info["group"])}</div>',
    ]
    if model.not_supported(tool):
        return "".join(html)

    limited = model.meta.get("license_limited", {}).get(tool)
    if result.get("status") == "ok":
        html.append(badge(f"✓ same optimal objective: {result['objective']:,.2f}"))
    elif limited:
        html.append(badge("⚠ not validated: license-limited", ok=False))
        html.append(f'<div style="color:#92400E;font-size:11.5px;margin:2px 0 4px 0;">{escape(limited)}</div>')
    else:
        html.append(badge("not validated", ok=False))
    html.append("<br>")
    package = info["packages"][0]
    if versions.get(package):
        html.append(info_chip(f"{package} {versions[package]}"))
    html.append(info_chip(f"Solver: {info['solver']}"))
    solvers_ok = [SOLVER_NAMES.get(s, s) for s, r in result.get("solvers", {}).items() if r.get("status") == "solved"]
    if solvers_ok:
        html.append(
            f'<div style="color:#15803D;font-size:11.5px;margin-top:2px;">'
            f"Same model validated with {escape(', '.join(solvers_ok))}</div>"
        )

    impl = model.implementations[tool]
    loc = sum(impl.blocks[b].lines_of_code for b in block_names if b in impl.blocks)
    links = " · ".join(
        f'<a href="{SOURCE_URL}/{model.key}/{name}" target="_blank" style="color:#6B7280;">{escape(name)}</a>'
        for name in impl.files
    )
    html.append(f'<div style="color:#9CA3AF;font-size:11.5px;margin-top:4px;">{loc} lines of code · {links}</div>')
    return "".join(html)


def load_ampl(model_key):
    """Run the AMPL setup and data-loading blocks shown on the page; return (model, ampl)."""
    _, models, _ = get_catalog()
    model = next(m for m in models if m.key == model_key)
    blocks = model.implementations["ampl"].blocks
    script = model.path / "ampl_model.py"
    namespace = {"__file__": str(script), "__name__": "__comparison__"}
    exec(compile(blocks["setup"].code + "\n" + blocks["data"].code, str(script), "exec"), namespace)
    return model, namespace["ampl"]


@st.cache_data(show_spinner=False, max_entries=16)
def reformulation_graph(model_key, solver):
    """MP's reformulation graph (JSONL lines) for this model and solver; the solve is stopped at once."""
    _, ampl = load_ampl(model_key)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "graph.jsonl")
        ampl.option["solver"] = solver
        ampl.option[f"{solver}_auxfiles"] = "rc"  # keep the model's names in the graph
        ampl.option[f"{solver}_options"] = f"writegraph={path} lim:time=0"
        ampl.get_output("solve;")
        with open(path) as f:
            return f.read().splitlines()


@st.cache_data(show_spinner=False, max_entries=32)
def solve_ampl(model_key, solver):
    """Run the AMPL implementation shown on the page (setup + data blocks), then solve."""
    model, ampl = load_ampl(model_key)
    ampl.option["solver"] = solver
    start = time.perf_counter()
    output = ampl.get_output("solve;")
    seconds = time.perf_counter() - start
    _, objective = next(iter(ampl.get_objectives()))
    tables = {}
    for name, caption in model.meta.get("display", {}).items():
        values = ampl.get_data(name).to_pandas().iloc[:, 0]
        if isinstance(values.index, pd.MultiIndex):
            table = values.unstack().fillna(0)
            table = table.loc[(table.abs() > 1e-9).any(axis=1)]  # hide all-zero rows (e.g. unused lanes)
            if isinstance(table.index, pd.MultiIndex):  # e.g. (orig, dest) lanes -> "orig → dest"
                table.index = [" → ".join(map(str, key)) for key in table.index]
        else:
            table = values.to_frame(caption)
        table.index.name, table.columns.name = None, None
        tables[caption] = table.round(2)
    return {
        "objective": objective.value(),
        "status": ampl.solve_result,
        "seconds": seconds,
        "output": output,
        "tables": tables,
    }


def live_solve(model):
    st.markdown("### ▶️ Run the AMPL model live")
    solvers = model.meta.get("ampl_solvers", ["highs"])
    col1, col2 = st.columns([1, 2], vertical_alignment="bottom")
    with col1:
        solver = st.selectbox(
            "Solver",
            solvers,
            format_func=lambda s: SOLVER_NAMES.get(s, s),
            key=f"sc_solver_{model.key}",
        )
    with col2:
        st.code(f'ampl.option["solver"] = "{solver}"  # the only line that changes', language="python")
    if st.button(f"Solve with AMPL + {SOLVER_NAMES.get(solver, solver)}", type="primary"):
        st.session_state["sc_solved"] = (model.key, solver)

    if st.session_state.get("sc_solved") != (model.key, solver):
        st.caption("Runs the AMPL model and data loading code shown above (only `ampl.solve()` is replaced).")
        return
    with st.spinner(f"Solving with {SOLVER_NAMES.get(solver, solver)}..."):
        result = solve_ampl(model.key, solver)

    expected = model.meta["expected_objective"]
    same = abs(result["objective"] - expected) <= model.meta.get("rel_tol", 1e-6) * max(1.0, abs(expected))
    c1, c2, c3 = st.columns(3)
    c1.metric(model.meta.get("objective_label", "Objective"), f"{result['objective']:,.2f}")
    c2.metric("Solve result", result["status"])
    c3.metric("Time (incl. model generation)", f"{result['seconds']:.1f} s")
    if result["status"] == "solved" and same:
        st.success("✓ Same optimal objective as every other implementation on this page.")
    else:
        st.warning(f"Objective differs from the validated value ({expected:,.2f}).")
    for caption, table in result["tables"].items():
        st.markdown(f"**{caption}**")
        st.dataframe(table, use_container_width=True)
    with st.expander("Solver output"):
        st.code(result["output"].strip() or "(no output)", language="text")


def reformulation_view(model):
    config = model.meta.get("reformulation")
    if not config:
        return
    st.markdown("### 🔧 What the solver receives")
    st.markdown(
        """
        The model is written once. AMPL's [MP library](https://mp.ampl.com/) rewrites it into what
        the chosen solver accepts: native logical and indicator constraints for solvers that support
        them, an equivalent linear model with binary variables for solvers that don't.
        """
    )
    solver = st.radio(
        "Solver",
        config["solvers"],
        format_func=lambda s: SOLVER_NAMES.get(s, s),
        horizontal=True,
        key=f"sc_reform_{model.key}",
    )
    name = SOLVER_NAMES.get(solver, solver)
    with st.spinner(f"Generating the {name} model..."):
        lines = reformulation_graph(model.key, solver)
    graph = read_explorer_model(lines)
    summary = model_summary(graph)

    nl_cons = summary["nl_constraints"]
    left, right = st.columns(2)
    with left:
        st.markdown("**In the AMPL model**")
        st.markdown(
            f"- Variables: **{summary['nl_variables']}**\n"
            f"- Constraints: **{sum(nl_cons.values())}** "
            f"({', '.join(f'{n} {kind.lower()}' for kind, n in nl_cons.items())})"
        )
    with right:
        st.markdown(f"**Sent to {name}**")
        cons = "\n".join(f"    - {label}: {n}" for label, n in sorted(summary["solver_constraints"].items()))
        st.markdown(
            f"- Variables: **{summary['solver_variables']}** "
            f"({summary['solver_integer_variables']} binary or integer)\n"
            f"- Constraints: **{sum(summary['solver_constraints'].values())}**\n{cons}"
        )

    focus = st.selectbox(
        "Look at one rule 👇",
        config.get("focus", []),
        format_func=lambda f: f["label"],
        key=f"sc_focus_{model.key}",
    )
    if focus:
        original, flat = match_submodel(graph, focus["pattern"], False, False)
        written = {k.removeprefix("NL "): v for k, v in original.get_data().items()}
        left, right = st.columns(2)
        with left:
            st.caption("The rule as AMPL hands it to MP (`a ==> b` is shown as `!a || b`)")
            st.html(code_panel(sections_as_code(written) or "(no match)", "ampl"))
        with right:
            st.caption(f"Sent to {name}, generated automatically")
            st.html(code_panel(sections_as_code(flat.get_data()) or "(no match)", "ampl", max_height=420))

    st.download_button(
        f"Download the full {name} reformulation graph (JSONL)",
        "\n".join(lines) + "\n",
        file_name=f"{model.key}_{solver}_graph.jsonl",
        key=f"sc_graph_{model.key}_{solver}",
    )
    st.caption(
        "Upload it to the [Reformulation Explorer](/Reformulation_Explorer) page to browse every "
        "variable and constraint."
    )


def talking_points(model):
    points = model.meta.get("talking_points", [])
    if not points:
        return
    with st.expander("🗣️ Talking points (presenter notes)"):
        st.markdown("\n".join(f"- **Point at** {p['point']} **Say:** {' '.join(p['say'].split())}" for p in points))


def main():
    widen_page()
    tools, models, validation = get_catalog()

    st.title("⚖️ Modeling Syntax Comparison")
    st.markdown(
        """
        The same optimization models written in **AMPL** and in other Python modeling tools.
        Every implementation reads the same CSV data and was checked to reach the **same optimal
        objective**, so the only difference is how the model is written.
        Each concept (sets, parameters, variables, objective, constraints) has its own color, so you
        can match the code across columns.
        """
    )

    params = st.query_params
    keys = [m.key for m in models]
    model_index = keys.index(params["model"]) if params.get("model") in keys else 0
    model = st.selectbox("Pick a model 👇", models, index=model_index, format_func=lambda m: m.title)

    others = [t for t in tools if t != "ampl"]
    default = [t for t in params.get("tools", "").split(",") if t in others][:MAX_TOOLS] or ["pyomo"]
    chosen = st.multiselect(
        f"Compare AMPL with (up to {MAX_TOOLS}) 👇",
        others,
        default=default,
        max_selections=MAX_TOOLS,
        format_func=lambda t: tools[t]["name"],
        help="Algebraic modeling (open source): Pyomo, PuLP. "
        "Solver-native Python APIs: gurobipy, docplex (CPLEX), xpress (FICO Xpress), highspy (HiGHS).",
    )
    view = st.radio("Show", list(VIEWS), horizontal=True, label_visibility="collapsed")
    st.query_params.update({"model": model.key, "tools": ",".join(chosen)})

    st.markdown(f"#### {model.title}")
    st.markdown(" ".join(model.meta.get("summary", "").split()))
    talking_points(model)

    shown = [model.implementations[t] for t in ["ampl"] + chosen if t in model.implementations]
    block_names = [b for b in VIEWS[view] if any(b in impl.blocks for impl in shown)]
    columns = []
    for tool in ["ampl"] + chosen:
        info = tools[tool]
        reason = model.not_supported(tool)
        impl = model.implementations.get(tool)
        columns.append(
            {
                "name": info["name"],
                "header": column_header(tool, info, model, validation, block_names),
                "blocks": impl.blocks if impl else {},
                "not_supported": reason,
            }
        )
    st.html(render_grid(columns, block_names))

    st.markdown("---")
    live_solve(model)
    if model.meta.get("reformulation"):
        st.markdown("---")
        reformulation_view(model)

    with st.expander("How this comparison was validated"):
        versions = validation.get("versions", {})
        st.markdown(
            f"""
            - Only the AMPL model runs live on this page; the other implementations are shown as static
              files. `validate.py` runs all of them offline and checks that they reach the same
              optimal objective (last run: {validation.get('generated', 'n/a')}, Python {validation.get('python', 'n/a')}).
            - Library versions: {', '.join(f'`{k} {v}`' for k, v in versions.items() if v)}.
            - Line counts exclude blank lines and comments, and cover the blocks currently shown.
            """
        )

    st.markdown(f"#### [[App Source Code on GitHub]({REPO_URL}/tree/master/apps/syntax_comparison)]")
