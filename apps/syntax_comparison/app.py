import time
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from .catalog import COMPARISONS, FULL_BLOCKS, MODEL_BLOCKS, changed_lines, load_models, load_tools, load_validation
from .render import badge, headline_tiles, info_chip, render_grid

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


def _gap(entry):
    return f"gap {entry['gap']:.2%}" if "gap" in entry else "gap unknown"


def column_header(tool, info, model, validation, block_names, changed_loc=None):
    result = validation.get("models", {}).get(model.key, {}).get(tool, {})
    versions = validation.get("versions", {})
    minutes = f"{validation.get('time_limit', 600) / 60:g} min"
    amber = "color:#92400E;font-size:11.5px;margin:2px 0 4px 0;"
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
        html.append(f'<div style="{amber}">{escape(limited)}</div>')
    elif result.get("status") == "unverified":
        html.append(badge(f"⚠ unverified: optimality not proven in {minutes}", ok=False))
        if "objective" in result:
            html.append(
                f'<div style="{amber}">Best objective found: {result["objective"]:,.2f} '
                f"({_gap(result)} when the {minutes} time limit was reached)</div>"
            )
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
    solvers_open = [
        f"{SOLVER_NAMES.get(s, s)} ({_gap(r)})" for s, r in result.get("solvers", {}).items() if r.get("status") == "unverified"
    ]
    if solvers_open:
        html.append(f'<div style="{amber}">Optimality not proven in {minutes}: {escape(", ".join(solvers_open))}</div>')

    impl = model.implementations[tool]
    loc = sum(impl.blocks[b].lines_of_code for b in block_names if b in impl.blocks)
    links = " · ".join(
        f'<a href="{SOURCE_URL}/{model.key}/{name}" target="_blank" style="color:#6B7280;">{escape(name)}</a>'
        for name in impl.files
    )
    change = f" · {changed_loc} added or changed for the new rule" if changed_loc is not None else ""
    html.append(
        f'<div style="color:#9CA3AF;font-size:11.5px;margin-top:4px;">{loc} lines of code{change} · {links}</div>'
    )
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
    if model.meta.get("solve_note") and solver in model.meta.get("slow_solvers", []):
        st.warning(f"⏱ {' '.join(model.meta['solve_note'].split())}")
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
        can match the code across the two columns.
        """
    )

    params = st.query_params
    base_models = {m.key: m for m in models if not m.variant_of}
    others = [t for t in tools if t != "ampl"]
    # Seed the widgets from the URL once; afterwards the widgets own their state. Passing a
    # URL-derived `index`/`value` on every run would make Streamlit recreate the widget whenever
    # the URL changes, discarding the user's next selection.
    if "sc_model" not in st.session_state:
        st.session_state.sc_model = params.get("model") if params.get("model") in base_models else next(iter(base_models))
    if "sc_tool" not in st.session_state:
        tool = params.get("tool", params.get("tools", "").split(",")[0])
        st.session_state.sc_tool = tool if tool in others else "pyomo"
    if "sc_change" not in st.session_state:
        st.session_state.sc_change = params.get("change", "none")

    model = base_models[
        st.selectbox("Pick a model 👇", list(base_models), key="sc_model", format_func=lambda k: base_models[k].title)
    ]
    if model.meta.get("headline"):
        st.html(headline_tiles(model.meta["headline"]))
    tool = st.selectbox(
        "Compare AMPL with 👇",
        others,
        key="sc_tool",
        format_func=lambda t: f"{tools[t]['name']}  ({tools[t]['group']})",
    )
    view = st.radio("Show", list(VIEWS), horizontal=True, label_visibility="collapsed")

    st.markdown(f"#### {model.title}")
    st.markdown(" ".join(model.meta.get("summary", "").split()))
    talking_points(model)

    variant = None
    if model.variants:
        variants = {v.key: v for v in model.variants}
        options = ["none"] + list(variants)
        if st.session_state.sc_change == "1":  # older links: first change
            st.session_state.sc_change = options[1]
        if st.session_state.sc_change not in options:
            st.session_state.sc_change = "none"
        choice = st.radio(
            "✚ Add a new rule",
            options,
            key="sc_change",
            horizontal=True,
            format_func=lambda k: "None (base model)" if k == "none" else variants[k].meta.get("change_name", k),
        )
        variant = variants.get(choice)
        if variant:
            st.success(
                f"**{variant.meta['change']}.** Highlighted lines are everything that had to be added or "
                "changed in each implementation for this rule. Both versions were validated to the same "
                "optimal objective."
            )
    st.query_params.update({"model": model.key, "tool": tool, "change": variant.key if variant else "none"})
    show_change = variant is not None
    shown_model = variant if show_change else model

    shown = [shown_model.implementations[t] for t in ("ampl", tool) if t in shown_model.implementations]
    block_names = [b for b in VIEWS[view] if any(b in impl.blocks for impl in shown)]
    columns = []
    for t in ("ampl", tool):
        info = tools[t]
        impl = shown_model.implementations.get(t)
        changed, changed_loc = {}, None
        if show_change and impl and t in model.implementations:
            base_blocks = model.implementations[t].blocks
            for b in block_names:
                if b in impl.blocks:
                    base_code = base_blocks[b].code if b in base_blocks else ""
                    changed[b] = changed_lines(base_code, impl.blocks[b].code)
            changed_loc = sum(
                1
                for b, idx in changed.items()
                for i, line in enumerate(impl.blocks[b].code.splitlines())
                if i in idx and not line.strip().startswith("#")
            )
        columns.append(
            {
                "name": info["name"],
                "header": column_header(t, info, shown_model, validation, block_names, changed_loc),
                "blocks": impl.blocks if impl else {},
                "not_supported": shown_model.not_supported(t),
                "changed": changed,
            }
        )
    st.html(render_grid(columns, block_names))

    st.markdown("---")
    live_solve(shown_model)

    with st.expander("How this comparison was validated"):
        versions = validation.get("versions", {})
        st.markdown(
            f"""
            - Only the AMPL model runs live on this page; the other implementations are shown as static
              files. `validate.py` runs all of them offline and checks that they reach the same
              optimal objective (last run: {validation.get('generated', 'n/a')}, Python {validation.get('python', 'n/a')}).
            - Each solve has a {validation.get('time_limit', 600) / 60:g}-minute time limit. A solve that reaches it
              before proving optimality is marked *unverified*, with the remaining optimality gap.
            - Library versions: {', '.join(f'`{k} {v}`' for k, v in versions.items() if v)}.
            - Line counts exclude blank lines and comments, and cover the blocks currently shown.
            """
        )

    st.markdown(f"#### [[App Source Code on GitHub]({REPO_URL}/tree/master/apps/syntax_comparison)]")
