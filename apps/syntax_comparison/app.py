from html import escape
from pathlib import Path

import streamlit as st

from .catalog import COMPARISONS, MODEL_BLOCKS, changed_lines, load_models, load_tools
from .render import context_card, info_chip, note_cell, render_grid

REPO_URL = "https://github.com/fdabrandao/amplopt.streamlit.app"
SOURCE_URL = f"{REPO_URL}/blob/master/apps/syntax_comparison/comparisons"
# The grid shows the model; the amplpy code that loads the data and solves it is shown once, below it.
PYTHON_BLOCKS = ["setup", "data", "solve"]


def _fingerprint():
    """Changes whenever a comparison file (or the loader) is edited, refreshing the cache."""
    files = [p for p in COMPARISONS.rglob("*") if p.is_file()] + [Path(__file__).with_name("catalog.py")]
    return max(p.stat().st_mtime_ns for p in files)


@st.cache_resource(max_entries=1)
def _load_catalog(fingerprint):
    tools = load_tools()
    return tools, load_models(tools)


def get_catalog():
    return _load_catalog(_fingerprint())


def page_style():
    # Two code columns need more room than Streamlit's centered layout; a tighter title and spacing
    # keep the controls and the first code rows on screen.
    st.markdown(
        """<style>
        .block-container, [data-testid="stMainBlockContainer"] { max-width: 1500px; padding-top: 3.5rem; }
        .block-container h1 { font-size: 1.9rem; padding: 0.4rem 0 0.2rem 0; }
        </style>""",
        unsafe_allow_html=True,
    )


def column_header(tool, info, model, block_names, extra=None):
    html = [
        f'<div style="font-weight:700;font-size:16px;color:#111827;">{escape(info["name"])} '
        f'<span style="font-weight:400;color:#6B7280;font-size:12px;">· {escape(info["group"])}</span></div>',
    ]
    if model.not_supported(tool):
        return "".join(html)
    html.append(info_chip(f"Solver: {info['solver']}"))
    impl = model.implementations[tool]
    loc = sum(impl.blocks[b].lines_of_code for b in block_names if b in impl.blocks)
    links = " · ".join(
        f'<a href="{SOURCE_URL}/{model.key}/{name}" target="_blank" style="color:#6B7280;">{escape(name)}</a>'
        for name in impl.files
    )
    extra = f" · {escape(extra)}" if extra else ""
    html.append(
        f'<div style="color:#9CA3AF;font-size:11.5px;margin-top:4px;">{loc} lines of code{extra} · {links}</div>'
    )
    return "".join(html)


def notes_for(models, tool):
    """Notes per column: tool-independent notes go under AMPL, `tools`-limited ones under the tool."""
    ampl, other = {}, {}
    for model in models:
        for note in model.meta.get("notes", []):
            if "tools" not in note:
                ampl.setdefault(note["at"], []).append(note["text"])
            elif tool in note["tools"]:
                other.setdefault(note["at"], []).append(note["text"])
    return ampl, other


def _code_lines(block, indexes):
    """How many of these lines are code (not comments)."""
    return sum(1 for i, line in enumerate(block.code.splitlines()) if i in indexes and not line.strip().startswith("#"))


# "Highlight a rule" and "Add a new rule" never apply together: picking one resets the other.
def _on_rule():
    if st.session_state.sc_rule != "none":
        st.session_state.sc_change = "none"


def _on_change():
    if st.session_state.sc_change != "none":
        st.session_state.sc_rule = "none"


def _changes(model, variant, tool, block_names):
    """Lines of each block that the variant adds or changes in a tool's implementation."""
    impl, base = variant.implementations.get(tool), model.implementations.get(tool)
    if not impl or not base:
        return {}
    return {
        b: changed_lines(base.blocks[b].code if b in base.blocks else "", impl.blocks[b].code)
        for b in block_names
        if b in impl.blocks
    }


def python_section(model, variant, notes):
    """The amplpy code around the AMPL model: it reads the model file, loads the data and solves."""
    shown_model = variant or model
    impl = shown_model.implementations["ampl"]
    blocks = [b for b in PYTHON_BLOCKS if b in impl.blocks]
    loc = sum(impl.blocks[b].lines_of_code for b in blocks)
    name = next(f for f in impl.files if f.endswith(".py"))
    header = (
        f'<div style="font-weight:700;font-size:16px;color:#111827;">{escape(name)} '
        f'<span style="font-weight:400;color:#6B7280;font-size:12px;">· AMPL from Python (amplpy)</span></div>'
        f'<div style="color:#9CA3AF;font-size:11.5px;margin-top:4px;">{loc} lines of code · '
        f'<a href="{SOURCE_URL}/{shown_model.key}/{name}" target="_blank" style="color:#6B7280;">{escape(name)}</a></div>'
    )
    st.markdown("#### Loading the data and solving, from Python")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.html(render_grid([{"name": "AMPL", "header": header, "blocks": impl.blocks}], blocks))
    with c2:
        if notes:
            st.html(note_cell(notes))


def main():
    page_style()
    tools, models = get_catalog()

    st.title("⚖️ Modeling Syntax Comparison")
    st.markdown(
        "The same production-planning model in **AMPL** (left) and in another Python modeling tool "
        "(right). Each part of the model has its own color, so the two columns line up; pick a "
        "business rule to highlight its code in both, or add a new rule to see what each one has to change."
    )

    params = st.query_params
    base_models = {m.key: m for m in models if not m.variant_of}
    others = [t for t in tools if t != "ampl"]
    # Seed the widgets from the URL once; afterwards the widgets own their state. Passing a
    # URL-derived `index`/`value` on every run would make Streamlit recreate the widget whenever
    # the URL changes, discarding the user's next selection.
    if st.session_state.get("sc_model") not in base_models:
        st.session_state.sc_model = params.get("model") if params.get("model") in base_models else next(iter(base_models))
    if "sc_tool" not in st.session_state:
        tool = params.get("tool", params.get("tools", "").split(",")[0])
        st.session_state.sc_tool = tool if tool in others else "pyomo"
    if "sc_change" not in st.session_state:
        st.session_state.sc_change = params.get("change", "none")
    if "sc_rule" not in st.session_state:
        st.session_state.sc_rule = params.get("rule", "none")
    if "sc_notes" not in st.session_state:
        st.session_state.sc_notes = params.get("notes") != "0"

    if len(base_models) > 1:
        st.selectbox("Model", list(base_models), key="sc_model", format_func=lambda k: base_models[k].title)
    model = base_models[st.session_state.sc_model]
    rules = {r["key"]: r for r in model.meta.get("rules", [])}
    variants = {v.key: v for v in model.variants}

    if model.meta.get("context"):
        optional = [{"label": v.meta["change_name"], "text": v.meta["change"]} for v in model.variants]
        st.html(context_card(model.meta["context"], list(rules.values()), optional))

    c1, c2 = st.columns([5, 3], vertical_alignment="bottom")
    with c1:
        tool = st.selectbox(
            "Compare AMPL with",
            others,
            key="sc_tool",
            format_func=lambda t: f"{tools[t]['name']}  ({tools[t]['group']})",
        )
    with c2:
        show_notes = st.toggle("📝 Show notes", key="sc_notes")

    for key, valid in (("sc_rule", rules), ("sc_change", variants)):
        if st.session_state.get(key) == "1" and valid:  # older links: first change
            st.session_state[key] = next(iter(valid))
        if st.session_state.get(key) not in ["none", *valid]:
            st.session_state[key] = "none"
    if st.session_state.sc_rule != "none" and st.session_state.sc_change != "none":
        st.session_state.sc_rule = "none"  # e.g. a link with both: the new rule wins

    c1, c2 = st.columns([8, 3], vertical_alignment="center")
    rule = variant = None
    if rules:
        with c1:
            rule = rules.get(
                st.radio(
                    "🔍 Highlight a rule",
                    ["none", *rules],
                    key="sc_rule",
                    horizontal=True,
                    on_change=_on_rule,
                    format_func=lambda k: "None" if k == "none" else rules[k]["label"],
                )
            )
    if variants:
        with c2:
            variant = variants.get(
                st.radio(
                    "✚ Add a new rule",
                    ["none", *variants],
                    key="sc_change",
                    horizontal=True,
                    on_change=_on_change,
                    format_func=lambda k: "None" if k == "none" else variants[k].meta.get("change_name", k),
                )
            )
    st.query_params.update(
        {
            "model": model.key,
            "tool": tool,
            "rule": rule["key"] if rule else "none",
            "change": variant.key if variant else "none",
            "notes": "1" if show_notes else "0",
        }
    )
    shown_model = variant or model
    notes = notes_for([variant, model] if variant else [model], tool) if show_notes else ({}, {})

    shown = [shown_model.implementations[t] for t in ("ampl", tool) if t in shown_model.implementations]
    block_names = [b for b in MODEL_BLOCKS if any(b in impl.blocks for impl in shown)]
    columns = []
    for t, column_notes in zip(("ampl", tool), notes):
        info = tools[t]
        impl = shown_model.implementations.get(t)
        changed, marked, extra = {}, {}, None
        if variant and impl:
            changed = _changes(model, variant, t, block_names)
            n = sum(_code_lines(impl.blocks[b], idx) for b, idx in changed.items())
            extra = f"{n} added or changed for the new rule"
        elif rule and impl:
            for b in block_names:
                if b in impl.blocks:
                    marked[b] = {i for i, keys in impl.blocks[b].rules.items() if rule["key"] in keys}
            n = sum(_code_lines(impl.blocks[b], idx) for b, idx in marked.items())
            extra = f"{n} for this rule"
        not_supported = shown_model.not_supported(t)
        columns.append(
            {
                "name": info["name"],
                "header": column_header(t, info, shown_model, block_names, extra),
                "blocks": impl.blocks if impl else {},
                "not_supported": not_supported,
                "changed": changed,
                "marked": marked,
                "notes": {} if not_supported else column_notes,
            }
        )
    st.html(render_grid(columns, block_names))
    st.caption(
        "Line counts exclude blank lines and comments, and cover the model code shown. Reading the CSV "
        "files and calling the solver are left out of both columns; the AMPL version of that part is below."
    )

    if "ampl" in shown_model.implementations:
        python_section(model, variant, notes[0].get("python", []))

    st.caption("Every implementation on this page is run and compared offline (validate.py in the source code).")
    st.markdown(f"#### [[App Source Code on GitHub]({REPO_URL}/tree/master/apps/syntax_comparison)]")
