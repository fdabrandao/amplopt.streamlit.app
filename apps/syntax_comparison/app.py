from html import escape
from pathlib import Path

import streamlit as st

from .catalog import COMPARISONS, FULL_BLOCKS, MODEL_BLOCKS, changed_lines, load_models, load_tools
from .render import headline_tiles, info_chip, render_grid

REPO_URL = "https://github.com/fdabrandao/amplopt.streamlit.app"
SOURCE_URL = f"{REPO_URL}/blob/master/apps/syntax_comparison/comparisons"
VIEWS = {"Model only": MODEL_BLOCKS, "Model + data loading": FULL_BLOCKS}


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


def column_header(tool, info, model, block_names, changed_loc=None):
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
    change = f" · {changed_loc} added or changed for the new rule" if changed_loc is not None else ""
    html.append(
        f'<div style="color:#9CA3AF;font-size:11.5px;margin-top:4px;">{loc} lines of code{change} · {links}</div>'
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


def main():
    page_style()
    tools, models = get_catalog()

    st.title("⚖️ Modeling Syntax Comparison")
    st.markdown(
        "The same optimization models in **AMPL** and in other Python modeling tools, side by side. "
        "All read the same CSV data and solve the same problem; each concept has its own color, so the "
        "code lines up across the two columns."
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
    if "sc_notes" not in st.session_state:
        st.session_state.sc_notes = params.get("notes") != "0"

    c1, c2, c3 = st.columns([5, 5, 3], vertical_alignment="bottom")
    with c1:
        model = base_models[
            st.selectbox("Model", list(base_models), key="sc_model", format_func=lambda k: base_models[k].title)
        ]
    with c2:
        tool = st.selectbox(
            "Compare AMPL with",
            others,
            key="sc_tool",
            format_func=lambda t: f"{tools[t]['name']}  ({tools[t]['group']})",
        )
    with c3:
        view = st.radio("Show", list(VIEWS), label_visibility="collapsed")

    if model.meta.get("headline"):
        st.html(headline_tiles(model.meta["headline"]))
    st.caption(" ".join(model.meta.get("summary", "").split()))

    variant = None
    c1, c2 = st.columns([4, 1], vertical_alignment="center")
    with c2:
        show_notes = st.toggle("📝 Show notes", key="sc_notes")
    if model.variants:
        variants = {v.key: v for v in model.variants}
        options = ["none"] + list(variants)
        if st.session_state.sc_change == "1":  # older links: first change
            st.session_state.sc_change = options[1]
        if st.session_state.sc_change not in options:
            st.session_state.sc_change = "none"
        with c1:
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
                "changed in each implementation for this rule."
            )
    st.query_params.update(
        {"model": model.key, "tool": tool, "change": variant.key if variant else "none", "notes": "1" if show_notes else "0"}
    )
    show_change = variant is not None
    shown_model = variant if show_change else model
    notes = notes_for([variant, model] if variant else [model], tool) if show_notes else ({}, {})

    shown = [shown_model.implementations[t] for t in ("ampl", tool) if t in shown_model.implementations]
    block_names = [b for b in VIEWS[view] if any(b in impl.blocks for impl in shown)]
    columns = []
    for t, column_notes in zip(("ampl", tool), notes):
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
        not_supported = shown_model.not_supported(t)
        columns.append(
            {
                "name": info["name"],
                "header": column_header(t, info, shown_model, block_names, changed_loc),
                "blocks": impl.blocks if impl else {},
                "not_supported": not_supported,
                "changed": changed,
                "notes": {} if not_supported else column_notes,
            }
        )
    st.html(render_grid(columns, block_names))

    st.caption(
        "The implementations are run and compared offline (validate.py in the source code). "
        "Line counts exclude blank lines and comments, and cover the blocks shown."
    )
    st.markdown(f"#### [[App Source Code on GitHub]({REPO_URL}/tree/master/apps/syntax_comparison)]")
