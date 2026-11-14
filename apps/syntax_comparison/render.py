"""HTML rendering of the side-by-side, color-banded code grid (inline styles only)."""

from html import escape

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

from .catalog import BLOCKS

# One color per concept, used for the left band and a light tint of the cell.
BLOCK_COLORS = {
    "setup": "#6B7280",
    "sets": "#2563EB",
    "params": "#7C3AED",
    "vars": "#059669",
    "objective": "#D97706",
    "constraints": "#DC2626",
    "data": "#0891B2",
    "solve": "#475569",
}
_FORMATTER = HtmlFormatter(nowrap=True, noclasses=True, style="friendly")
_MONO = "'Source Code Pro', 'SFMono-Regular', Menlo, Consolas, monospace"


def _tint(hex_color, alpha):
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def highlight_code(code, language):
    """Highlight line by line so long lines can wrap with a hanging indent instead of scrolling."""
    lexer = get_lexer_by_name(language)
    lines = []
    for line in code.splitlines():
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if not stripped:
            lines.append('<div style="height:1.45em;"></div>')
            continue
        html = highlight(stripped, lexer, _FORMATTER).rstrip("\n")
        lines.append(
            f'<div style="white-space:pre-wrap;overflow-wrap:break-word;'
            f'padding-left:{indent + 4}ch;text-indent:-4ch;">{html}</div>'
        )
    return "".join(lines)


def code_panel(code, language, max_height=None):
    """A plain code box with wrapped lines (st.code in this Streamlit version cannot wrap)."""
    scroll = f"max-height:{max_height}px;overflow-y:auto;" if max_height else ""
    return (
        f'<div style="background:#F8F9FB;border:1px solid #E5E7EB;border-radius:6px;padding:10px 12px;{scroll}'
        f'font-family:{_MONO};font-size:12.5px;line-height:1.45;color:#1F2937;">'
        f"{highlight_code(code, language)}</div>"
    )


def _chip(text, color, filled=False):
    style = (
        f"display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;"
        f"font-weight:600;letter-spacing:.02em;white-space:nowrap;margin:2px 4px 2px 0;"
    )
    if filled:
        style += f"background:{color};color:#fff;"
    else:
        style += f"background:{_tint(color, 0.10)};color:{color};border:1px solid {_tint(color, 0.35)};"
    return f'<span style="{style}">{escape(text)}</span>'


def badge(text, ok=True):
    return _chip(text, "#15803D" if ok else "#B45309")


def info_chip(text):
    return _chip(text, "#475569")


def _code_cell(block_name, block):
    color = BLOCK_COLORS.get(block_name, "#6B7280")
    label = (
        f'<div style="font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;'
        f'color:{color};margin-bottom:4px;">{escape(BLOCKS.get(block_name, block_name))}</div>'
    )
    base = (
        f"border-left:4px solid {color};background:{_tint(color, 0.06)};border-radius:4px;"
        "padding:6px 10px 8px 10px;min-width:0;flex:1;"
    )
    if block is None or not block.code.strip():
        body = '<div style="color:#9CA3AF;font-size:12px;font-style:italic;">not needed</div>'
        return f'<div style="{base}opacity:.6;">{label}{body}</div>'
    pre = (
        f'<div style="margin:0;font-family:{_MONO};font-size:12.5px;line-height:1.45;color:#1F2937;">'
        f"{highlight_code(block.code, block.language)}</div>"
    )
    return f'<div style="{base}">{label}{pre}</div>'


def render_grid(columns, block_names):
    """columns: list of dicts with keys
    - header: HTML for the column header
    - blocks: dict block name -> Block (ignored when `not_supported` is set)
    - not_supported: optional reason (renders one card spanning all block rows)
    """
    n = len(columns)
    cells = []
    for c, col in enumerate(columns, start=1):
        cells.append(
            f'<div style="grid-column:{c};grid-row:1;padding:4px 2px 8px 2px;'
            f'border-bottom:2px solid #E5E7EB;min-width:0;">{col["header"]}</div>'
        )
        if col.get("not_supported"):
            cells.append(
                f'<div style="grid-column:{c};grid-row:2 / span {len(block_names)};border:1px dashed #D1D5DB;'
                'border-radius:6px;padding:24px 16px;color:#4B5563;background:#F9FAFB;align-self:start;">'
                f'<div style="font-weight:700;margin-bottom:6px;">Not supported by {escape(col["name"])}</div>'
                f'<div style="font-size:13px;">{escape(col["not_supported"])}</div></div>'
            )
            continue
        for r, name in enumerate(block_names, start=2):
            cells.append(
                f'<div style="grid-column:{c};grid-row:{r};display:flex;min-width:0;">'
                f'<div style="flex:1;min-width:0;display:flex;flex-direction:column;">'
                f"{_code_cell(name, col['blocks'].get(name))}</div></div>"
            )
    grid_style = (
        f"display:grid;grid-template-columns:repeat({n}, minmax(0, 1fr));"
        "column-gap:14px;row-gap:10px;align-items:stretch;"
    )
    return f'<div style="{grid_style}">{"".join(cells)}</div>'
