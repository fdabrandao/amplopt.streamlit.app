"""HTML rendering of the side-by-side, color-banded code grid (inline styles only)."""

import re
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
CODE_FONT_SIZE = "13.5px"  # two columns at a booth monitor: readable from a step away
CHANGED_COLOR = "#16A34A"  # lines added or modified by a model change
RULE_COLOR = "#F63366"  # lines implementing the business rule picked under "Highlight a rule"
# Notes are dark so they never look like a code block (code blocks are all light tints).
NOTE_BG = "#1E293B"
ACCENT = "#F63366"  # notes and the business context card


def _tint(hex_color, alpha):
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def highlight_code(code, language, changed=frozenset(), marked=frozenset()):
    """Highlight line by line so long lines can wrap with a hanging indent instead of scrolling.

    `changed` holds the indexes of lines to mark as added/modified; `marked` the lines of the
    business rule picked on the page.
    """
    lexer = get_lexer_by_name(language)
    lines = []
    for i, line in enumerate(code.splitlines()):
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if not stripped:
            lines.append('<div style="height:1.45em;"></div>')
            continue
        html = highlight(stripped, lexer, _FORMATTER).rstrip("\n")
        color = CHANGED_COLOR if i in changed else RULE_COLOR if i in marked else None
        mark = f"background:{_tint(color, 0.18)};box-shadow:-6px 0 0 {color};" if color else ""
        lines.append(
            f'<div style="white-space:pre-wrap;overflow-wrap:break-word;{mark}'
            f'padding-left:{indent + 4}ch;text-indent:-4ch;">{html}</div>'
        )
    return "".join(lines)


def context_card(text, rules, optional=()):
    """The business problem: a short paragraph, then the rules (lists of {label, text}).

    `optional` rules are the ones added under "Add a new rule".
    """
    item = lambda r, tag="": (
        f'<li style="margin:2px 0;"><b>{escape(r["label"])}</b>{tag}: {escape(" ".join(r["text"].split()))}</li>'
    )
    tag = ' <span style="color:#6B7280;">(optional, under “Add a new rule”)</span>'
    items = "".join(item(r) for r in rules) + "".join(item(r, tag) for r in optional)
    return (
        f'<div style="display:flex;flex-wrap:wrap;gap:6px 28px;background:#F7F8FA;border-radius:8px;'
        f'border-left:4px solid {ACCENT};padding:10px 16px;margin:2px 0 8px 0;font-size:14px;'
        'line-height:1.5;color:#1F2937;">'
        f'<div style="flex:1 1 340px;">{escape(" ".join(text.split()))}</div>'
        f'<div style="flex:1.4 1 420px;"><ul style="margin:0;padding-left:18px;">{items}</ul></div>'
        "</div>"
    )


def _inline(text):
    """Escape a note and render `code` spans."""
    html = escape(" ".join(text.split()))
    return re.sub(
        r"`([^`]+)`",
        lambda m: f'<code style="font-family:{_MONO};font-size:.9em;background:rgba(255,255,255,.12);'
        f'padding:0 4px;border-radius:3px;color:#FDE68A;">{m.group(1)}</code>',
        html,
    )


def note_cell(texts):
    """A dark note card with one paragraph per note."""
    body = "".join(f'<div style="margin:2px 0;">{_inline(t)}</div>' for t in texts)
    return (
        f'<div style="background:{NOTE_BG};border-left:5px solid {ACCENT};border-radius:6px;'
        'padding:8px 12px;font-size:13.5px;line-height:1.5;color:#F1F5F9;'
        'box-shadow:0 1px 3px rgba(15,23,42,.25);display:flex;gap:8px;">'
        f'<div>📝</div><div>{body}</div></div>'
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


def info_chip(text):
    return _chip(text, "#475569")


def _code_cell(block_name, block, changed=frozenset(), marked=frozenset()):
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
        f'<div style="margin:0;font-family:{_MONO};font-size:{CODE_FONT_SIZE};line-height:1.45;color:#1F2937;">'
        f"{highlight_code(block.code, block.language, changed, marked)}</div>"
    )
    return f'<div style="{base}">{label}{pre}</div>'


def render_grid(columns, block_names):
    """columns: list of dicts with keys
    - header: HTML for the column header
    - blocks: dict block name -> Block (ignored when `not_supported` is set)
    - not_supported: optional reason (renders one card in place of the code)
    - changed: optional dict block name -> indexes of lines to highlight as changed
    - marked: optional dict block name -> indexes of lines of the highlighted business rule
    - notes: optional dict "header" or block name -> list of note texts shown in that column,
      just above the block (or just below the header)
    """
    n = len(columns)
    has_notes = lambda at: any(col.get("notes", {}).get(at) for col in columns)
    rows = [("header", None)] + ([("note", "header")] if has_notes("header") else [])
    for name in block_names:
        rows += ([("note", name)] if has_notes(name) else []) + [("block", name)]

    cells = []
    for c, col in enumerate(columns, start=1):
        first_block = True
        for r, (kind, name) in enumerate(rows, start=1):
            place = f"grid-column:{c};grid-row:{r};min-width:0;"
            if kind == "header":
                cells.append(f'<div style="{place}padding:4px 2px 6px 2px;border-bottom:2px solid #E5E7EB;">{col["header"]}</div>')
            elif kind == "note":
                texts = col.get("notes", {}).get(name)
                if texts:
                    cells.append(f'<div style="{place}">{note_cell(texts)}</div>')
            elif col.get("not_supported"):
                if first_block:
                    cells.append(
                        f'<div style="{place}border:1px dashed #D1D5DB;border-radius:6px;padding:24px 16px;'
                        'color:#4B5563;background:#F9FAFB;align-self:start;">'
                        f'<div style="font-weight:700;margin-bottom:6px;">Not supported by {escape(col["name"])}</div>'
                        f'<div style="font-size:13px;">{escape(col["not_supported"])}</div></div>'
                    )
                first_block = False
            else:
                cells.append(
                    f'<div style="{place}display:flex;"><div style="flex:1;min-width:0;display:flex;flex-direction:column;">'
                    f"{_code_cell(name, col['blocks'].get(name), col.get('changed', {}).get(name, frozenset()), col.get('marked', {}).get(name, frozenset()))}"
                    "</div></div>"
                )
    grid_style = (
        f"display:grid;grid-template-columns:repeat({n}, minmax(0, 1fr));"
        "column-gap:14px;row-gap:8px;align-items:stretch;"
    )
    return f'<div style="{grid_style}">{"".join(cells)}</div>'
