"""Load the comparison catalog: tools, models, code blocks and validation results.

No Streamlit here, so it can be reused by scripts and tested on its own.
"""

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import toml

COMPARISONS = Path(__file__).resolve().parent / "comparisons"

# Canonical block order and labels. The page aligns blocks across tools by these names.
BLOCKS = {
    "setup": "Setup",
    "sets": "Sets",
    "params": "Parameters",
    "vars": "Variables",
    "objective": "Objective",
    "constraints": "Constraints",
    "data": "Data loading",
    "solve": "Solve",
}
MODEL_BLOCKS = ["sets", "params", "vars", "objective", "constraints"]
FULL_BLOCKS = ["setup"] + MODEL_BLOCKS + ["data", "solve"]

_MARKER = re.compile(r"^\s*#\s*@block\s+(\w+)\s*$")
# `# @rule <key> [<key> ...]` tags the following lines with model rules (meta.toml [[rules]]) until
# `# @rule end` or the next block; the page highlights a rule's lines on demand. Markers are dropped.
_RULE = re.compile(r"^\s*#\s*@rule\s+(.+?)\s*$")
_LANGUAGES = {".mod": "ampl", ".py": "python"}


@dataclass
class Block:
    name: str
    code: str
    language: str
    source: str  # file name the block comes from
    rules: dict = field(default_factory=dict)  # line index -> frozenset of rule keys

    @property
    def lines_of_code(self):
        """Non-blank lines that are not just comments."""
        return sum(1 for line in self.code.splitlines() if line.strip() and not line.strip().startswith("#"))


@dataclass
class Implementation:
    tool: str
    files: list
    blocks: dict = field(default_factory=dict)  # block name -> Block


@dataclass
class Model:
    key: str
    path: Path
    meta: dict
    implementations: dict = field(default_factory=dict)  # tool -> Implementation

    variants: list = field(default_factory=list)  # models that add one change to this one

    @property
    def title(self):
        return self.meta["title"]

    @property
    def variant_of(self):
        return self.meta.get("variant_of")

    def not_supported(self, tool):
        return self.meta.get("not_supported", {}).get(tool)


def changed_lines(base_code, new_code):
    """Indexes of the lines in new_code that were added or modified with respect to base_code."""
    base, new = base_code.splitlines(), new_code.splitlines()
    changed = set()
    for tag, _, _, j1, j2 in difflib.SequenceMatcher(None, base, new, autojunk=False).get_opcodes():
        if tag in ("insert", "replace"):
            changed.update(range(j1, j2))
    return {i for i in changed if new[i].strip()}  # blank lines are not "changes"


def parse_blocks(text, language, source):
    """Split a file into blocks using `# @block <name>` markers; `# @rule` markers tag lines.

    Markers are dropped from the code shown.
    """
    blocks, name, lines, tags = {}, None, [], frozenset()

    def flush():
        if name is None:
            return
        # Same display rules for every tool: no leading/trailing blank lines, at most one blank line in a row
        kept = []
        for line, line_tags in lines:
            if line.strip():
                kept.append((line, line_tags))
            elif kept and kept[-1][0]:
                kept.append(("", frozenset()))
        while kept and not kept[-1][0]:
            kept.pop()
        code = "\n".join(line for line, _ in kept)
        rules = {i: line_tags for i, (_, line_tags) in enumerate(kept) if line_tags}
        if name in blocks:  # a block may be split in several parts of a file
            previous = blocks[name]
            offset = len(previous.code.splitlines()) + 1
            code = previous.code + "\n\n" + code
            rules = {**previous.rules, **{i + offset: t for i, t in rules.items()}}
        blocks[name] = Block(name, code, language, source, rules)

    for line in text.splitlines():
        match = _MARKER.match(line)
        if match:
            flush()
            name, lines, tags = match.group(1), [], frozenset()
            continue
        rule = _RULE.match(line)
        if rule:
            tags = frozenset() if rule.group(1) == "end" else frozenset(rule.group(1).split())
            continue
        lines.append((line, tags))
    flush()
    return blocks


def load_tools():
    return toml.load(COMPARISONS / "tools.toml")


def load_validation():
    path = COMPARISONS / "validation.json"
    return json.loads(path.read_text()) if path.exists() else {"models": {}, "versions": {}}


def load_models(tools):
    models = []
    for meta_path in sorted(COMPARISONS.glob("*/meta.toml")):
        model_dir = meta_path.parent
        model = Model(model_dir.name, model_dir, toml.load(meta_path))
        for tool, info in tools.items():
            files = [model_dir / name for name in info["files"]]
            if model.not_supported(tool) or not all(f.exists() for f in files):
                continue
            impl = Implementation(tool, [f.name for f in files])
            for f in files:
                impl.blocks.update(parse_blocks(f.read_text(), _LANGUAGES[f.suffix], f.name))
            model.implementations[tool] = impl
        models.append(model)
    by_key = {m.key: m for m in models}
    models.sort(key=lambda m: (m.meta.get("order", 99), m.title))
    for m in models:
        if m.variant_of in by_key:
            by_key[m.variant_of].variants.append(m)
    return models
