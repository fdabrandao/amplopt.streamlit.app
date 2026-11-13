"""Load the comparison catalog: tools, models, code blocks and validation results.

No Streamlit here, so it can be reused by scripts and tested on its own.
"""

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
_LANGUAGES = {".mod": "ampl", ".py": "python"}


@dataclass
class Block:
    name: str
    code: str
    language: str
    source: str  # file name the block comes from

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

    @property
    def title(self):
        return self.meta["title"]

    def not_supported(self, tool):
        return self.meta.get("not_supported", {}).get(tool)


def parse_blocks(text, language, source):
    """Split a file into blocks using `# @block <name>` markers (markers are dropped)."""
    blocks, name, lines = {}, None, []

    def flush():
        if name is None:
            return
        # Same display rules for every tool: no leading/trailing blank lines, at most one blank line in a row
        code = re.sub(r"\n\s*\n(\s*\n)+", "\n\n", "\n".join(lines)).strip("\n")
        if name in blocks:  # a block may be split in several parts of a file
            code = blocks[name].code + "\n\n" + code
        blocks[name] = Block(name, code, language, source)

    for line in text.splitlines():
        match = _MARKER.match(line)
        if match:
            flush()
            name, lines = match.group(1), []
        else:
            lines.append(line)
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
    return sorted(models, key=lambda m: (m.meta.get("order", 99), m.title))
