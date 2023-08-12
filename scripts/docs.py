#!/usr/bin/env python3

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT))

from typing import get_type_hints

from lazydocs import MarkdownGenerator
from ctransformers import Config, LLM, AutoModelForCausalLM
from ctransformers.llm import docs as config_docs

# Config Docs

docs = """

### Config

| Parameter | Type  | Description | Default |
| :-------- | :---- | :---------- | :------ |
"""
for param, description in config_docs.items():
    if param == "stop":
        type_ = "List[str]"
    else:
        type_ = get_type_hints(Config)[param].__name__
    default = getattr(Config, param)
    docs += f"| `{param}` | `{type_}` | {description} | `{default}` |\n"
docs += """
> **Note:** Currently only LLaMA, MPT and Falcon models support the `context_length` parameter.
"""

# Class Docs

generator = MarkdownGenerator()
for class_ in (AutoModelForCausalLM, LLM):
    docs += generator.class2md(class_, depth=3)
docs += "---\n" + generator.func2md(LLM.__call__, clsname="LLM", depth=4)

# Save

README = ROOT / "README.md"
MARKER = "<!-- API_DOCS -->"

with open(README) as f:
    contents = f.read()

parts = contents.split(MARKER)
if len(parts) != 3:
    raise RuntimeError(
        f"README should have exactly 2 '{MARKER}' but has {len(parts) - 1}."
    )
parts[1] = docs
contents = MARKER.join(parts)

with open(README, mode="w") as f:
    f.write(contents)
