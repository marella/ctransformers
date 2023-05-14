#!/usr/bin/env python3

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT))

from lazydocs import MarkdownGenerator
from ctransformers import LLM, AutoModelForCausalLM

generator = MarkdownGenerator()
docs = ''
for class_ in (AutoModelForCausalLM, LLM):
    docs += generator.class2md(class_, depth=3)
docs += '---\n' + generator.func2md(LLM.__call__, clsname='LLM', depth=4)

README = ROOT / 'README.md'
MARKER = '<!-- API_DOCS -->'

with open(README) as f:
    contents = f.read()

parts = contents.split(MARKER)
if len(parts) != 3:
    raise RuntimeError(
        f"README should have exactly 2 '{MARKER}' but has {len(parts) - 1}.")
parts[1] = docs
contents = MARKER.join(parts)

with open(README, mode='w') as f:
    f.write(contents)
