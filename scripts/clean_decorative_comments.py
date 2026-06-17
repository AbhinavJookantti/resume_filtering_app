#!/usr/bin/env python3
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [p for p in ROOT.glob('*.py') if p.name != 'scripts']

def clean_line(line: str) -> str | None:
    if not line.lstrip().startswith('#'):
        return line
    # strip leading '#'
    content = line.lstrip()[1:]
    # remove surrounding whitespace
    content = content.strip()
    # remove leading/trailing decorative chars (dashes, box drawings, spaces)
    content = re.sub(r'^[\W_\-\u2500\u2501\u2502\u2503\ufe31\uff0d\s]+', '', content)
    content = re.sub(r'[\W_\-\u2500\u2501\u2502\u2503\ufe31\uff0d\s]+$', '', content)
    # If nothing remains, remove the line
    if not content:
        return None
    # If it looks like a short header with multiple words, keep it
    return '# ' + content + '\n'

for file in FILES:
    p = Path(file)
    text = p.read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)
    changed = False
    new_lines = []
    for ln in lines:
        cleaned = clean_line(ln)
        if cleaned is None:
            # remove decorative-only comment
            if ln.strip():
                changed = True
            continue
        if cleaned != ln:
            changed = True
        new_lines.append(cleaned)
    if changed:
        print(f'Updating: {p}')
        p.write_text(''.join(new_lines), encoding='utf-8')
