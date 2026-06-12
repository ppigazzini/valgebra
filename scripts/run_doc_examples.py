"""Execute every runnable Python example in the documentation.

Each fenced ```python block in README.md is run as its own process, so module
semantics (class definitions, get_type_hints) behave exactly as they would for
a reader who copies the snippet. Blocks marked PLANNED (target APIs that do not
run yet) are skipped. Exit non-zero if any example fails, so CI catches a stale
or broken example.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "error-model.md",
    ROOT / "docs" / "json.md",
]
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def run_block(block: str, doc_name: str, index: int) -> bool:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "example.py"
        path.write_text(block, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        print(f"{doc_name} block {index} failed:\n{proc.stderr}")
        return False
    return True


def main() -> int:
    checked = 0
    failures = 0
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for index, block in enumerate(BLOCK.findall(text), start=1):
            if "PLANNED" in block:
                continue
            checked += 1
            if not run_block(block, doc.name, index):
                failures += 1
    print(f"checked {checked} example(s), {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
