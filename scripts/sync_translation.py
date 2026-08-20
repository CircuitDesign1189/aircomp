# Copyright (c) 2026 Circuit Design, Inc. and Y.Chosokabe
# SPDX-License-Identifier: Apache-2.0

"""Keep CLAUDE.ja.md in step with CLAUDE.md.

English is the source of truth. The translation records the SHA-256 of the exact
English bytes it was made from, so drift is detectable rather than a thing
someone notices months later.

    python scripts/sync_translation.py            # report status, exit 1 if stale
    python scripts/sync_translation.py --update   # stamp the current hash

`--update` only rewrites the hash line. It cannot translate, and it deliberately
refuses when the section structure of the two files disagrees -- that mismatch
means new English text exists with no Japanese counterpart, and stamping the
hash then would hide exactly the problem this file is here to surface.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "CLAUDE.md"
TRANSLATION = ROOT / "CLAUDE.ja.md"

HASH_LINE = re.compile(r"^<!-- source-sha256: ([0-9a-f]{64}) -->$", re.MULTILINE)
#: Headings inside fenced code blocks are content, not structure -- the repo
#: layout tree is full of '#' comments.
FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
HEADING = re.compile(r"^(#{1,6}) ", re.MULTILINE)


def source_hash() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def recorded_hash() -> str | None:
    match = HASH_LINE.search(TRANSLATION.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def heading_levels(text: str) -> list:
    """The heading skeleton: levels only, not titles -- titles are translated."""
    return [len(m.group(1)) for m in HEADING.finditer(FENCE.sub("", text))]


def structure_problems() -> list:
    """Differences that mean the translation is missing or has extra content."""
    src = SOURCE.read_text(encoding="utf-8")
    dst = TRANSLATION.read_text(encoding="utf-8")
    problems = []

    src_headings = heading_levels(src)
    # The translation carries one extra H1 for its own title/disclaimer block.
    dst_headings = heading_levels(dst)
    if dst_headings[:1] == [1] and src_headings[:1] == [1]:
        dst_headings = dst_headings[1:]
        src_headings = src_headings[1:]
    if src_headings != dst_headings:
        problems.append(
            f"heading structure differs: CLAUDE.md has {src_headings}, "
            f"CLAUDE.ja.md has {dst_headings} (excluding each file's title)"
        )

    src_fences, dst_fences = len(FENCE.findall(src)), len(FENCE.findall(dst))
    if src_fences != dst_fences:
        problems.append(f"code blocks: CLAUDE.md has {src_fences}, CLAUDE.ja.md has {dst_fences}")

    return problems


def run_as_hook() -> int:
    """Claude Code PostToolUse hook: complain the moment either file is edited.

    The test in tests/ is the backstop; this is the fast path, so the reminder
    arrives while the edit is still in mind rather than at the next test run.
    Silent unless one of the two files was touched AND they are now out of step.

    Exit 2 puts the message in front of the model. Anything unexpected -- no
    stdin, malformed payload, missing files -- exits 0: a broken hook must never
    block editing.
    """
    import json

    try:
        payload = json.load(sys.stdin)
        edited = Path(payload.get("tool_input", {}).get("file_path", "")).name
    except Exception:
        return 0

    # Matched by filename, not by resolved path: the hook payload's path spelling
    # varies (drive-letter case, POSIX-style paths under Git Bash) and a
    # mismatch there would silently disarm the hook. A same-named file elsewhere
    # just re-runs a check that costs nothing.
    if edited not in (SOURCE.name, TRANSLATION.name):
        return 0
    try:
        if source_hash() == recorded_hash() and not structure_problems():
            return 0
    except Exception:
        return 0

    print(
        f"{SOURCE.name} and {TRANSLATION.name} are now out of step. Translate the changed "
        f"passages into {TRANSLATION.name}, then run: python scripts/sync_translation.py --update",
        file=sys.stderr,
    )
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true", help="stamp the current hash after translating")
    parser.add_argument("--hook", action="store_true", help="run as a Claude Code PostToolUse hook (reads stdin)")
    args = parser.parse_args()

    if args.hook:
        return run_as_hook()

    current, recorded = source_hash(), recorded_hash()
    if recorded is None:
        print(f"{TRANSLATION.name} has no '<!-- source-sha256: ... -->' line")
        return 1

    problems = structure_problems()

    if args.update:
        if problems:
            print("refusing to stamp -- the files do not have the same shape:")
            for p in problems:
                print(f"  - {p}")
            print("\nTranslate the new sections first; the hash is not the point, the translation is.")
            return 1
        TRANSLATION.write_text(
            HASH_LINE.sub(f"<!-- source-sha256: {current} -->", TRANSLATION.read_text(encoding="utf-8"), count=1),
            encoding="utf-8",
        )
        print(f"stamped {TRANSLATION.name} with {current[:12]}...")
        return 0

    if problems:
        for p in problems:
            print(f"STRUCTURE: {p}")
    if current != recorded:
        print(f"STALE: {SOURCE.name} is {current[:12]}... but {TRANSLATION.name} records {recorded[:12]}...")
    if problems or current != recorded:
        print("\nTranslate the changed passages, then run: python scripts/sync_translation.py --update")
        return 1

    print(f"{TRANSLATION.name} is in step with {SOURCE.name} ({current[:12]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
