"""CLAUDE.ja.md must not drift away from CLAUDE.md.

A translation with no enforcement rots: the English gets edited, the Japanese
quietly describes an older repo, and the next reader trusts it. Since
`pytest -m "not slow" -q` is run constantly here, putting the check in the suite
means drift surfaces within one cycle rather than months later.

The hash alone would be too weak -- it can be silenced by stamping without
translating -- so the structural checks in scripts/sync_translation.py run too.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from sync_translation import (  # noqa: E402
    SOURCE,
    TRANSLATION,
    heading_levels,
    recorded_hash,
    source_hash,
    structure_problems,
)


def test_the_translation_records_the_english_it_was_made_from():
    assert recorded_hash() is not None, (
        f"{TRANSLATION.name} needs a '<!-- source-sha256: ... -->' line so staleness is detectable"
    )


def test_the_translation_is_not_stale():
    assert source_hash() == recorded_hash(), (
        f"{SOURCE.name} changed since {TRANSLATION.name} was written.\n"
        f"Translate the changed passages, then run:\n"
        f"    python scripts/sync_translation.py --update"
    )


def test_the_two_files_have_the_same_shape():
    """Catches the failure the hash cannot: new English sections stamped but
    never translated."""
    problems = structure_problems()

    assert not problems, "\n".join(problems)


def test_the_translation_says_it_is_not_authoritative():
    """A reader who lands on the translation first must know which file wins."""
    text = TRANSLATION.read_text(encoding="utf-8")

    assert "CLAUDE.md" in text.splitlines()[0] or "translation-of: CLAUDE.md" in text
    assert "英語版が正しい" in text


def test_the_english_points_at_the_translation():
    """Otherwise someone edits CLAUDE.md without ever learning the other file exists."""
    assert "CLAUDE.ja.md" in SOURCE.read_text(encoding="utf-8")


def test_headings_inside_code_blocks_are_not_counted_as_structure():
    """The repo-layout tree is full of '#' comments; treating them as headings
    would make the structural check fire on every unrelated edit."""
    text = "# Title\n\n```\nfoo.py   # not a heading\n```\n\n## Real\n"

    assert heading_levels(text) == [1, 2]


@pytest.mark.parametrize("path", [SOURCE, TRANSLATION])
def test_both_files_exist_and_are_utf8(path):
    assert path.read_text(encoding="utf-8").strip()
