"""The isoform-change tag vocabulary must stay identical across its 3 sources.

The controlled tag vocabulary is duplicated in three places that must agree:

1. ``ISOFORM_TAG_VOCAB`` in ``website/src/swissisoform_site/data.py`` — the
   website mirror; synthesis tags are whitelisted against this on read.
2. ``properties.tags.items.enum`` in
   ``scripts/site/prompts/output_schemas/synthesis.json`` — the output schema
   that constrains the synthesis LLM's ``tags`` output.
3. The ``Controlled tag vocabulary`` bullets in
   ``scripts/site/prompts/synthesis-pass.txt`` — the prose the model reads.

Nothing enforces the sync at runtime (the website silently drops off-vocab tags,
and the prompt/schema are plain files), so this test fails loudly if a future
edit adds or removes a tag in one place and forgets the others.

Fragility note: the prompt extractor is coupled to the prompt's format — it
anchors on the ``Controlled tag vocabulary`` header line and the
``- "Tag" — ...`` bullet shape. If that section is reformatted, update
``_prompt_tags`` deliberately (do not delete this test).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("swissisoform_site")  # optional website package; skip if absent

from swissisoform_site.data import ISOFORM_TAG_VOCAB

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "scripts" / "site" / "prompts" / "output_schemas" / "synthesis.json"
PROMPT_PATH = REPO_ROOT / "scripts" / "site" / "prompts" / "synthesis-pass.txt"

# Matches a controlled-vocabulary bullet: `  - "Tag name" — description`.
_BULLET_RE = re.compile(r'^\s*-\s*"([^"]+)"\s*—')  # — = em dash


def _schema_enum() -> list[str]:
    """The tag enum from the synthesis output schema (ordered)."""
    data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        enum = data["properties"]["tags"]["items"]["enum"]
    except (KeyError, TypeError) as exc:  # schema shape changed
        raise AssertionError(
            f"synthesis.json no longer has properties.tags.items.enum: {exc}"
        ) from exc
    assert isinstance(enum, list) and all(isinstance(t, str) for t in enum)
    return enum


def _prompt_tags() -> list[str]:
    """Tags parsed from the 'Controlled tag vocabulary' bullet block (ordered)."""
    lines = PROMPT_PATH.read_text(encoding="utf-8").splitlines()
    header = next(
        (i for i, ln in enumerate(lines) if "Controlled tag vocabulary" in ln),
        None,
    )
    assert header is not None, (
        "synthesis-pass.txt has no 'Controlled tag vocabulary' header — "
        "the prompt was reformatted; update this extractor."
    )
    tags: list[str] = []
    for ln in lines[header + 1 :]:
        m = _BULLET_RE.match(ln)
        if m:
            tags.append(m.group(1))
        elif tags:
            break  # first non-bullet line after the block ends it
    return tags


def test_tag_vocab_sources_are_nonempty() -> None:
    """Guard against a silent empty parse trivially passing set-equality below."""
    assert ISOFORM_TAG_VOCAB, "ISOFORM_TAG_VOCAB is empty"
    assert _schema_enum(), "synthesis.json tag enum is empty"
    assert _prompt_tags(), "no tags parsed from synthesis-pass.txt"


def test_tag_vocab_sets_match() -> None:
    """Adds/removes: the three vocabularies must contain exactly the same tags."""
    data_set = set(ISOFORM_TAG_VOCAB)
    schema_set = set(_schema_enum())
    prompt_set = set(_prompt_tags())
    assert data_set == schema_set, (
        "data.py vs synthesis.json tag mismatch — "
        f"only in data.py: {sorted(data_set - schema_set)}; "
        f"only in schema: {sorted(schema_set - data_set)}"
    )
    assert data_set == prompt_set, (
        "data.py vs synthesis-pass.txt tag mismatch — "
        f"only in data.py: {sorted(data_set - prompt_set)}; "
        f"only in prompt: {sorted(prompt_set - data_set)}"
    )


def test_tag_vocab_ordered_artifacts_align() -> None:
    """The two ordered machine artifacts must also agree on order (display order).

    data.py display order is meaningful and currently mirrors the schema enum;
    the prompt bullets are prose, so their order is checked only as a set (above).
    """
    assert list(ISOFORM_TAG_VOCAB) == _schema_enum()
