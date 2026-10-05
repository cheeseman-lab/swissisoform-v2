"""Materialize one arm's prompt directory by splicing the tracked base prompts.

Eight arms x three base prompts would be 24 hand-forked files that diverge on the
first prose fix — and the three already differ in ways that matter:
``category-pass-M.txt`` redefines what "neutral" means and adds a consistency
check the other two lack. Editing one and not the others would change the verdict
bar for four categories but not M, a confound *inside* a single arm.

So the base files stay single-source and carry HTML-comment markers around the
blocks an arm needs to vary:

``input_contract``
    The "you receive" paragraph. It must be swapped, not merely kept — it promises
    "pre-computed values, reason strings, raw metrics and interpretation_hints",
    none of which exist under ``raw`` / ``tags`` / ``dist``. Leaving it is a broken
    prompt, and would surface as degraded quality wrongly blamed on grounding.
``directionality``
    The system-level reading rules (signal directions, the P2 confidence gate).
    Kept in **every** arm, hint or not: dropping it in ``-hint`` made the hint axis
    two changes at once — the per-member hints *and* the gate — so a hint effect
    could not be told from a gate effect. The ``-hint`` arms now vary only the
    hint content itself (per-member ``interpretation_hint`` / per-tag ``means``).
``roster`` / ``machinery`` / ``id_examples`` / ``id_example`` (shared prompt only)
    The CDLMPS member roster and the writing rules that name member ids and
    thresholds. ``raw`` and ``dist`` payloads carry neither ids nor thresholds, so
    for them these — and ``directionality`` — are swapped for id- and
    threshold-free wordings of the same rules. Left as-is they taught those arms
    to write "C1/C2/C3" and cite thresholds the payload never showed (audit38_C).

Markers are stripped in **every** arm, so the ``+hint`` `criteria` assembly is the
base file verbatim. ``tests/test_prompt_variants.py`` asserts exactly that.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

BLOCK_RE = re.compile(r"^<!-- @block:([a-z_]+) -->$")
END_RE = re.compile(r"^<!-- @end -->$")

BASE_PROMPTS: tuple[str, ...] = (
    "category-pass.txt",
    "category-pass-M.txt",
    "category-pass-P.txt",
)
SECTIONS: tuple[str, ...] = ("input_contract", "directionality", "judgment_tags")
# Declared by the shared prompt only; required there when an id-free arm renders it.
ID_SECTIONS: tuple[str, ...] = ("roster", "machinery", "id_examples", "id_example")
# Groundings whose payload names no member ids and states no thresholds.
ID_FREE_GROUNDINGS: frozenset[str] = frozenset({"raw", "dist"})

# The shared prompt's id-bearing blocks, reworded for ID_FREE_GROUNDINGS. Same
# rules, same order; only the member ids, the scorer/reason-string references and
# the thresholds are gone.
ID_FREE: dict[str, str] = {
    "roster": (
        "The six categories (CDLMPS) and what each one asks:\n"
        "- C Conservation: amino-acid identity across primates and across mammals, and "
        "absolute phyloP constraint over the unique region.\n"
        "- D Detection: reproducibility across cell lines, initiation efficiency "
        "(start-site usage), and unique-peptide mass spectrometry.\n"
        "- L Localization: DeepLoc compartment/signal change, and SignalP/TargetP "
        "targeting change.\n"
        "- M Mutation Landscape: germline tolerance/constraint, and disease-variant "
        "density enrichment.\n"
        "- P Predicted Structure: unique-region folding (pLDDT), shared-region core-fold "
        "perturbation (Cα RMSD), and secondary structure — a helix or strand gained "
        "(extension) or lost (truncation) in the differential region.\n"
        "- S Structural Characteristics: real-InterPro-domain gain/loss, whole-protein "
        "biophysical shift (isoform vs canonical gravy/charge/disorder), and differential "
        "ESM-C sparse-autoencoder interpretability features (the size of the strongest "
        "shared-feature activation shift; gained/lost counts are context only)."
    ),
    "directionality": (
        "Directionality — get these right:\n"
        "- The two germline-constraint signals are INDEPENDENT, either-or evidence, not "
        "two measurements of one quantity, and they run in OPPOSITE numerical directions "
        "for the same conclusion (gnomAD: lower = constrained; ESM-C constraint_delta: "
        "higher = constrained). Either one alone can carry the signal. Do not expect them "
        "to corroborate, and do not treat one being quiet as undercutting the other.\n"
        "- Germline TOLERANCE/CONSTRAINT: a gnomAD depletion ratio < 1 means population "
        "variation AVOIDS the unique region (evidence it is important); a positive ESM-C "
        "constraint_delta means the model predicts the unique region's residues better "
        "than the shared core, i.e. finds them more conserved (it is mean logP(wt) unique "
        "minus shared). Never read gnomAD variants as disease/pathogenic burden.\n"
        "- DISEASE density: a ClinVar/COSMIC enrichment ratio >= 1 means disease variants "
        "CONCENTRATE in the unique region.\n"
        "- Unique-region pLDDT measures folding only; the biophysical signal "
        "(whole-protein gravy/charge/disorder delta, isoform vs canonical) is a separate "
        "question in S. Domain gain/loss counts only real InterPro domains (not "
        "disorder/coiled-coil/low-complexity).\n"
        "- A shared-region core-fold change (Cα RMSD) is only meaningful when the shared "
        "region is confidently folded in BOTH structures — read it against the "
        "shared-region pLDDT. Never call an elevated RMSD a real refold when that "
        "confidence is low; a floppy, low-pLDDT loop moves the backbone without any "
        "conformational meaning. Weigh the RMSD against its pLDDT before treating it as "
        "signal.\n"
        "- Primate/mammalian conservation is amino-acid identity; phyloP is absolute "
        "constraint, larger |mean| = stronger.\n"
        "- For the ESM-C sparse-autoencoder features, the MAGNITUDE of the strongest "
        "shared-feature activation shift is the signal, not presence. Gained/lost feature "
        "counts are context only — two proteins of different length always differ in "
        "hundreds of features, so a large count is not evidence of anything. Feature "
        "labels are auto-generated and provisional, not curated annotation: many carry no "
        "content at all, and many others merely describe the feature's own activation "
        "pattern or restate sequence composition the biophysics already covers. Never "
        "name, quote or interpret an SAE feature label in the reasoning, and never let a "
        "label move the verdict — cite the activation-shift magnitude, not the labels. "
        "Never invent biology the inputs do not state."
    ),
    "machinery": (
        "Cite the measurement, not a judgement about it. Never report that a value "
        "cleared, exceeded, met or fell short of a standard, however you phrase it: "
        '"well above the level treated as notable" and "comfortably in range" are the '
        "same move, and naming no number does not make it one. State what was measured "
        'and what it came to, and let the size speak for itself — "the strongest '
        'shared-feature activation shifts by 13.7".'
    ),
    "id_examples": (
        '- Name each measurement by what it measures ("primate amino-acid identity", '
        '"the unique-peptide mass-spec check", "whole-protein hydropathy/charge/disorder '
        'shift", "the shared-core fold RMSD"). Give the two or three numbers that carry '
        "the read — unusually large or small, or the one that tempers a claim."
    ),
    "id_example": (
        'Example — Bad, reads as an inventory: "Conservation is high across the board: '
        'primate identity 97.7%, mammalian 93.4%, phyloP 3.4 over the region." Good, the '
        'same evidence as interplay: "The removed segment is under purifying selection: '
        "primate and mammalian identity agree (97.7% and 93.4%), so this is not one "
        "clade's artefact, and phyloP of 3.4 confirms it at the nucleotide level. Its "
        'loss is likely consequential."'
    ),
}
SCHEMA_REL = Path("output_schemas") / "category_read.json"
# Read by llm._tool_categories when present; the loop validates against this
# instead of the shared file. Must match llm.TOOL_VERDICT_SCHEMA.
TOOL_SCHEMA_REL = Path("output_schemas") / "category_read_tools.json"

# What each grounding's payload actually contains, in the model's own terms. The
# identity sentence is shared because every arm carries the same identity block.
_IDENTITY = (
    "You receive: (1) the isoform's identity (gene, transcript, ORF type, lengths, "
    "differential sequence, and `differential_region_location` — an explicit statement "
    "of where the differential region sits; trust it verbatim over any prior about "
    "this gene); (2) the category name and letter; and (3) "
)

# The shared prompt's opening role sentence sits inside its input_contract block
# (the M/P prompts keep theirs outside), so a swapped contract must restore it or
# the alternative arms lose the one line saying what the task is.
_ROLE = (
    "You are interpreting ONE evidence CATEGORY for an alternative protein isoform (an "
    "alt-TIS isoform: an N-terminal extension, a truncation, or a uORF/altORF) in "
    "SwissIsoform v2. "
)

CONTRACTS: dict[str, str] = {
    "raw": (
        _IDENTITY + "`evidence` — every metric this pipeline computes for the category, "
        "as raw column names and values, with NO pre-computed verdicts and no guidance "
        "about which ones matter. Some are uninformative or redundant; deciding which "
        "carry signal is your job. A `truncated` or `hits_note` block, when present, "
        "states exactly how many rows were withheld and why."
    ),
    "tags": (
        _IDENTITY + "`tags` — a controlled vocabulary of binary findings, each already "
        "evaluated against a frozen cutoff. `state` is `on` (the test passed), `off` "
        "(it was tested and failed) or `not_evaluable` (it could not be tested at all, "
        "which is NOT the same as off). Every tag carries the single `value` its `test` "
        "cut on. A tag that restates a scored criterion additionally carries `metrics` "
        "— the full set of supporting numbers behind that criterion, of which `value` "
        "is only the one the cutoff reads — and a `note` where its definition has a "
        "caveat. A tag with no `metrics` is single-metric by construction: its `value` "
        "is the whole of its evidence, not an excerpt. There are no verdict strings and "
        "no row-level records: reason from the states, their numbers, and the "
        "supporting metrics where they are given. The `state`, `test` and `cutoff` are how the payload was built, not findings — read them, never cite them."
    ),
    "dist": (
        _IDENTITY + "`fields` — every numeric metric for the category with its `value` "
        "and `pctile`, its rank against a frozen reference population of isoforms, plus "
        "that population's five-number summary (p05/p25/p50/p75/p95). There are no "
        "verdicts and no thresholds: a value's importance is what its position in the "
        "distribution tells you. `stratum` names which population it was ranked "
        "against, and `reference_population` says what that population is."
    ),
}

# The base prompts refer to the payload as `members` in places outside the marked
# block — the task sentence, the all-None verdict rule, the Bad/Good example. They
# are not contiguous, so marking each would mean four more markers per file to keep
# in sync. One terminology line in the contract resolves every downstream reference
# at once, including the decision rule ("when the members are all None / not
# evaluable, the verdict is neutral") that an alternative arm genuinely needs an
# equivalent of. The CDLMPS roster above it stays untouched: it describes what the
# categories are *about*, which is true regardless of how the payload is shaped.
NOUNS: dict[str, str] = {
    "raw": "the metrics in `evidence`",
    "tags": "the tags in `tags`",
    "dist": "the fields in `fields`",
}
# Appended in the `+hint` arms only. `means` is the criterion's own
# interpretation_hint, so leaving it unconditional would give `tags_nohint` the
# guidance the hint axis exists to remove.
_HINT_CLAUSE: dict[str, str] = {
    "tags": (
        " Where a tag carries `means`, that states what the tag is asking and which "
        "of its `metrics` the call rests on; read it before weighing the numbers."
    ),
}
_TERMINOLOGY = (
    ' Terminology: where the rest of this prompt says "members" or "submodules", '
    'read {noun}. Where it says a member is "None / not evaluable", read {na}.'
)
_NA = {
    "raw": "a metric that is null",
    "tags": "a tag whose state is `not_evaluable`",
    "dist": "a field absent from `fields`",
}


def _union_extras(per_category: dict[str, dict]) -> dict:
    """Merge per-category verdict extras, unioning any shared enum.

    ``_verdict_violations`` validates a tool-loop payload against the one shared
    ``category_read.json``, so a narrower enum there would reject a legitimate
    firing from whichever category lost the merge.
    """
    merged: dict = {}
    for fields in per_category.values():
        for key, schema in fields.items():
            if key not in merged:
                merged[key] = json.loads(json.dumps(schema))
                continue
            have = merged[key].get("items", {}).get("enum")
            add = schema.get("items", {}).get("enum")
            if have is not None and add is not None:
                merged[key]["items"]["enum"] = sorted(set(have) | set(add))
    return merged


# The judgment-tag instruction, per (grounding, file). Only the `tags` grounding
# has judgment tags at all; every other arm gets the block deleted.
#
# Both halves are load-bearing. Without the M/P half nothing asks for
# `tags_fired`, and the smoke test showed all four tool loops omitting it — the
# field is optional by design, so an unasked-for optional field is simply never
# emitted. Without the single-shot half, C/D/L/S see `tags_fired` in the shared
# output schema with no idea it is not theirs, and fill it.
JUDGMENT_TAGS: dict[str, str] = {
    "tools": (
        "The payload's `open_questions` are judgment tags: findings no cutoff can "
        "express, which is why they are asked of you rather than computed. Each names "
        "the `reader` tool that answers it and the `citation` it must rest on. Decide "
        "each one from the data you actually read, then pass the ids that fired as "
        "`tags_fired` on your `emit_verdict` call and cite each one's number in "
        "`reasoning`. **Always include `tags_fired`, even when it is empty** — send "
        "`[]` to say you judged them and none fired. Omitting it is not the same "
        "answer, and is not an available one. Never name a tag that is not in "
        "`open_questions`."
    ),
    "single_shot": (
        "This category has no judgment tags: `tags_fired` does not apply to it. Leave "
        "it out of your response entirely — a tag id from another category is not a "
        "finding about this one."
    ),
}


class AssembleError(RuntimeError):
    """Raised when a base prompt's markers are missing or malformed."""


def parse_blocks(text: str) -> dict[str, tuple[int, int]]:
    """``{section: (start, end)}`` line spans, exclusive of the marker lines.

    Raises:
        AssembleError: A marker is unclosed, duplicated, or unknown.
    """
    spans: dict[str, tuple[int, int]] = {}
    open_name: str | None = None
    open_at = 0
    for i, line in enumerate(text.splitlines()):
        opened = BLOCK_RE.match(line.strip())
        if opened:
            if open_name is not None:
                raise AssembleError(f"{open_name!r} not closed before {opened.group(1)!r}")
            open_name, open_at = opened.group(1), i + 1
            if open_name in spans:
                raise AssembleError(f"section {open_name!r} declared twice")
            continue
        if END_RE.match(line.strip()):
            if open_name is None:
                raise AssembleError("@end with no open @block")
            spans[open_name] = (open_at, i)
            open_name = None
    if open_name is not None:
        raise AssembleError(f"section {open_name!r} is never closed")
    return spans


def render(text: str, *, grounding: str, hints: bool, is_tool_prompt: bool = False) -> str:
    """One base prompt, spliced for an arm and with every marker removed.

    Replacement is keyed by *section*, not by line index: a block can legitimately
    be empty in the base file (``judgment_tags`` is), and an index-keyed insert
    would land on the ``@end`` marker — which is stripped before replacements are
    consulted, so the content silently vanished.

    ``hints`` reaches only the contract's hint clause: the Directionality block is
    kept either way, so the hint axis varies the hint content and nothing else.
    """
    spans = parse_blocks(text)
    id_free = grounding in ID_FREE_GROUNDINGS and not is_tool_prompt
    required = SECTIONS + (ID_SECTIONS if id_free else ())
    missing = [s for s in required if s not in spans]
    if missing:
        raise AssembleError(f"base prompt is missing section(s): {', '.join(missing)}")

    # section -> replacement lines, or None to delete the block's body entirely.
    edits: dict[str, list[str] | None] = {}
    if grounding in CONTRACTS:
        contract = (
            ("" if is_tool_prompt else _ROLE)
            + CONTRACTS[grounding]
            + (_HINT_CLAUSE.get(grounding, "") if hints else "")
        )
        edits["input_contract"] = [
            contract + _TERMINOLOGY.format(noun=NOUNS[grounding], na=_NA[grounding])
        ]
    if id_free:
        for section in ("directionality", *ID_SECTIONS):
            edits[section] = ID_FREE[section].split("\n")
    edits["judgment_tags"] = (
        [JUDGMENT_TAGS["tools" if is_tool_prompt else "single_shot"], ""]
        if grounding == "tags"
        else None
    )

    body_of: dict[int, str] = {}
    for name, (start, end) in spans.items():
        for i in range(start, end):
            body_of[i] = name

    out: list[str] = []
    for i, line in enumerate(text.splitlines()):
        stripped = line.strip()
        opened = BLOCK_RE.match(stripped)
        if opened:
            replacement = edits.get(opened.group(1), "keep")
            if replacement not in ("keep", None):
                out.extend(replacement)
            continue
        if END_RE.match(stripped):
            continue
        section = body_of.get(i)
        if section is not None and edits.get(section, "keep") != "keep":
            continue  # body replaced at the opening marker, or deleted
        out.append(line)
    return _collapse_blank_runs(out)


def _collapse_blank_runs(lines: list[str]) -> str:
    """Join, squeezing runs of blank lines a deletion may have left behind."""
    kept: list[str] = []
    for line in lines:
        if not line.strip() and kept and not kept[-1].strip():
            continue
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept) + "\n"


def materialize(
    dest: Path,
    *,
    grounding: str,
    hints: bool,
    base_dir: Path,
    verdict_extras: dict[str, dict] | None = None,
) -> Path:
    """Write one arm's prompt root and return it.

    ``verdict_extras`` maps a category letter to extra ``category_read.json``
    properties — the ``tags`` arms use it for ``tags_fired``. The schema is
    per-arm rather than global because ``llm.main`` already resolves it from
    ``prompts_dir``, so varying it costs nothing.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for name in BASE_PROMPTS:
        source = base_dir / name
        if not source.exists():
            raise AssembleError(f"missing base prompt {source}")
        (dest / name).write_text(
            render(
                source.read_text(encoding="utf-8"),
                grounding=grounding,
                hints=hints,
                # M and P emit through a tool loop; the base file serves C/D/L/S.
                is_tool_prompt=name != "category-pass.txt",
            ),
            encoding="utf-8",
        )

    (dest / SCHEMA_REL.parent).mkdir(parents=True, exist_ok=True)
    schema = json.loads((base_dir / SCHEMA_REL).read_text(encoding="utf-8"))
    # Two schemas, because they answer different questions. The shared
    # category_read.json drives *constrained decoding* on the single-shot path, so
    # anything declared there is a slot C/D/L/S will fill — measured twice on the
    # smoke test, Conservation emitting an M tag it was never shown. Relaxing
    # additionalProperties instead made it worse: an open door rather than a
    # labelled slot. So the shared file stays closed and silent about tags_fired,
    # and the tool loop — which only has its payload jsonschema-checked afterwards,
    # never decoded against — gets its own.
    extras = _union_extras(verdict_extras or {})
    (dest / SCHEMA_REL).write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    if extras:
        tool_schema = json.loads(json.dumps(schema))
        tool_schema["properties"].update(extras)
        (dest / TOOL_SCHEMA_REL).write_text(
            json.dumps(tool_schema, indent=2) + "\n", encoding="utf-8"
        )

    for extra in (base_dir / "output_schemas").glob("*.json"):
        if extra.name != SCHEMA_REL.name:
            shutil.copyfile(extra, dest / "output_schemas" / extra.name)
    return dest


__all__ = [
    "BASE_PROMPTS",
    "CONTRACTS",
    "ID_FREE",
    "ID_FREE_GROUNDINGS",
    "ID_SECTIONS",
    "JUDGMENT_TAGS",
    "NOUNS",
    "SECTIONS",
    "TOOL_SCHEMA_REL",
    "AssembleError",
    "materialize",
    "parse_blocks",
    "render",
]
