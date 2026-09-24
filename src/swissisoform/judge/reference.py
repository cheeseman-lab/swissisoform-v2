"""The shared evidence every arm in a cell is judged against.

Each arm saw a *different* payload -- that is the variable under test -- so an
arm's own payload cannot be the yardstick: an arm shown little would score full
marks for reasoning perfectly over little. The judge is instead given one
reference per cell, and an arm that reaches a well-supported read from less
input scores higher, not equal.

The reference is ``criteria + tags + dist``, measured rather than chosen:

    mean input tokens/call   criteria 5,097   tags 3,660   dist 7,104   raw 9,522

criteria+tags+dist is ~16k tokens (~13k after the shared identity block is
deduplicated). Adding ``raw`` reaches ~25k, and against Prometheus's 32k context
that leaves no room for two responses plus a rubric in a pairwise call. ``raw`` is
the one to drop: criteria's curated ``evidence_cols`` already carries the
underlying numbers, so what ``raw`` adds is breadth, not depth. The cost is stated
as a confound -- the ``raw`` arms are judged against evidence shaped less like
their own input than the other six are.

Nothing here is imported by ``site/evidence.py``: that module is staged into the
website deploy with only ``__init__.py`` + ``config.py`` for company
(``website/prepare_deploy.sh:43-49``), so a heavy import there breaks the deploy
at import time on the live site.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from swissisoform.judge import CATEGORY_NAMES, SYNTHESIS_UNIT
from swissisoform.site import evidence as ev
from swissisoform.site import grounding
from swissisoform.site.llm import _tis_slug as tis_slug

# Keys `slice_category` wraps around a builder's body. Dropped from the merged
# reference so the three bodies do not each restate the identity block.
_IDENTITY_KEYS = ("category", "name", "isoform")


class ReferenceError(RuntimeError):
    """A cell's reference payload could not be built."""


@dataclass
class ReferenceBuilder:
    """Builds one cell's reference payload.

    Holds the three grounding bodies so the catalog, tag registry and frozen
    distributions are each read once rather than per cell -- ``_dist_body``'s
    metric lookup is an O(n) scan over 4,864 rows, which is cheap once and
    ruinous 2,100 times.
    """

    tags_body: Any
    dist_body: Any
    dist_version: str
    tag_version: str

    @classmethod
    def build(
        cls,
        *,
        dist_version: str = grounding.DEFAULT_DIST_VERSION,
        tag_version: str = grounding.DEFAULT_TAG_VERSION,
    ) -> ReferenceBuilder:
        """Construct the builder, loading each reference input once."""
        return cls(
            tags_body=grounding.build("tags", tag_version=tag_version),
            dist_body=grounding.build("dist", dist_version=dist_version),
            dist_version=dist_version,
            tag_version=tag_version,
        )

    def category(self, record: dict[str, Any], letter: str) -> dict[str, Any]:
        """The reference for one ``(isoform, category)`` cell.

        Merges the three groundings under distinct keys so the judge can tell a
        scorer's verdict (``criteria``) from a fired tag (``tags``) from a
        population percentile (``distribution``) -- collapsing them would let the
        judge treat one number as three pieces of corroborating evidence.
        """
        category = _category_spec(letter)
        criteria = ev.slice_category(record, category)
        members, shared = _dedupe_hits(criteria.get("members", []))
        merged: dict[str, Any] = {
            "category": criteria.get("category"),
            "name": criteria.get("name"),
            "isoform": criteria.get("isoform"),
            "criteria": members,
        }
        if shared:
            merged["shared_hit_tables"] = shared
        for key, body in (("tags", self.tags_body), ("distribution", self.dist_body)):
            if body is None:
                continue
            part = body(record, category)
            merged[key] = {k: v for k, v in part.items() if k not in _IDENTITY_KEYS}
        return merged

    def synthesis(self, record: dict[str, Any], category_reads: dict[str, Any]) -> dict[str, Any]:
        """The reference for a synthesis cell.

        Synthesis is judged on coherence with **its own** inputs, so the
        reference is that arm's six category reads plus the criteria evidence.
        Verified on the corpus: across arms ``category_reads`` is the only key
        that differs, while ``criteria_evidence`` (95,316 chars), ``gene``,
        ``isoform``, ``key_metrics``, ``localization``, ``scoring``, the
        11,487-char system prompt and the output schema are byte-identical. So
        this unit measures downstream propagation with the framing held constant.

        ``criteria_evidence`` keeps its scalars and reasons but drops the
        individual hit rows, which are 60% of it (95,884 -> 38,438 chars on a
        median isoform; M1 and M2 alone fall from 30,750 and 29,743 chars to
        2,411 and 1,404). Unstripped, the worst synthesis reference is ~50k tokens
        against a 32k context, so this is not a preference. Each member keeps
        ``n_hits_total``, so "there are rows you are not seeing" stays visible,
        and synthesis prose cites scalars and reasons rather than single variants.
        """
        return {
            "isoform": {
                k: record.get(k)
                for k in (
                    "tis_id",
                    "orf_type",
                    "isoform_length_aa",
                    "canonical_length_aa",
                    "diff_space",
                )
            },
            "category_reads": category_reads,
            "criteria_evidence": [
                _without_hit_rows(ev.slice_criterion(record, cid)) for cid in sorted(ev.CRITERIA)
            ],
        }

    def for_cell(
        self,
        record: dict[str, Any],
        unit: str,
        *,
        category_reads: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dispatch to the category or synthesis reference."""
        if unit == SYNTHESIS_UNIT:
            if category_reads is None:
                raise ReferenceError(
                    "the synthesis reference needs that arm's category_reads; "
                    "synthesis is judged against its own inputs"
                )
            return self.synthesis(record, category_reads)
        return self.category(record, unit)


def _without_hit_rows(slice_: dict[str, Any]) -> dict[str, Any]:
    """Replace every populated ``hits`` list with a count, recursively.

    ``n_hits_total`` is left alone, so "there are rows and you are not seeing
    them" stays visible -- silently emptying the list would read as "no variants
    found", which is the not-evaluable-as-absent failure the rubrics exist to
    catch.
    """
    if isinstance(slice_, dict):
        out: dict[str, Any] = {}
        for key, value in slice_.items():
            if key == "hits" and isinstance(value, list) and value:
                out[key] = f"[{len(value)} rows omitted; see n_hits_total]"
            else:
                out[key] = _without_hit_rows(value)
        return out
    if isinstance(slice_, list):
        return [_without_hit_rows(v) for v in slice_]
    return slice_


def _dedupe_hits(
    members: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Hoist a hit list that two members share into one table.

    Members declaring the same ``evidence_hits_col`` each carry an identical copy,
    so the rows are serialised twice -- ``llm.py:2057`` records the same artifact
    in the production path. On the worst M cell that is 29,965 chars duplicated,
    taking the reference from 114,481 to ~84,500 chars with nothing lost. Each
    member keeps its own scalars and a pointer to the shared table.

    Returns ``(members, shared)``; ``shared`` is empty when nothing was duplicated,
    so single-member and unique-hit categories are untouched.
    """
    import hashlib

    counts: dict[str, int] = {}
    digests: dict[int, str] = {}
    for i, member in enumerate(members):
        hits = member.get("hits")
        if not hits:
            continue
        digest = hashlib.sha1(json.dumps(hits, sort_keys=True, default=str).encode()).hexdigest()[
            :12
        ]
        digests[i] = digest
        counts[digest] = counts.get(digest, 0) + 1

    shared: dict[str, Any] = {}
    out: list[dict[str, Any]] = []
    for i, member in enumerate(members):
        digest = digests.get(i)
        if digest is None or counts[digest] < 2:
            out.append(member)
            continue
        key = f"hits_{digest}"
        shared.setdefault(key, member["hits"])
        out.append({**member, "hits": f"see shared_hit_tables.{key}"})
    return out, shared


def _category_spec(letter: str) -> dict[str, Any]:
    """The ``CATEGORIES`` entry for a letter."""
    for cat in ev.CATEGORIES:
        if cat.get("letter") == letter:
            return cat
    raise ReferenceError(f"unknown category letter {letter!r}; expected one of {CATEGORY_NAMES}")


def render(reference: dict[str, Any]) -> str:
    """The reference as the text the judge sees.

    JSON rather than prose: it is what every arm's own payload was, so the judge
    reads the evidence in the form the system under test produced it, and no
    prose-rendering step can quietly editorialise.
    """
    return json.dumps(reference, indent=2, ensure_ascii=False, default=str, sort_keys=True)


# Reference budget. The context is 32,768; a pairwise call also carries two
# responses, a rubric, the template and 512 tokens of feedback to generate. 29,000
# leaves ~3,700 for all of that, measured against the corpus's longest responses.
REFERENCE_BUDGET_TOKENS = 29_000


@dataclass
class Fitted:
    """A reference trimmed to fit, and what trimming it took."""

    reference: dict[str, Any]
    tokens: int
    steps_applied: list[str]

    @property
    def trimmed(self) -> bool:
        """Whether anything had to be dropped."""
        return bool(self.steps_applied)


def fit_reference(
    reference: dict[str, Any],
    count_tokens: Callable[[str], int],
    *,
    budget: int = REFERENCE_BUDGET_TOKENS,
) -> Fitted:
    """Trim a reference until it fits, dropping the least load-bearing parts first.

    The `--check-context` gate found 414 of 37,350 prompts over the 32,768 context,
    the worst at 48,166 tokens -- against a 4-chars/token estimate that said
    26,274. JSON dense with numbers and punctuation tokenizes far worse than
    prose, so the estimate was 1.8x optimistic and only the real tokenizer settles
    it. An over-long prompt is not an error in vLLM; it is silently truncated, so
    the judge would score evidence whose end it never saw.

    Trimming order, least informative first:

    1. the distribution spread (p05-p95), keeping each metric's value + percentile
    2. hit rows, replaced by their counts -- ``n_hits_total`` survives, so "there
       are rows you are not seeing" stays visible rather than reading as "none"
    3. the distribution block entirely
    4. hard truncation, marked in the payload

    Applied per *cell*, so all 9 arms in that cell are still judged against exactly
    the same evidence -- the property the whole design rests on. What was dropped is
    returned rather than logged away, because a trimmed cell is weaker evidence and
    the write-up has to be able to say which cells those were.
    """
    steps: list[tuple[str, Callable[[dict[str, Any]], dict[str, Any]]]] = [
        ("drop_distribution_spread", _drop_spread),
        ("drop_hit_rows", _without_hit_rows),
        ("drop_distribution", _drop_distribution),
    ]

    current = reference
    applied: list[str] = []
    tokens = count_tokens(render(current))
    if tokens <= budget:
        return Fitted(reference=current, tokens=tokens, steps_applied=applied)

    for name, step in steps:
        current = step(current)
        applied.append(name)
        tokens = count_tokens(render(current))
        if tokens <= budget:
            return Fitted(reference=current, tokens=tokens, steps_applied=applied)

    # Still over: truncate the rendered JSON and say so in the payload, rather
    # than let vLLM cut it off where the judge cannot tell.
    applied.append("truncated")
    current = dict(current)
    current["_truncation_note"] = (
        f"This evidence was truncated to fit the judge's {budget}-token budget "
        f"after {', '.join(applied[:-1])}. Absent fields were not measured as "
        f"absent; they were dropped."
    )
    current = _shrink_to_budget(current, count_tokens, budget)
    return Fitted(reference=current, tokens=count_tokens(render(current)), steps_applied=applied)


def _drop_spread(reference: dict[str, Any]) -> dict[str, Any]:
    """Remove p05-p95 from every distribution entry, keeping value + percentile."""
    keep = {"value", "pctile", "percentile", "stratum", "n", "metric"}

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "pctile" in node or "percentile" in node:
                return {k: v for k, v in node.items() if k in keep}
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    out = dict(reference)
    if "distribution" in out:
        out["distribution"] = walk(out["distribution"])
    return out


def _drop_distribution(reference: dict[str, Any]) -> dict[str, Any]:
    """Drop the population-percentile block entirely."""
    return {k: v for k, v in reference.items() if k != "distribution"}


def _shrink_to_budget(
    reference: dict[str, Any],
    count_tokens: Callable[[str], int],
    budget: int,
) -> dict[str, Any]:
    """Shorten the longest list until the reference fits, and no further.

    Binary search on list length, not repeated halving. Halving overshot badly:
    on the 25 synthesis cells that reached this step it cut ``criteria_evidence``
    from ~30k tokens to ~15k against a 29k budget -- discarding ~14k tokens of
    evidence to save 1k. Every token dropped here is evidence the judge will not
    see, so even the last resort should land just under the line.
    """
    out = dict(reference)
    for _ in range(12):  # each pass shrinks the then-largest list
        if count_tokens(render(out)) <= budget:
            return out
        target = _largest_list_key(out)
        if target is None:
            return out
        items = out[target]
        lo, hi, best = 1, len(items), None
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_tokens(render({**out, target: items[:mid]})) <= budget:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        if best is None:
            # Even one item overflows; drop the block and try the next largest.
            out.pop(target, None)
            continue
        out[target] = items[:best]
        return out
    return out


def _largest_list_key(reference: dict[str, Any]) -> str | None:
    """The top-level key holding the longest-serialising list."""
    sizes = {
        k: len(json.dumps(v, default=str))
        for k, v in reference.items()
        if isinstance(v, list) and len(v) > 1
    }
    return max(sizes, key=lambda k: sizes[k]) if sizes else None


def isoform_records(records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """``{tis_slug: isoform_record}``, with the gene folded in.

    ``slice_category`` and ``slice_criterion`` both read gene-level fields off the
    isoform record, so the gene has to be merged in rather than passed alongside.
    """
    out: dict[str, dict[str, Any]] = {}
    for gene_name, record in records.items():
        gene = record.get("gene") or {}
        for iso in record.get("isoforms") or []:
            merged = {**iso, "gene": gene, "gene_name": gene.get("name") or gene_name}
            out[tis_slug(iso.get("tis_id") or "")] = merged
    return out
