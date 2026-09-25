"""Facts about the corpus that arithmetic settles, so no rubric has to ask.

Prometheus grades whether a verdict is supported by the text in front of it. It
has no isoform biology, and asking it to check whether a number appears in a
payload wastes a judgment on something a regex decides exactly. These checks run
first, cost nothing, and can change the rubrics -- a corpus riddled with
fabricated numbers would make R1 the whole story instead of one axis of four.

Three checks:

1. :func:`check_fabrication` -- every number in the reasoning must appear in that
   cell's reference. Automates the PR #24 audit, which found zero fabrications
   across 18 isoforms and is the reason the rubrics measure judgment rather than
   fidelity.
2. :func:`check_not_evaluable` -- reasoning that calls a never-measured property
   absent. An objective floor under R2, the axis issue #30 exists for.
3. :func:`check_synthesis_fields` -- schema-valid but degenerate synthesis output.

Each returns findings, never a verdict. A finding is evidence about an arm, not a
score for it.

**The fabrication rate is an upper bound, not a measurement.** Five systematic
false-positive classes were found and fixed by reading the output of successive
runs: range dashes read as minus signs (a 54% rate on the first run was almost
entirely "0.05-0.12" and "41st-75th"), decimal truncation, scientific notation,
percentile landmarks, and arithmetic derived from the isoform's own lengths. What
survives on cheeseman50 is 147 findings over the 1,800 non-tool-loop category
outputs (8.2%), and spot-checking says some of that is still derived arithmetic in
forms not enumerated here. Treat it as a ceiling and read the *per-arm spread*,
which is the durable signal: the two ``raw`` arms carry ~75% more findings than
the two ``criteria`` arms (49/47 vs 28/28), which is the reference-payload confound
made visible -- the reference deliberately excludes the ``raw`` grounding, so those
arms cite numbers it does not contain.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable

from swissisoform.judge import SYNTHESIS_UNIT

# A number as it appears in prose: optional sign, digits, optional decimals,
# optional exponent, optional trailing %. Commas in thousands are handled by
# stripping them before the match.
#
# The leading `(?<![\w.])` is load-bearing. Without it a range reads as a negative
# number: "pLDDT 0.05-0.12" yields -0.12, "91-97%" yields -97, "residues 21-28"
# yields -28, and -- because the dash can follow a letter -- "41st-75th percentile"
# yields -75. This prose is full of both forms. A `\d`-only lookbehind still let
# the ordinal ranges through; `\w` catches both, and a genuine "-0.42" after a
# space is unaffected.
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?")

# Numbers too common to be evidence of anything. Matching "1" or "0" against a
# payload proves nothing, and small integers appear as list indices, counts and
# rhetorical ("one of two") alike.
_UNINFORMATIVE = frozenset(
    {
        "0",
        "1",
        "2",
        "3",
        "100",
        # Percentile landmarks. "above the 95th percentile" names a threshold the
        # writer chose, not a value read out of the evidence.
        "5",
        "10",
        "25",
        "50",
        "75",
        "90",
        "95",
        "99",
    }
)

# Phrasings that assert a property is absent. Paired with a field the reference
# marks unmeasured, each is a not-evaluable violation.
_ABSENCE_PATTERNS: tuple[str, ...] = (
    r"\bno\s+{kw}\b",
    r"\bnot\s+{kw}\b",
    r"\b{kw}\s+(?:is|are|was|were)\s+(?:absent|not\s+(?:present|detected|found))\b",
    r"\bwithout\s+{kw}\b",
    r"\blacks?\s+{kw}\b",
    r"\bfails?\s+to\s+show\s+{kw}\b",
)

# Values in an evidence payload that mean "never measured", as opposed to
# "measured and negative". Keeping these two apart is the entire point of R2.
UNMEASURED_MARKERS: frozenset[str] = frozenset(
    {
        "not_evaluable",
        "not_run",
        "no_cache",
        "no_skeleton",
        "no_hits",
        "no_alignment",
        "no_support",
        "unknown",
        "unavailable",
        "too_long",
        "region_map_not_implemented",
    }
)


@dataclass
class Finding:
    """One check firing on one output."""

    check: str
    arm: str
    slug: str
    unit: str
    detail: str
    context: str = ""


@dataclass
class CheckReport:
    """Findings plus the denominators needed to read them."""

    findings: list[Finding] = field(default_factory=list)
    n_outputs: int = 0
    n_with_findings: int = 0
    counts_by_arm: dict[str, int] = field(default_factory=dict)
    counts_by_unit: dict[str, int] = field(default_factory=dict)
    words_by_arm: dict[str, list[int]] = field(default_factory=dict)
    words_by_arm_unit: dict[str, list[int]] = field(default_factory=dict)

    def add(self, finding: Finding) -> None:
        """Record a finding and its per-arm / per-unit tallies."""
        self.findings.append(finding)
        self.counts_by_arm[finding.arm] = self.counts_by_arm.get(finding.arm, 0) + 1
        self.counts_by_unit[finding.unit] = self.counts_by_unit.get(finding.unit, 0) + 1

    def measure(self, *, arm: str, unit: str, reasoning: str) -> None:
        """Record one output's length. A control, not a check.

        Nothing in the judge measured response size, so an ordering that moved
        after the rubric started weighing economy could not be told apart from
        drift. Deliberately not a ``Finding``: length is evidence about an arm,
        and there is no threshold at which it is wrong.
        """
        n = len((reasoning or "").split())
        self.words_by_arm.setdefault(arm, []).append(n)
        self.words_by_arm_unit.setdefault(f"{arm}\t{unit}", []).append(n)

    def rate(self) -> float:
        """Share of outputs with at least one finding."""
        return self.n_with_findings / self.n_outputs if self.n_outputs else 0.0


def word_stats(values: Iterable[int]) -> dict[str, float]:
    """``n`` / ``median`` / ``p90`` over one group's word counts."""
    xs = sorted(values)
    if not xs:
        return {"n": 0, "median": 0.0, "p90": 0.0}
    return {
        "n": len(xs),
        "median": float(statistics.median(xs)),
        "p90": float(xs[min(len(xs) - 1, int(0.9 * len(xs)))]),
    }


def numbers_in(text: str) -> set[str]:
    """Every number in *text*, normalised, minus the uninformative ones."""
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text or "")
    out: set[str] = set()
    for raw in _NUMBER.findall(cleaned):
        token = raw.rstrip("%").lstrip("+")
        if token in _UNINFORMATIVE or token.lstrip("-") in _UNINFORMATIVE:
            continue
        out.add(token)
    return out


def _numeric_strings(value: Any, into: set[str]) -> None:
    """Collect every number reachable in a nested payload, as strings."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        into.add(_fmt(value))
        return
    if isinstance(value, str):
        into |= numbers_in(value)
        return
    if isinstance(value, dict):
        for key, sub in value.items():
            into |= numbers_in(str(key))
            _numeric_strings(sub, into)
        return
    if isinstance(value, (list, tuple)):
        for sub in value:
            _numeric_strings(sub, into)


def _fmt(value: float) -> str:
    """A number as the shortest string that round-trips."""
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def reference_numbers(reference: dict[str, Any]) -> set[str]:
    """Every number in a cell's reference, plus common roundings.

    A response quoting 4.03 for a stored 4.031841 is being accurate, not
    fabricating. So each stored value is also emitted rounded to 0-6 decimal
    places, truncated (not rounded) at each of those lengths, and as a percentage
    of a fraction and the reverse. Truncation matters as much as rounding: 0.0063
    is the natural way to cite 0.006312, and rounding alone does not produce it.
    Without this the check flags correct citation as invention.
    """
    raw: set[str] = set()
    _numeric_strings(reference, raw)

    out: set[str] = set(raw)
    for token in raw:
        try:
            value = float(token)
        except ValueError:
            continue
        for places in range(7):
            out.add(f"{value:.{places}f}")
            out.add(_fmt(round(value, places)))
            # Truncation, e.g. 0.006312 -> "0.0063".
            text = f"{value:.10f}"
            if "." in text:
                whole, frac = text.split(".")
                out.add(whole if places == 0 else f"{whole}.{frac[:places]}")
        # Cited as a percentage of a stored fraction, and the reverse.
        for scaled in (value * 100, value / 100 if value else 0.0):
            for places in range(5):
                out.add(f"{scaled:.{places}f}")
                out.add(_fmt(round(scaled, places)))
        # Scientific notation, at several precisions. A p-value stored as 0.00025
        # is naturally cited as 2.5e-4, and one stored as 3.4e-61 as "~1e-61";
        # neither is reachable by decimal rounding, and D's reasoning is full of
        # both -- they were most of what survived the range-dash fix.
        for places in range(4):
            out.add(f"{value:.{places}e}")
            out.add(f"{value:.{places}e}".replace("e-0", "e-").replace("e+0", "e+"))
        # Order-of-magnitude form: 3.4e-61 cited as "~1e-61". The exponent is
        # derived from a real value, so the number is not *invented* -- which is
        # all this check decides. Whether it was quoted precisely is R1's job
        # ("quotes their values correctly"), and that division is deliberate: a
        # regex should not arbitrate citation style.
        if value:
            exponent = f"{value:.0e}".split("e")[1]
            out.add(f"1e{exponent}")
            out.add(f"1e{exponent}".replace("e-0", "e-").replace("e+0", "e+"))

    # Differences and sums of the isoform's own lengths. "removes the N-terminal
    # 198 residues" cites 243 - 45, which is arithmetic on reference values, not
    # invention -- and it was the single most common finding in L. Restricted to
    # the handful of length fields on purpose: pairwise combinations of *every*
    # number would accept almost anything and the check would stop checking.
    lengths = _identity_lengths(reference)
    # The lengths themselves, not only their combinations. A sequence field
    # contributes exactly one length, so the pairwise loop below never reaches it
    # and "the added 55-aa extension" stayed flagged -- which is what
    # test_sequence_length_is_not_fabrication caught.
    for length in lengths:
        out.add(_fmt(length))
    for i, a in enumerate(lengths):
        for b in lengths[i + 1 :]:
            for derived in (abs(a - b), a + b):
                out.add(_fmt(derived))
                if b:
                    out.add(_fmt(round(100 * a / b)))
                    out.add(f"{100 * a / b:.1f}")
    return {token.rstrip(".") for token in out}


def _identity_lengths(reference: dict[str, Any]) -> list[float]:
    """Protein/region lengths anywhere in a reference, deduplicated."""
    keys = (
        "isoform_length_aa",
        "canonical_length_aa",
        "length_aa",
        "diff_region_length",
        "unique_length_aa",
        "aa_len",
    )
    # Sequence fields whose *length* is the quantity, not their contents. "the
    # added 55-aa N-terminal extension" is len(differential_sequence); the number
    # appears nowhere in the payload as a number, so without this every response
    # that says how long the differential region is looks like a fabrication. It
    # was the largest remaining class after the range and rounding fixes.
    sequence_keys = ("differential_sequence", "unique_sequence", "diff_sequence")
    found: set[float] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in keys and isinstance(value, (int, float)) and not isinstance(value, bool):
                    found.add(float(value))
                elif key in sequence_keys and isinstance(value, str) and value:
                    found.add(float(len(value)))
                else:
                    walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(reference)
    return sorted(found)


# Categories whose verdict comes from a tool loop. Their arms queried the full
# variant / structure tables through readers, while the reference carries only the
# MAX_HITS=30 sample -- 30 of 13,690 variant rows on the worst M cell. So a number
# absent from the reference is not evidence of fabrication there; it is usually a
# correctly-read tool result. The check is skipped rather than reported loosely,
# and the skip is stated in the report.
#
# It costs little: on M and P the noise floor is already 16% and 18%, so those two
# categories were never going to carry a fidelity finding.
TOOL_LOOP_UNITS: frozenset[str] = frozenset({"M", "P"})


def check_fabrication(
    *, arm: str, slug: str, unit: str, reasoning: str, reference: dict[str, Any]
) -> list[Finding]:
    """Numbers in *reasoning* that the cell's reference does not contain.

    Returns nothing for the tool-loop units -- see :data:`TOOL_LOOP_UNITS`.
    """
    if unit in TOOL_LOOP_UNITS:
        return []
    allowed = reference_numbers(reference)
    unmatched = sorted(numbers_in(reasoning) - allowed)
    if not unmatched:
        return []
    return [
        Finding(
            check="fabrication",
            arm=arm,
            slug=slug,
            unit=unit,
            detail=f"{len(unmatched)} number(s) not in the reference: {unmatched[:8]}",
            context=_excerpt(reasoning, unmatched[0]),
        )
    ]


def unmeasured_fields(reference: dict[str, Any]) -> dict[str, str]:
    """``{field_name: marker}`` for everything the reference says was not measured.

    A null is included: a criterion whose value is ``None`` was not evaluable, and
    conflating that with ``False`` is the failure R2 scores.
    """
    found: dict[str, str] = {}

    def walk(node: Any, name: str = "") -> None:
        if isinstance(node, dict):
            label = str(node.get("id") or node.get("label") or node.get("name") or name)
            for key, value in node.items():
                if isinstance(value, str) and value.strip().lower() in UNMEASURED_MARKERS:
                    found[f"{label}.{key}" if label else key] = value.strip().lower()
                elif key in ("value", "state") and value is None:
                    found[label or key] = "null"
                else:
                    walk(value, key)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item, name)

    walk(reference)
    return found


def check_not_evaluable(
    *,
    arm: str,
    slug: str,
    unit: str,
    reasoning: str,
    reference: dict[str, Any],
    keywords: Iterable[str] | None = None,
) -> list[Finding]:
    """Reasoning that asserts absence of something the reference never measured.

    Keyword-driven and therefore a **lower bound**: it catches the explicit
    phrasings in :data:`_ABSENCE_PATTERNS` against the field names the reference
    marks unmeasured. R2 is what catches the rest; this is the part that needs no
    judgment at all.
    """
    unmeasured = unmeasured_fields(reference)
    if not unmeasured:
        return []
    terms = list(keywords) if keywords is not None else _keywords_from(unmeasured)
    low = (reasoning or "").lower()
    hits: list[str] = []
    for term in terms:
        for pattern in _ABSENCE_PATTERNS:
            if re.search(pattern.format(kw=re.escape(term)), low):
                hits.append(term)
                break
    if not hits:
        return []
    return [
        Finding(
            check="not_evaluable",
            arm=arm,
            slug=slug,
            unit=unit,
            detail=(
                f"asserts absence of {sorted(set(hits))[:5]} which the reference "
                f"marks unmeasured ({len(unmeasured)} unmeasured field(s))"
            ),
            context=_excerpt(reasoning, sorted(set(hits))[0]),
        )
    ]


# Field-name fragments that do not name a property. "no evidence for X" is normal
# correct prose, and matching it against a field called `..._evidence` reported a
# violation on every P cell in every arm -- the first run's only not_evaluable
# findings were all this.
_GENERIC_FRAGMENTS: frozenset[str] = frozenset(
    {
        "evidence",
        "value",
        "score",
        "scores",
        "status",
        "result",
        "results",
        "data",
        "total",
        "count",
        "counts",
        "mean",
        "region",
        "reason",
        "label",
        "isoform",
        "canonical",
        "threshold",
        "cutoff",
        "metric",
        "summary",
    }
)


def _keywords_from(unmeasured: dict[str, str]) -> list[str]:
    """Prose terms to look for, derived from unmeasured field names."""
    terms: set[str] = set()
    for field_name in unmeasured:
        head = field_name.split(".")[0]
        for part in re.split(r"[_.]", head):
            part = part.strip().lower()
            if len(part) > 4 and not part.isdigit() and part not in _GENERIC_FRAGMENTS:
                terms.add(part)
    return sorted(terms)


def _excerpt(text: str, needle: str, width: int = 90) -> str:
    """A short window of *text* around *needle*, for eyeballing a finding."""
    low = (text or "").lower()
    at = low.find(str(needle).lower())
    if at < 0:
        return (text or "")[:width]
    start = max(0, at - width // 2)
    return (text or "")[start : start + width].replace("\n", " ")


def check_synthesis_fields(*, arm: str, slug: str, payload: dict[str, Any]) -> list[Finding]:
    """Degenerate synthesis output that is still schema-valid.

    Measured on the corpus and uniform across arms: ``confidence`` is never
    ``high`` in any of the 450 outputs, and ``tags`` is empty on 18-23 of 50 per
    arm. Both are reported as corpus facts rather than arm defects -- but the
    empty-``tags`` rate is a candidate metric in its own right, so it is counted
    per arm here.
    """
    out: list[Finding] = []
    for key in ("headline", "divergence_hypothesis", "function_relevance"):
        if not (payload.get(key) or "").strip():
            out.append(
                Finding(
                    check="synthesis_empty_field",
                    arm=arm,
                    slug=slug,
                    unit=SYNTHESIS_UNIT,
                    detail=f"{key} is empty",
                )
            )
    if not (payload.get("tags") or []):
        out.append(
            Finding(
                check="synthesis_no_tags",
                arm=arm,
                slug=slug,
                unit=SYNTHESIS_UNIT,
                detail="no isoform-change tag fired",
            )
        )
    return out
