r"""Query-time readers over ``variants_long.parquet`` for the M-category LLM pass.

The Mutation-Landscape read gets ~20 scalars plus at most 30 of an isoform's
several hundred variants, so it can quote an enrichment ratio but cannot ask
whether a cluster is tight or diffuse. These readers move that compression from
precompute time to query time: exposed as Anthropic tools (:data:`M_TOOLS`),
executed locally by :func:`make_m_dispatch`, driven by
``swissisoform.site.llm.run_tool_loop``.

``variants_long.parquet`` is an existing required build artifact (written by
``evidence.write_variants_long``, already read in full by the website's clinical
panel), so this adds a consumer, not a dependency. Pure reads throughout; the
only persisted state is an in-process cache of the parquet.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import pandas as pd

# ``_to_native`` is the package's numpy/pandas -> JSON-native coercion (NaN and
# pd.NA become None). Imported rather than duplicated so a tool result and the
# evidence record serialise a value identically. ``_SEPARATE_ORFS`` likewise —
# the frame caveat and the evidence record must agree on which ORF types share
# no reading frame with the canonical CDS.
from swissisoform.site.evidence import _SEPARATE_ORFS, _to_native, clinsig_family

__all__ = [
    "DATA_TOOL_NAMES",
    "EMIT_VERDICT",
    "M_TOOLS",
    "VariantsLongMissing",
    "make_m_dispatch",
    "query_variants",
    "require_variants_long",
    "variant_effect_stats",
    "variant_position_histogram",
]


# ── Limits ────────────────────────────────────────────────────────────────

DEFAULT_QUERY_LIMIT = 25
MAX_QUERY_LIMIT = 100
# Cap on the per-residue histogram so a 1,600-variant isoform cannot flood the
# context. The concentration statistics are computed over the FULL distribution
# before truncation, so the shape summary stays accurate when rows are dropped.
MAX_HISTOGRAM_POSITIONS = 100

# Columns returned by query_variants. A deliberate subset: enough to reason about
# clustering, clinical call, predicted effect and population frequency, without
# the genomic-coordinate and cross-frame columns the model has no use for. BOTH
# protein-position columns are carried because only one of them is populated over
# the differential region — see _position_column.
QUERY_VARIANT_COLUMNS = (
    "variant_id",
    "source",
    "protein_pos",
    "isoform_protein_pos",
    "in_isoform_unique",
    "clinical_significance",
    "consequence",
    # The frame-resolved consequence varianteffect actually scored (isoform frame
    # for unique-region hits), which is where a start_lost call surfaces —
    # ``consequence`` alone is canonical-frame and reads "intronic" there.
    "effect_consequence",
    "isoform_codon_ref",
    "isoform_codon_alt",
    "hgvsp",
    "am_pathogenicity",
    "plm_delta_llr",
    "plm_status",
    "cosmic_sample_count",
    "allele_frequency",
)

_REGION_COLUMN = {"unique": "in_isoform_unique", "shared": "in_isoform_shared"}
_REGIONS = ("unique", "shared", "any")
_CLINSIG_FILTERS = ("pathogenic", "benign", "uncertain", "conflicting", "any")
_SOURCES = ("gnomAD", "ClinVar", "COSMIC")

# Sort keys other than "position", which is resolved per isoform by
# _position_column. (column, ascending). ``plm_delta_llr`` sorts ascending
# because the ESM-C log-likelihood ratio is negative for destabilising
# substitutions, so the most damaging variant is the smallest value.
_SORT_KEYS: dict[str, tuple[str, bool]] = {
    "am_pathogenicity": ("am_pathogenicity", False),
    "plm_delta_llr": ("plm_delta_llr", True),
    "cosmic_sample_count": ("cosmic_sample_count", False),
    "allele_frequency": ("allele_frequency", False),
}
_SORT_CHOICES = ("position", *_SORT_KEYS)


def _position_column(orf_type: Any) -> tuple[str, str]:
    """Pick the protein-position column that is populated for this ORF type.

    The differential region lives in a different coordinate space depending on
    the isoform kind, and ``variants_long`` reflects that exactly: over the
    unique region, a truncation's variants have ``protein_pos`` (canonical
    numbering — the removed region only exists in the canonical protein) and a
    NULL ``isoform_protein_pos``, while an extension's have the reverse (the
    added region only exists in the isoform). Measured on the cheeseman set,
    each is 100% null in the other's unique region.

    Keying a histogram on one column alone therefore returns an empty
    distribution for half of all isoforms, over precisely the region the M
    category is about. Returns ``(column, space)`` so callers can label which
    numbering their residue positions are in.
    """
    if str(orf_type or "").strip().lower() == "truncated":
        return "protein_pos", "canonical"
    return "isoform_protein_pos", "isoform"


class VariantsLongMissing(FileNotFoundError):
    """Raised when ``variants_long.parquet`` is absent but tools were requested."""


# ── Loading ───────────────────────────────────────────────────────────────


def _key(path: Path | str) -> str:
    """Cache key: an absolute path string, so equivalent spellings share a load."""
    return str(Path(path).expanduser().resolve())


def require_variants_long(path: Path | str) -> Path:
    """Validate that the variants table exists, with an actionable message.

    Called once at wiring time so a misconfigured run fails immediately rather
    than after the first API call. Deliberately NOT a soft fallback to the
    single-shot path: an M verdict produced with tools and one produced without
    are not comparable, and silently choosing between them per run is exactly the
    hidden state the fresh-rerun contract forbids.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise VariantsLongMissing(
            f"variants_long.parquet not found at {p}. It is produced by the "
            "`evidence` stage (scripts/site/build_evidence_records.py "
            "--variants-long-out ...), which runs immediately before the LLM "
            "category pass in scripts/build_website.sh. Re-run that stage, pass "
            "--variants-long explicitly, or pass --no-tools to run the M "
            "category through the single-shot path instead."
        )
    return p


@lru_cache(maxsize=4)
def _load(path_str: str) -> pd.DataFrame:
    return pd.read_parquet(path_str)


@lru_cache(maxsize=4)
def _index_by_tis(path_str: str) -> dict[str, pd.DataFrame]:
    """Group the table by ``tis_id`` once so each reader call is a dict lookup."""
    df = _load(path_str)
    if "tis_id" not in df.columns:
        return {}
    return {str(k): v for k, v in df.groupby("tis_id", sort=False)}


def _empty(path_str: str) -> pd.DataFrame:
    """An empty frame with the real column set, so readers need no null-guards."""
    return _load(path_str).iloc[0:0]


# ── Filtering ─────────────────────────────────────────────────────────────


def _check(name: str, value: Any, allowed: tuple[str, ...]) -> None:
    if value is not None and value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}; got {value!r}")


def _iso(
    path: Path | str,
    tis_id: str,
    *,
    region: str | None = None,
    clinsig: str | None = None,
    source: str | None = None,
    consequence: str | None = None,
) -> pd.DataFrame:
    """Rows for one isoform, narrowed by region / clinical call / source.

    The single place that knows this table's column semantics, so the readers
    cannot drift apart. Three subtleties it exists to encapsulate: ``unique`` and
    ``shared`` partition the rows (every row is in exactly one; unique is only ~14%
    of them, and on a truncation it is the canonical segment the isoform LOSES),
    so each filter returns one side of the alternative start codon and
    ``region="any"`` is the only way to see both at once — there is no positional
    window. The shared segment is sequence-identical to the corresponding stretch
    of the canonical protein, except that ``install_initiator_met`` forces residue
    0 to M, so a near-cognate truncation start differs there. ``clinsig`` matches by family
    via :func:`evidence.clinsig_family`, never equality (ClinVar spells a
    pathogenic call three ways, and equality undercounts by ~20%); and ``clinsig``
    is ClinVar-only, so setting it at all excludes every gnomAD and COSMIC row —
    hence ``source`` being separate, to keep COSMIC recurrence reachable.

    Raises:
        ValueError: on an unrecognised region / clinsig / source value.
    """
    _check("region", region, _REGIONS)
    _check("clinsig", clinsig, _CLINSIG_FILTERS)
    _check("source", source, _SOURCES)

    key = _key(path)
    sub = _index_by_tis(key).get(str(tis_id))
    if sub is None:
        return _empty(key)

    if region and region != "any":
        col = _REGION_COLUMN[region]
        if col in sub.columns:
            sub = sub[sub[col].fillna(False).astype(bool)]

    if clinsig and "clinical_significance" in sub.columns:
        families = sub["clinical_significance"].map(clinsig_family)
        sub = sub[families != "none"] if clinsig == "any" else sub[families == clinsig]

    if source and "source" in sub.columns:
        sub = sub[sub["source"].astype(str).str.lower() == source.lower()]

    if consequence and "consequence" in sub.columns:
        sub = sub[sub["consequence"].fillna("").astype(str).str.contains(consequence, case=False)]

    return sub


def _effect_caveat(orf_type: Any) -> str | None:
    """Per-predictor validity note for a unique region that was never canonical CDS.

    The two effect predictors behave differently there, and merging them was
    wrong in one direction:

    - ``am_pathogenicity`` (AlphaMissense) IS canonical-frame — a precomputed
      table keyed by canonical transcript position, with no entry for a position
      outside the canonical CDS. Absent by construction (0 of 958 extension
      unique-region rows on cheeseman_test carry a score).
    - ``plm_delta_llr`` (ESM-C) is frame-aware: ``varianteffect._score_hit_plm``
      scores an ``in_isoform_unique`` hit against the ISOFORM's log-probs. It is
      populated (807 of those same 958 rows) and usable. What "never coding"
      costs it is calibration, not validity — the sequence exists and is
      scoreable, it simply has no history of purifying selection behind it.

    Emitted on the tool result because the M prompt treats a ``caveat`` field as
    a hard constraint, so this text is what the model is bound by.
    """
    s = str(orf_type or "").strip().lower()
    if s == "extended":
        subject = (
            "this is an extension, so its unique region was 5'UTR or intron and "
            "was never canonical coding sequence"
        )
    elif s in _SEPARATE_ORFS:
        subject = (
            "this isoform is a separate ORF and does not share a reading frame "
            "with the canonical CDS, so no part of it is canonical coding sequence"
        )
    else:
        return None
    return (
        f"FRAME CAVEAT: {subject}. The two effect predictors differ here and must "
        "not be treated alike. am_pathogenicity is ABSENT BY CONSTRUCTION over "
        "this region: AlphaMissense is a precomputed canonical-transcript table "
        "with no entry for a position outside the canonical CDS, so a null is a "
        "missing lookup, never evidence of tolerance — do not average it, do not "
        "report its absence as a finding. plm_delta_llr IS scored here, in the "
        "ISOFORM's own frame (the pipeline scores unique-region variants against "
        "the isoform protein), so the values are real and you may use them. "
        "Calibrate rather than discard: ESM-C reports how disruptive the "
        "substitution is in this sequence context, NOT evidence of purifying "
        "selection, because this sequence has no coding evolutionary history. "
        "Read a low delta LLR as model-perceived disruptiveness; do not convert "
        "it into a claim about constraint, and do not read a mild one as proof "
        "the region tolerates variation. Both predictors are unaffected over the "
        "shared region. "
        "START CODON: a hit at isoform_protein_pos 0 sits in this isoform's own "
        "start codon and carries NO effect score by construction — plm_status is "
        "'start_codon' and AlphaMissense is absent — because what decides such a "
        "variant is whether the trinucleotide still initiates translation (CTG "
        "does, CTA does not), which an amino-acid substitution score cannot "
        "express. A third-base change can abolish the start while leaving the "
        "residue identical, so a 'synonymous' call there is not reassurance. "
        "Treat a position-0 hit as inherently notable and read it on the codon: "
        "isoform_codon_ref -> isoform_codon_alt is carried on every row, and a "
        "substitution that leaves the codon able to initiate (ATG or a near-cognate: "
        "CTG, GTG, TTG, ACG, AGG, AAG, ATA, ATC, ATT) keeps the isoform intact, "
        "while one that drops out of that set ABLATES the start — the whole "
        "proteoform, not one residue. That case is labelled "
        "effect_consequence='start_lost' and scored damaging through the "
        "loss-of-function branch, so absent effect scores there are a definitive "
        "finding, not a gap. Note the plain 'consequence' column is canonical-frame "
        "and reads 'intronic' over an extension; effect_consequence is the one "
        "scored in the isoform's own frame."
    )


# ── Readers ───────────────────────────────────────────────────────────────


def query_variants(
    path: Path | str,
    tis_id: str,
    *,
    region: str | None = None,
    clinsig: str | None = None,
    source: str | None = None,
    consequence: str | None = None,
    sort_by: str = "position",
    limit: int = DEFAULT_QUERY_LIMIT,
    orf_type: Any = None,
) -> dict[str, Any]:
    """Return the actual variant rows matching a filter, capped at ``limit``.

    ``n_matched`` is the count BEFORE the cap, so a truncated result never reads
    as a complete one. ``position_space`` names the coordinate system the
    positional sort used — see :func:`_position_column`.
    """
    if sort_by not in _SORT_CHOICES:
        raise ValueError(f"sort_by must be one of {list(_SORT_CHOICES)}; got {sort_by!r}")
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        raise ValueError(f"limit must be an integer; got {limit!r}") from None
    limit = max(1, min(limit, MAX_QUERY_LIMIT))

    sub = _iso(path, tis_id, region=region, clinsig=clinsig, source=source, consequence=consequence)
    n_matched = int(len(sub))

    pos_col, pos_space = _position_column(orf_type)
    col, ascending = (pos_col, True) if sort_by == "position" else _SORT_KEYS[sort_by]
    if n_matched and col in sub.columns:
        # variant_id is the deterministic tiebreak: without it, rows sharing a
        # sort value could come back in a different order between runs.
        keys = [col, "variant_id"] if "variant_id" in sub.columns else [col]
        sub = sub.sort_values(
            keys, ascending=[ascending, True][: len(keys)], na_position="last", kind="mergesort"
        )

    cols = [c for c in QUERY_VARIANT_COLUMNS if c in sub.columns]
    rows = [_to_native(rec) for rec in sub.head(limit)[cols].to_dict(orient="records")]
    return {
        "n_matched": n_matched,
        "n_returned": len(rows),
        "truncated": n_matched > len(rows),
        "sort_by": sort_by,
        "position_column": pos_col,
        "position_space": pos_space,
        "rows": rows,
    }


def variant_position_histogram(
    path: Path | str,
    tis_id: str,
    *,
    region: str | None = None,
    clinsig: str | None = None,
    source: str | None = None,
    orf_type: Any = None,
) -> dict[str, Any]:
    """Per-residue variant counts over the protein, plus clustering statistics.

    Answers the question an enrichment ratio structurally cannot: is the signal a
    tight hotspot on a few residues or spread evenly across the region?

    Positions are read from the column that is populated for this isoform kind
    (:func:`_position_column`) and the coordinate system is reported as
    ``position_space`` — residue numbers over a truncation's unique region are
    CANONICAL numbering, over an extension's they are ISOFORM numbering. Rows with
    no position in that space are excluded from the histogram and counted in
    ``n_missing_pos``.
    """
    pos_col, pos_space = _position_column(orf_type)
    sub = _iso(path, tis_id, region=region, clinsig=clinsig, source=source)
    n = int(len(sub))

    def _empty_result(n_missing: int) -> dict[str, Any]:
        return {
            "n": n,
            "n_with_position": 0,
            "n_missing_pos": n_missing,
            "position_column": pos_col,
            "position_space": pos_space,
            "residue_span": None,
            "n_distinct_positions": 0,
            "per_residue_counts": {},
            "per_residue_counts_truncated": False,
            "top_positions": [],
            "frac_in_top10_positions": None,
        }

    if n == 0 or pos_col not in sub.columns:
        return _empty_result(n)

    pos = sub[pos_col].dropna()
    n_missing = n - int(len(pos))
    counts = pos.astype(int).value_counts().sort_index()
    total = int(counts.sum())
    if total == 0:
        return _empty_result(n_missing)

    # Concentration statistics come from the FULL distribution, before any
    # truncation of the emitted histogram.
    by_count = counts.sort_values(ascending=False, kind="mergesort")
    top10 = by_count.head(10)
    truncated = len(counts) > MAX_HISTOGRAM_POSITIONS
    kept = by_count.head(MAX_HISTOGRAM_POSITIONS).sort_index() if truncated else counts

    return {
        "n": n,
        "n_with_position": total,
        "n_missing_pos": n_missing,
        "position_column": pos_col,
        "position_space": pos_space,
        "residue_span": [int(counts.index.min()), int(counts.index.max())],
        "n_distinct_positions": int(len(counts)),
        "per_residue_counts": {int(k): int(v) for k, v in kept.items()},
        "per_residue_counts_truncated": bool(truncated),
        "top_positions": [{"pos": int(k), "count": int(v)} for k, v in top10.items()],
        "frac_in_top10_positions": round(float(top10.sum()) / total, 3),
    }


def _describe(series: pd.Series) -> dict[str, Any]:
    """Summary of a numeric column, reporting how many rows were actually scored."""
    vals = series.dropna() if series is not None else pd.Series(dtype="float64")
    if len(vals) == 0:
        return {"n_scored": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n_scored": int(len(vals)),
        "mean": round(float(vals.mean()), 4),
        "median": round(float(vals.median()), 4),
        "min": round(float(vals.min()), 4),
        "max": round(float(vals.max()), 4),
    }


def variant_effect_stats(
    path: Path | str,
    tis_id: str,
    *,
    region: str | None = None,
    clinsig: str | None = None,
    source: str | None = None,
    orf_type: Any = None,
) -> dict[str, Any]:
    """Distribution of the predicted-effect scores over a filtered variant set.

    ``n_rows`` and ``n_scored`` are reported separately on purpose: AlphaMissense
    scores missense SNVs only and is null for roughly half the table, so a mean
    over the scored subset is not a statement about the whole set.
    """
    sub = _iso(path, tis_id, region=region, clinsig=clinsig, source=source)
    n = int(len(sub))

    def _col(name: str) -> pd.Series:
        return sub[name] if name in sub.columns else pd.Series(dtype="float64")

    am_class: dict[str, int] = {}
    if n and "am_class" in sub.columns:
        am_class = {str(k): int(v) for k, v in sub["am_class"].dropna().value_counts().items()}

    damaging = _col("effect_damaging")
    out: dict[str, Any] = {
        "n_rows": n,
        "alphamissense": {**_describe(_col("am_pathogenicity")), "class_counts": am_class},
        "esm_llr": _describe(_col("plm_delta_llr")),
        "effect_damaging": {
            "true": int((damaging == True).sum()),  # noqa: E712 - nullable boolean
            "false": int((damaging == False).sum()),  # noqa: E712
            "null": int(damaging.isna().sum()) if n else 0,
        },
    }
    caveat = _effect_caveat(orf_type)
    if caveat:
        out["caveat"] = caveat
    return out


# ── Tool schemas ──────────────────────────────────────────────────────────

EMIT_VERDICT = "emit_verdict"

_REGION_PROP = {
    "type": "string",
    "enum": list(_REGIONS),
    "description": (
        "Which part of the isoform to read. 'unique' and 'shared' partition this "
        "isoform's variants — every row is in exactly one, so each filter returns "
        "one side of the alternative start codon. 'unique' = the differential "
        "region (what M1/M2 are about; a minority of variants): sequence the "
        "isoform GAINS on an extension, the canonical segment it LOSES on a "
        "truncation — there these are the variants the isoform no longer carries. "
        "'shared' = the retained core, sequence-identical to the corresponding "
        "stretch of the canonical protein (that stretch, not the whole protein). "
        "'any' = the union; with no positional filter here it is the only way to "
        "see a neighbourhood spanning the start codon in one result set. "
        "Default: any."
    ),
}
_CLINSIG_PROP = {
    "type": "string",
    "enum": list(_CLINSIG_FILTERS),
    "description": (
        "Clinical call family. Matched by family, so 'pathogenic' covers "
        "Pathogenic, Pathogenic/Likely pathogenic and Likely pathogenic. 'any' = "
        "any asserted call. NOTE: clinical significance is a ClinVar-only field, "
        "so setting this at all excludes every gnomAD and COSMIC variant. Omit it "
        "to read all sources."
    ),
}
_SOURCE_PROP = {
    "type": "string",
    "enum": list(_SOURCES),
    "description": (
        "Restrict to one database: gnomAD (population frequency), ClinVar "
        "(clinical assertions), COSMIC (somatic recurrence via "
        "cosmic_sample_count). Omit for all three."
    ),
}

M_TOOLS: list[dict[str, Any]] = [
    {
        "name": "variant_position_histogram",
        "description": (
            "Per-residue variant counts across the protein, with clustering "
            "statistics (residue span, top positions, and the fraction of the POSITIONED "
            "variants falling on the 10 busiest residues — rows with no position in the "
            "reported space are excluded from that fraction and counted in n_missing_pos, "
            "so read it against n and n_with_position). Use this to tell a tight hotspot "
            "from a diffuse "
            "spread — an enrichment ratio cannot express the difference. The result's "
            "position_space says whether the residue numbers are ISOFORM or CANONICAL "
            "numbering; always cite positions in the space it reports."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": _REGION_PROP,
                "clinsig": _CLINSIG_PROP,
                "source": _SOURCE_PROP,
            },
            "required": [],
        },
    },
    {
        "name": "query_variants",
        "description": (
            "The actual variant rows matching a filter, with clinical call, "
            "consequence, HGVSp, AlphaMissense score, ESM-C LLR, COSMIC sample count "
            "and allele frequency. Returns n_matched (the true total) alongside the "
            "capped row list. Use it to inspect specific variants — e.g. the "
            "pathogenic ones, or whatever sits on a hotspot residue."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": _REGION_PROP,
                "clinsig": _CLINSIG_PROP,
                "source": _SOURCE_PROP,
                "consequence": {
                    "type": "string",
                    "description": (
                        "Case-insensitive substring match on the consequence term, "
                        "e.g. 'missense', 'frameshift', 'stop_gained', 'start_lost'."
                    ),
                },
                "sort_by": {
                    "type": "string",
                    "enum": list(_SORT_CHOICES),
                    "description": (
                        "Ordering. 'position' ascends the protein; 'am_pathogenicity' "
                        "and 'cosmic_sample_count' and 'allele_frequency' descend "
                        "(most severe / most recurrent / most common first); "
                        "'plm_delta_llr' ascends, which is most-damaging-first because "
                        "the LLR is negative for destabilising substitutions. "
                        "Default: position."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Rows to return, 1-{MAX_QUERY_LIMIT} (default {DEFAULT_QUERY_LIMIT})."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "variant_effect_stats",
        "description": (
            "Distribution of the predicted-effect scores (AlphaMissense "
            "pathogenicity + class counts, ESM-C delta LLR, damaging flag) over a "
            "filtered variant set. Reports n_rows and n_scored separately because "
            "AlphaMissense covers missense SNVs only. Use it to test whether the "
            "clinically-flagged variants are also the ones predicted damaging."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": _REGION_PROP,
                "clinsig": _CLINSIG_PROP,
                "source": _SOURCE_PROP,
            },
            "required": [],
        },
    },
    {
        "name": EMIT_VERDICT,
        "description": (
            "Terminal call: record the category verdict and stop. Call it exactly "
            "once, after you have gathered enough data with the reader tools to "
            "justify the call."
        ),
        # Strict mode: the API constrains sampling to this schema, so the input
        # cannot arrive with a missing key, an extra key, or a parameter typed as
        # something other than what is declared. Applied to the terminal tool
        # specifically because its input is *persisted* — a malformed reader call
        # is answered with an error the model can recover from mid-loop, whereas a
        # malformed verdict is written to categories.json and rendered.
        #
        # Strict requires additionalProperties:false and rejects minItems>1, which
        # is why P's pae_block (two-element range params) stays unstrict.
        #
        # It guarantees the *shape*, not the prose: a string parameter may still
        # contain anything, so llm.py's _verdict_violations / _strip_verdict_markup
        # remain as the backstop over the value itself.
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["interesting", "neutral", "not_interesting"],
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "A few tight sentences, bottom line first, citing the specific "
                        "numbers that justify the verdict. Never write the submodule "
                        "ID codes (M1, M2)."
                    ),
                },
                "evidence_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Short notes on which tool findings actually drove the "
                        "verdict, one per entry."
                    ),
                },
            },
            # evidence_used is required so a verdict cannot be recorded without
            # its citations — under strict that is a guarantee, not a request.
            "required": ["verdict", "reasoning", "evidence_used"],
            "additionalProperties": False,
        },
    },
]

DATA_TOOL_NAMES = frozenset(t["name"] for t in M_TOOLS) - {EMIT_VERDICT}


# ── Dispatch ──────────────────────────────────────────────────────────────


def make_m_dispatch(
    variants_long: Path | str, tis_id: str, orf_type: Any = None
) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """Build the local executor for the M reader tools, bound to one isoform.

    The returned callable never raises: a bad tool name or a rejected argument
    comes back as ``{"error": ...}`` so the model can correct itself on the next
    turn instead of killing the loop.
    """
    readers: dict[str, Callable[..., dict[str, Any]]] = {
        "query_variants": query_variants,
        "variant_position_histogram": variant_position_histogram,
        "variant_effect_stats": variant_effect_stats,
    }

    def dispatch(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        reader = readers.get(name)
        if reader is None:
            return {
                "error": f"Unknown tool {name!r}. Available: {sorted(readers)} plus {EMIT_VERDICT}."
            }
        kwargs = dict(tool_input or {})
        # "any" is the schema's way of spelling "no filter"; None reads everything.
        if kwargs.get("region") == "any":
            kwargs["region"] = None
        # orf_type is bound here, never model-supplied: it selects the populated
        # position column and gates the frame caveat, both properties of the
        # annotation rather than choices the model gets to make.
        kwargs["orf_type"] = orf_type
        try:
            return reader(variants_long, tis_id, **kwargs)
        except ValueError as e:
            return {"error": str(e)}
        except TypeError as e:
            return {"error": f"Bad arguments for {name}: {e}"}
        except Exception as e:  # pragma: no cover - defensive
            return {"error": f"{name} failed: {type(e).__name__}: {e}"}

    return dispatch
