"""Load the paired-evidence parquet + LLM JSONs into per-gene records.

The website is a thin reader over the SwissIsoform v2 pipeline output. Data
layout under ``SWISSISOFORM_DATA_DIR`` (default ``./data``):

::

    <DATA_DIR>/all_paired.parquet            # one row per (gene, TIS)
    <DATA_DIR>/variants_long.parquet         # per-variant rows for the clinical panel
    <DATA_DIR>/transcript_skeletons.parquet  # exon skeletons for the transcript diagram
    <DATA_DIR>/structures/*.cif              # baked ESMFold2/Boltz isoform structures
    <DATA_DIR>/structures/colors/*.colors.json  # per-residue 3Dmol colouring
    <DATA_DIR>/llm/<slug>/                   # optional — per-isoform interpretation
                                             #   (synthesis.json + categories.json)

LLM JSONs are produced by a separate pipeline and may not be present yet.
Missing files degrade gracefully (the page renders with a placeholder).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from itertools import zip_longest
from pathlib import Path
from typing import Any

import pandas as pd
from markupsafe import escape

# Single source of truth for the CDLMPS category → member mapping. Lives in the
# backend evidence module (which the LLM per-category pass also consumes) and is
# copied into the site package by prepare_deploy.sh, so UI and LLM never drift.
from swissisoform.clinical.significance import is_pathogenic
from swissisoform.contract import DEFAULT_REGION_NOUN, region_noun, stratum_label
from swissisoform.site.evidence import CATEGORIES as _CATEGORIES
from swissisoform.site.evidence import active_scoring as _active_scoring
from swissisoform.site.evidence import p3_min_sse_length as _p3_min_sse_length
from swissisoform.site.evidence import p3_min_sse_plddt as _p3_min_sse_plddt
from swissisoform.site.evidence import use_scoring_config as _use_scoring_config

logger = logging.getLogger(__name__)


# Evidence axis labels. Order is load-bearing — the templates iterate these.
EXISTENCE_CRITERIA = [
    ("C1_primate_conservation", "C1", "Primate frame conservation"),
    ("C2_mammalian_conservation", "C2", "Mammalian frame conservation"),
    ("C3_phylop_coding_selection", "C3", "Coding Selection"),
    ("D1_multi_cell_line", "D1", "Expression Breadth"),
    ("D2_initiation_efficiency", "D2", "Start-Site Usage"),
    ("D3_mass_spec", "D3", "Peptide Evidence"),
]

FUNCTIONAL_CRITERIA = [
    ("L1_localization_change", "L1", "Compartment"),
    ("L2_targeting_change", "L2", "Sorting Signals"),
    ("M1_pathogenic_variant_enrichment", "M1", "Germline Variants"),
    ("M2_clinical_variant_overlap", "M2", "Clinical Variants"),
    ("P1_structured_extension", "P1", "Fold Confidence"),
    ("S1_domain_change", "S1", "Domain (InterProScan) change"),
]


@dataclass
class Isoform:
    """One row of the paired parquet, projected to the keys the templates use."""

    tis_id: str
    transcript_id: str
    chrom: str
    position: int
    strand: str
    start_codon: str
    orf_type: str
    aa_len: int
    canonical_len: int
    isoform_len: int
    differential_sequence: str
    diff_start: int
    diff_end: int
    diff_space: str
    kozak_context: str | None
    # Evidence
    existence_score: int | None
    existence_evaluable: int | None
    functional_score: int | None
    functional_evaluable: int | None
    criteria: dict[str, bool | None]
    reasons: dict[str, str]
    # Per-category LLM reads (raw ``categories.json`` blob): category name ->
    # {"reasoning": str, ...}. Files written before the verdict was dropped also
    # carry {"verdict": "interesting"|"neutral"|"not_interesting"}.
    category_verdicts: dict[str, dict[str, Any]]
    # Key metrics
    localization_canonical: str | None
    localization_isoform: str | None
    localization_changed: bool | None
    phylop_unique: float | None
    phylop_shared: float | None
    phylop_enrichment: float | None
    plddt_isoform: float | None
    plddt_diffregion: float | None
    tm_score: float | None
    rmsd_global: float | None
    n_variants_total: int | None
    n_variants_pathogenic_unique: int | None
    # Variant table (Pathogenic in unique region)
    pathogenic_variants: list[dict[str, Any]]
    # Structure lookup
    isoform_cif: str | None
    canonical_cif: str | None
    # Precomputed per-residue colour maps (dual pLDDT ramp; diff region recoloured)
    isoform_colors: str | None
    canonical_colors: str | None
    # Precomputed PAE (predicted aligned error) heatmap JSONs, or None if the
    # structure was folded before PAE capture / isn't available.
    isoform_pae: str | None
    canonical_pae: str | None
    # All clinical variants in the isoform-unique region (any significance)
    variants_in_unique: list[dict[str, Any]] = field(default_factory=list)
    # Every clinical variant over the isoform's coding region (unique + shared)
    variants_all: list[dict[str, Any]] = field(default_factory=list)
    # Controlled isoform-change tags from the synthesis pass (vocab-whitelisted);
    # displayed on the isoform dropdown row and unioned for the landing-page filter.
    tags: list[str] = field(default_factory=list)
    # Raw row (kept for /api/data.json)
    raw: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def category_flags(self) -> dict[str, dict[str, Any]]:
        """Category letter -> :func:`category_flag`, in CARD_GROUPS order.

        Cached: net_score, n_interesting, fired_categories and the index page each
        read it, several times per isoform per render, and its inputs
        (category_verdicts, criteria) are fixed when the Isoform is built.
        """
        return {
            group["letter"]: category_flag(
                group, self.category_verdicts.get(group["name"]), self.criteria
            )
            for group in CARD_GROUPS
        }

    def _verdict_counts(self) -> tuple[int, int, int]:
        """(# green, # red, # fired) over the six CDLMPS categories.

        Green is a legacy ``interesting`` verdict or, without one, a category with
        at least one criterion met; red is a legacy ``not_interesting``. Neutral,
        unmet and unscored categories do not fire.
        """
        states = [f["state"] for f in self.category_flags.values()]
        n_green = sum(s in FIRING_GREEN for s in states)
        n_red = sum(s == "not_interesting" for s in states)
        return n_green, n_red, n_green + n_red

    @property
    def n_interesting(self) -> int:
        return self._verdict_counts()[0]

    @property
    def n_fired(self) -> int:
        return self._verdict_counts()[2]

    @property
    def net_score(self) -> int:
        """Net CDLMPS score: +1 per green category, −1 per red (legacy only)."""
        n_green, n_red, _ = self._verdict_counts()
        return n_green - n_red

    @property
    def fired_categories(self) -> list[str]:
        """Category letters (CARD_GROUPS order) that fire (green or red)."""
        return [
            letter
            for letter, flag in self.category_flags.items()
            if flag["state"] in FIRING_GREEN or flag["state"] == "not_interesting"
        ]


@dataclass
class GeneRecord:
    """One gene + all its alternative isoforms + the (optional) LLM blob."""

    name: str
    uniprot_id: str | None
    uniprot_url: str | None
    function: str | None
    location: str | None
    # UniProt keywords (controlled vocabulary) — powers the functional query facet.
    keywords: list[str]
    canonical_len: int | None
    isoforms: list[Isoform]
    llm: dict[str, Any] | None
    canonical_cif: str | None

    @property
    def best_isoform(self) -> "Isoform | None":
        """The highest net-CDLMPS-score isoform — headline for the gene card.

        Ranks by net_score (green +1, red −1) descending; ties fall back to
        more green, then the existing evidence scores, then ``aa_len`` so genes
        with no scored categories still surface a deterministic representative.
        """
        if not self.isoforms:
            return None

        def _key(iso: Isoform) -> tuple[int, int, int, int]:
            return (iso.net_score, iso.n_interesting, iso.existence_score or 0, iso.aa_len or 0)

        return max(self.isoforms, key=_key)

    @property
    def iso_tags(self) -> list[str]:
        """Vocab-ordered union of the gene's isoforms' change tags.

        Drives the landing-page "Isoform change" filter facet (a gene matches when
        ANY of its isoforms carries the selected tag). Displayed per-isoform in the
        dropdown, not on the gene card itself.
        """
        present = {t for iso in self.isoforms for t in iso.tags}
        return [t for t in ISOFORM_TAG_VOCAB if t in present]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _clean_nan(obj: Any) -> Any:
    """Recursively replace NaN/inf with None for JSON serialization."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nan(v) for v in obj]
    # numpy arrays are sequences, not scalars, and must be handled BEFORE the
    # .item() branch: a size-1 object array satisfies hasattr(obj, "item") and
    # .item() returns the single element, silently turning a one-hit list into a
    # bare dict. Consumers then iterate it and get key strings instead of hits —
    # which is how a real 19-residue helix rendered as "No helix or strand in
    # region" for every isoform whose region held exactly one element. numpy
    # scalars (np.float64, np.int64) are ndim 0 and still take the branch below.
    if getattr(obj, "ndim", 0) > 0:
        return [_clean_nan(v) for v in obj]
    # pandas/numpy scalars: convert via item() if available
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return _clean_nan(obj.item())
        except (ValueError, AttributeError):
            return obj
    return obj


def _none_if_nan(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _maybe_int(v: Any) -> int | None:
    v = _none_if_nan(v)
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _maybe_float(v: Any) -> float | None:
    v = _none_if_nan(v)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _maybe_str(v: Any) -> str | None:
    v = _none_if_nan(v)
    if v is None:
        return None
    s = str(v)
    return s if s and s.lower() != "nan" else None


def _split_terms(v: Any) -> list[str]:
    """Split a ``"; "``-joined term string (e.g. generef keywords) into a list."""
    s = _maybe_str(v)
    if not s:
        return []
    return [t.strip() for t in s.split(";") if t.strip()]


def _maybe_bool(v: Any) -> bool | None:
    v = _none_if_nan(v)
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    # Pandas may give us numpy bools or strings
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
    return None


def _criteria_dict(row_value: Any) -> dict[str, bool | None]:
    """Normalize the scoring criteria dict — values are True / False / None."""
    if row_value is None or not isinstance(row_value, dict):
        return {}
    out: dict[str, bool | None] = {}
    for k, v in row_value.items():
        if v is None:
            out[k] = None
        elif isinstance(v, bool):
            out[k] = v
        else:
            # NaN slips through here when the row was parquet-encoded
            try:
                if isinstance(v, float) and not math.isfinite(v):
                    out[k] = None
                else:
                    out[k] = bool(v)
            except (TypeError, ValueError):
                out[k] = None
    return out


def _reasons_dict(row_value: Any) -> dict[str, str]:
    if row_value is None or not isinstance(row_value, dict):
        return {}
    return {k: ("" if v is None else str(v)) for k, v in row_value.items()}


def _to_record_list(row_value: Any) -> list[dict[str, Any]]:
    """Hit arrays come back as numpy arrays; normalize to a list of dict records.

    Some hit columns are arrays of plain strings (e.g. peptide sequences) rather
    than dicts — keep only the dict-shaped entries so callers never choke on a
    str/scalar element.
    """
    if row_value is None:
        return []
    try:
        return [dict(x) for x in row_value if isinstance(x, dict)]
    except (TypeError, ValueError):
        return []


# --------------------------------------------------------------------------- #
# Structure file index
# --------------------------------------------------------------------------- #

# Filenames look like:
#   <GENE>__canonical__<N>aa.cif
#   <GENE>__extended__<N>aa__<chrom>-<pos>---<codon>-<tid>.cif
#   <GENE>__truncated__<N>aa__<chrom>-<pos>---<codon>-<tid>.cif
# The "chrom-pos---codon-tid" segment is a sanitized tis_id.

_TIS_ID_RE = re.compile(r"^chr[^:]+:\d+:[+-]:[A-Z]+:.+$")


def _tis_id_to_struct_segment(tis_id: str) -> str:
    """Replicate the manifest's sanitization: ``:`` -> ``-`` and ``::`` -> ``---``.

    The structure pipeline writes ``chr17:48101353:-:CTG:ENST00000225603.9`` as
    ``chr17-48101353---CTG-ENST00000225603.9``: the strand colon collapses
    with the surrounding colons into a triple-dash, and remaining colons
    become single dashes. We reproduce that here so we can match by segment.
    """
    parts = tis_id.split(":")
    if len(parts) >= 5:
        chrom, pos, codon, tid = parts[0], parts[1], parts[3], ":".join(parts[4:])
        # parts[2] is the strand colon, which collapses with the surrounding
        # colons into "---" in the structure filenames.
        return f"{chrom}-{pos}---{codon}-{tid}"
    # Fallback: just swap colons
    return tis_id.replace(":", "-")


@lru_cache(maxsize=1)
def _structure_index(structures_dir_str: str) -> dict[tuple[str, str], str]:
    """Map ``(gene, tis_segment)`` -> isoform .cif filename.

    Canonical lookups use ``(gene, "canonical")`` as the key.
    """
    structures_dir = Path(structures_dir_str)
    index: dict[tuple[str, str], str] = {}
    if not structures_dir.is_dir():
        logger.warning("structures dir not found: %s", structures_dir)
        return index
    for cif in structures_dir.glob("*.cif"):
        name = cif.name
        parts = name[:-4].split("__")  # strip .cif
        if len(parts) < 3:
            continue
        gene = parts[0]
        kind = parts[1]  # canonical / extended / truncated
        if kind == "canonical":
            index[(gene, "canonical")] = name
        elif len(parts) >= 4:
            segment = parts[3]
            index[(gene, segment)] = name
    logger.info("structure index: %d entries", len(index))
    return index


def _lookup_isoform_cif(index: dict[tuple[str, str], str], gene: str, tis_id: str) -> str | None:
    segment = _tis_id_to_struct_segment(tis_id)
    return index.get((gene, segment))


@lru_cache(maxsize=1)
def _colors_index(colors_dir_str: str) -> frozenset[str]:
    """Set of precomputed colour-map filenames in ``<structures>/colors/``.

    Maps are named ``<gene>__<side>__<segment>.colors.json`` by
    ``scripts/export/build_folding_colors.py`` — the same gene + tis-segment used for
    CIF lookup, so the page can resolve them deterministically.
    """
    colors_dir = Path(colors_dir_str)
    if not colors_dir.is_dir():
        return frozenset()
    return frozenset(p.name for p in colors_dir.glob("*.colors.json"))


def _lookup_colors(colors: frozenset[str], gene: str, tis_id: str, side: str) -> str | None:
    name = f"{gene}__{side}__{_tis_id_to_struct_segment(tis_id)}.colors.json"
    return name if name in colors else None


@lru_cache(maxsize=1)
def _pae_index(pae_dir_str: str) -> frozenset[str]:
    """Set of precomputed PAE heatmap filenames in ``<structures>/pae/``.

    Maps are named ``<gene>__<side>__<segment>.pae.json`` by
    ``swissisoform.export.pae.export_pae`` — the same gene + tis-segment used for
    CIF / colour-map lookup, so the page resolves them deterministically.
    """
    pae_dir = Path(pae_dir_str)
    if not pae_dir.is_dir():
        return frozenset()
    return frozenset(p.name for p in pae_dir.glob("*.pae.json"))


def _lookup_pae(pae_files: frozenset[str], gene: str, tis_id: str, side: str) -> str | None:
    name = f"{gene}__{side}__{_tis_id_to_struct_segment(tis_id)}.pae.json"
    return name if name in pae_files else None


# --------------------------------------------------------------------------- #
# Row -> Isoform
# --------------------------------------------------------------------------- #


def _build_isoform(
    row: pd.Series,
    struct_index: dict[tuple[str, str], str],
    colors: frozenset[str] = frozenset(),
    pae: frozenset[str] = frozenset(),
    llm_dir: Path | None = None,
) -> Isoform:
    gene = row["gene_name"]
    tis_id = str(row["tis_id"])

    category_verdicts = (
        category_verdicts_for_isoform(llm_dir=llm_dir, tis_slug=tis_slug(tis_id))
        if llm_dir is not None
        else {}
    )

    # Pathogenic variants in the unique region — pull from the clinical hits
    # rather than the variant-intersection module so we get hgvsp + source.
    variants_all = _to_record_list(row.get("isoform_clinical_hits"))
    variants_in_unique = [v for v in variants_all if v.get("in_isoform_unique")]
    pathogenic = [v for v in variants_in_unique if is_pathogenic(v.get("clinical_significance"))]

    isoform_cif = _lookup_isoform_cif(struct_index, gene, tis_id)
    canonical_cif = struct_index.get((gene, "canonical"))

    return Isoform(
        tis_id=tis_id,
        transcript_id=_maybe_str(row.get("transcript_id")) or "",
        chrom=_maybe_str(row.get("chrom")) or "",
        position=_maybe_int(row.get("position")) or 0,
        strand=_maybe_str(row.get("strand")) or "",
        start_codon=_maybe_str(row.get("start_codon")) or "",
        orf_type=_maybe_str(row.get("orf_type")) or "",
        aa_len=_maybe_int(row.get("aa_len")) or 0,
        canonical_len=_maybe_int(row.get("canonical_len")) or 0,
        isoform_len=_maybe_int(row.get("isoform_len")) or 0,
        differential_sequence=_maybe_str(row.get("differential_sequence")) or "",
        diff_start=_maybe_int(row.get("diff_start")) or 0,
        diff_end=_maybe_int(row.get("diff_end")) or 0,
        diff_space=_maybe_str(row.get("diff_space")) or "",
        kozak_context=_maybe_str(row.get("kozak_context")),
        existence_score=_maybe_int(row.get("isoform_scoring_existence_score")),
        existence_evaluable=_maybe_int(row.get("isoform_scoring_existence_evaluable")),
        functional_score=_maybe_int(row.get("isoform_scoring_functional_score")),
        functional_evaluable=_maybe_int(row.get("isoform_scoring_functional_evaluable")),
        criteria=_criteria_dict(row.get("isoform_scoring_criteria")),
        reasons=_reasons_dict(row.get("isoform_scoring_reasons")),
        category_verdicts=category_verdicts,
        tags=synthesis_tags_for_isoform(llm_dir=llm_dir, tis_slug=tis_slug(tis_id)),
        localization_canonical=_maybe_str(row.get("cmp_localization_deeploc_prediction_canonical")),
        localization_isoform=_maybe_str(row.get("cmp_localization_deeploc_prediction_isoform")),
        localization_changed=_maybe_bool(row.get("cmp_localization_deeploc_prediction_changed")),
        phylop_unique=_maybe_float(row.get("isoform_conservation_phylop_unique_region_mean")),
        phylop_shared=_maybe_float(row.get("isoform_conservation_phylop_shared_region_mean")),
        phylop_enrichment=_maybe_float(row.get("isoform_conservation_phylop_enrichment")),
        plddt_isoform=_maybe_float(row.get("isoform_structure_plddt_isoform_mean")),
        plddt_diffregion=_maybe_float(row.get("isoform_structure_plddt_diffregion_mean")),
        tm_score=_maybe_float(row.get("isoform_structure_tm_score")),
        rmsd_global=_maybe_float(row.get("isoform_structure_rmsd_global")),
        n_variants_total=_maybe_int(row.get("isoform_variant_intersection_n_total")),
        n_variants_pathogenic_unique=_maybe_int(
            row.get("isoform_variant_intersection_n_pathogenic_in_unique_region")
        ),
        pathogenic_variants=pathogenic,
        variants_in_unique=variants_in_unique,
        variants_all=variants_all,
        isoform_cif=isoform_cif,
        canonical_cif=canonical_cif,
        isoform_colors=_lookup_colors(colors, gene, tis_id, "isoform"),
        canonical_colors=_lookup_colors(colors, gene, tis_id, "canonical"),
        isoform_pae=_lookup_pae(pae, gene, tis_id, "isoform"),
        canonical_pae=_lookup_pae(pae, gene, tis_id, "canonical"),
        raw=_clean_nan({k: row[k] for k in row.index}),
    )


# --------------------------------------------------------------------------- #
# Public load
# --------------------------------------------------------------------------- #


def data_dir() -> Path:
    """Resolve the data dir from the env var; default to ./data."""
    return Path(os.environ.get("SWISSISOFORM_DATA_DIR", "data")).resolve()


@lru_cache(maxsize=1)
def load_all() -> dict[str, GeneRecord]:
    """Load the parquet + per-gene LLM JSONs into a dict keyed by gene name.

    Cached for the lifetime of the worker — flip ``load_all.cache_clear()``
    in tests if you need to repoint at a different ``SWISSISOFORM_DATA_DIR``.
    """
    root = data_dir()
    parquet_path = root / "all_paired.parquet"
    llm_dir = root / "llm"
    structures_dir = root / "structures"

    if not parquet_path.is_file():
        logger.warning("parquet not found at %s — site will render empty.", parquet_path)
        return {}

    # Adopt the thresholds this parquet was scored with before reading anything
    # out of it: the P3 card's "qualifies" language has to match the verdicts
    # baked into the file, including when an older parquet is rendered after a
    # retune. Falls back to defaults (with a warning) for pre-sidecar runs.
    _use_scoring_config(root)

    df = pd.read_parquet(parquet_path)
    struct_index = _structure_index(str(structures_dir))
    colors = _colors_index(str(structures_dir / "colors"))
    pae = _pae_index(str(structures_dir / "pae"))

    out: dict[str, GeneRecord] = {}
    for gene_name, sub in df.groupby("gene_name", sort=True):
        gene_name = str(gene_name)
        head = sub.iloc[0]
        uniprot_id = _maybe_str(head.get("generef_uniprot_id"))
        uniprot_url = (
            f"https://www.uniprot.org/uniprotkb/{uniprot_id}/entry" if uniprot_id else None
        )

        llm_path = llm_dir / f"{gene_name}.json"
        llm: dict[str, Any] | None = None
        if llm_path.is_file():
            try:
                llm = json.loads(llm_path.read_text())
            except json.JSONDecodeError as e:
                logger.warning("failed to parse %s: %s", llm_path, e)
                llm = None

        isoforms = [
            _build_isoform(r, struct_index, colors, pae, llm_dir) for _, r in sub.iterrows()
        ]
        out[gene_name] = GeneRecord(
            name=gene_name,
            uniprot_id=uniprot_id,
            uniprot_url=uniprot_url,
            function=_maybe_str(head.get("generef_function")),
            location=_maybe_str(head.get("generef_subcellular_location")),
            keywords=_split_terms(head.get("generef_keywords")),
            canonical_len=_maybe_int(head.get("canonical_len")),
            isoforms=isoforms,
            llm=llm,
            canonical_cif=struct_index.get((gene_name, "canonical")),
        )

    n_iso = sum(len(g.isoforms) for g in out.values())
    logger.info("loaded %d genes (%d isoforms)", len(out), n_iso)
    return out


@lru_cache(maxsize=1)
def load_orf_index():
    """Load ``orf_index.parquet`` for variant→ORF resolution.

    A separate artifact from ``all_paired.parquet`` on purpose: it carries the
    *whole* catalogue's ORF coordinates in ~1.4 MB, so a VCF can be scanned
    against every annotated gene even when the deployed run displays only a
    handful. Built by ``scripts/export/build_orf_index.py``.

    Returns None when the file is absent, so the scan endpoint can report a clear
    "index not staged" error instead of silently matching nothing.
    """
    from swissisoform.variantquery.load import load_index

    path = data_dir() / "orf_index.parquet"
    if not path.is_file():
        logger.warning("orf_index.parquet missing from %s — variant scan disabled", data_dir())
        return None
    index = load_index(path)
    logger.info(
        "loaded ORF index: %d isoforms, %d genes, %d intervals, version=%s",
        index.n_isoforms,
        index.n_genes,
        index.n_intervals,
        index.version,
    )
    return index


@dataclass(frozen=True)
class TranscriptSkeleton:
    """Slim view of a transcript's exon structure for the V2 transcript graph."""

    transcript_id: str
    gene_name: str | None
    chrom: str
    strand: str
    exons: tuple[tuple[int, int], ...]
    cds_start: int | None
    cds_end: int | None
    length_nt: int
    length_aa: int | None


@lru_cache(maxsize=1)
def load_transcript_skeletons(path: Path) -> dict[str, TranscriptSkeleton]:
    """Read transcript_skeletons.parquet into a dict keyed by transcript_id."""
    if not Path(path).exists():
        return {}
    df = pd.read_parquet(path)
    out: dict[str, TranscriptSkeleton] = {}
    for _, row in df.iterrows():
        # Exons come back as a numpy array of dicts; iterate directly and
        # treat None as empty (avoid `or []` because numpy arrays don't
        # have an unambiguous truth value).
        raw_exons = row["exons"]
        exons = tuple(
            (int(e["start"]), int(e["end"])) for e in (raw_exons if raw_exons is not None else [])
        )
        out[row["transcript_id"]] = TranscriptSkeleton(
            transcript_id=row["transcript_id"],
            gene_name=_maybe_str(row.get("gene_name")),
            chrom=row["chrom"],
            strand=row["strand"],
            exons=exons,
            cds_start=_maybe_int(row.get("cds_start")),
            cds_end=_maybe_int(row.get("cds_end")),
            length_nt=int(row["length_nt"]),
            length_aa=_maybe_int(row.get("length_aa")),
        )
    return out


def _markdown_to_html(text: str) -> str:
    """Convert the tiny markdown subset our LLM emits to safe HTML.

    Escapes first (XSS-safe), then applies bold/italic/code and paragraphs.
    """
    if not text:
        return ""
    safe = str(escape(text))
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", safe)
    safe = re.sub(r"`(.+?)`", r"<code>\1</code>", safe)
    paras = [p.strip().replace("\n", "<br>") for p in safe.split("\n\n") if p.strip()]
    return "".join(f"<p>{p}</p>" for p in paras)


# Controlled isoform-change tag vocabulary. MUST stay in sync with the enum in
# scripts/site/prompts/output_schemas/synthesis.json — the website is a separate
# deployable that can't read the schema at runtime, so it's mirrored here. Tags
# are whitelisted against this set on read, so an off-vocab value (e.g. an older
# free-form synthesis.json, or model drift) can never leak into the filter facet.
ISOFORM_TAG_VOCAB: tuple[str, ...] = (
    "Localization conflict",
    "Domain loss",
    "Domain gain",
    "Truncated functional region",
    "Structured N-terminal extension",
    "Biophysical shift",
    "Core refold",
    "Interpretable-feature shift",
)
_TAG_VOCAB_SET = frozenset(ISOFORM_TAG_VOCAB)


def llm_synthesis_for_isoform(*, llm_dir: Path, tis_slug: str) -> dict | None:
    """Return the per-isoform synthesis JSON, or None if missing.

    Renders the keyed-dict prose fields (``divergence_hypothesis`` /
    ``function_relevance``) to ``*_html``; falls back to the legacy ``narrative``
    field so pre-reframe synthesis.json files still display.
    """
    p = Path(llm_dir) / tis_slug / "synthesis.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    for field_name in ("divergence_hypothesis", "function_relevance"):
        if data.get(field_name):
            data[f"{field_name}_html"] = _markdown_to_html(data[field_name])
    if data.get("narrative"):  # legacy shape
        data["narrative_html"] = _markdown_to_html(data["narrative"])
    if isinstance(data.get("tags"), list):
        data["tags"] = [t for t in data["tags"] if t in _TAG_VOCAB_SET]
    return data


def synthesis_tags_for_isoform(*, llm_dir: Path | None, tis_slug: str) -> list[str]:
    """Vocab-whitelisted isoform-change tags for one isoform (empty if none/absent)."""
    if llm_dir is None:
        return []
    p = Path(llm_dir) / tis_slug / "synthesis.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    tags = data.get("tags")
    if not isinstance(tags, list):
        return []
    # Preserve vocab order for stable display/union across isoforms.
    present = {t for t in tags if t in _TAG_VOCAB_SET}
    return [t for t in ISOFORM_TAG_VOCAB if t in present]


def tis_slug(tis_id: str) -> str:
    """URL-safe form of a tis_id (replaces ``:`` and ``.`` with ``-``)."""
    return re.sub(r"[:.]+", "-", tis_id or "unknown")


def variant_url(variant_id: str, source: str, rs_id: str | None = None) -> str | None:
    """Best-effort external link for a variant given its id + source."""
    if not variant_id:
        return None
    src = (source or "").lower()
    if "clinvar" in src and ":" in variant_id:
        return f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{variant_id.split(':')[-1]}/"
    if "gnomad" in src:
        # variant_id already looks like chr17-48071438-G-C
        slug = variant_id.replace("chr", "").replace("-", "-")
        return f"https://gnomad.broadinstitute.org/variant/{slug}"
    if "cosmic" in src:
        return f"https://cancer.sanger.ac.uk/cosmic/search?q={variant_id.split(':')[-1]}"
    if rs_id:
        return f"https://www.ncbi.nlm.nih.gov/snp/{rs_id}"
    return None


@lru_cache(maxsize=1)
def _variants_long_df(path: Path) -> pd.DataFrame:
    """Read variants_long.parquet once per worker, keyed on path."""
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def variant_rows_for_isoform(path: Path, tis_id: str) -> list[dict[str, Any]]:
    """Return every variant_long row for one isoform (no LLM 30-hit cap)."""
    df = _variants_long_df(path)
    if df.empty or "tis_id" not in df.columns:
        return []
    sub = df[df["tis_id"] == tis_id]
    rows = [_clean_nan(rec) for rec in sub.to_dict(orient="records")]
    return rows


def _sae_num(v: Any) -> float | None:
    """Coerce a numpy/py scalar to a native float, or None if missing/NaN."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(f) else f


def _sae_records(raw_val: Any) -> list[dict[str, Any]]:
    """Normalize a nested SAE list column (ndarray/list of dicts) to plain dicts."""
    if raw_val is None:
        return []
    try:
        items = list(raw_val)
    except TypeError:
        return []
    return [dict(d) for d in items if hasattr(d, "keys")]


# The SAE module already filters prevalence >= 2 and keeps the top 30 per category
# before writing the parquet, so the card only needs to distinguish generic /
# uninterpretable features (hidden until the table is expanded).
_SAE_GENERIC_LABELS = {"unknown generic feature", "unknown", ""}


def _sae_is_generic(label: Any) -> bool:
    """True when a feature label is the placeholder / uninterpretable one."""
    return (str(label).strip().lower() if label is not None else "") in _SAE_GENERIC_LABELS


def _sae_simple_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape a simple SAE feature list ({max,mean,prevalence}) into display rows.

    Used for the isoform-only, canonical-only, and differential-coordinate
    (unique-region) feature lists, which share the same per-feature schema. The
    backend already saved the top 30 at prevalence >= 2, so no filtering happens
    here — each row is tagged ``generic`` and non-generic rows are returned first
    (activation order preserved) so the card can hide generic ones until expanded.
    """
    non_generic: list[dict[str, Any]] = []
    generic: list[dict[str, Any]] = []
    for rec in records:
        idx = rec.get("feature_index")
        is_generic = _sae_is_generic(rec.get("label"))
        (generic if is_generic else non_generic).append(
            {
                "feature_index": int(idx) if idx is not None else None,
                "label": rec.get("label") or None,
                "description": rec.get("description") or None,
                "activation": _sae_num(rec.get("max")),
                "prevalence": _sae_num(rec.get("prevalence")),
                "generic": is_generic,
            }
        )
    return non_generic + generic


def _sae_shared_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape the shared-feature deltas list into display rows (already |Δ|-sorted).

    The backend already saved the top 30 at prevalence >= 2; each row is tagged
    ``generic`` and non-generic rows are returned first (|Δ| order preserved) so
    the card can hide generic ones until expanded.
    """
    non_generic: list[dict[str, Any]] = []
    generic: list[dict[str, Any]] = []
    for rec in records:
        idx = rec.get("feature_index")
        is_generic = _sae_is_generic(rec.get("label"))
        (generic if is_generic else non_generic).append(
            {
                "feature_index": int(idx) if idx is not None else None,
                "label": rec.get("label") or None,
                "description": rec.get("description") or None,
                "delta_max": _sae_num(rec.get("delta_max")),
                "isoform_max": _sae_num(rec.get("isoform_max")),
                "canonical_max": _sae_num(rec.get("canonical_max")),
                "isoform_prevalence": _sae_num(rec.get("isoform_prevalence")),
                "canonical_prevalence": _sae_num(rec.get("canonical_prevalence")),
                "generic": is_generic,
            }
        )
    return non_generic + generic


def sae_card_for_isoform(iso: "Isoform") -> dict[str, Any] | None:
    """Everything the F8 "SAE features" card needs, or None when not computed.

    Reads the ``isoform_sae_*`` columns already carried on ``iso.raw`` (populated
    by the SAE SiteModule and flattened into all_paired.parquet). Returns None
    when the SAE step did not run for this protein pair (status != "ok").

    The SAE module already saved the top 30 features per category (prevalence >= 2,
    ranked by activation / |Δ|) into the parquet, so this function only shapes them
    for display: each row is tagged ``generic`` and non-generic rows come first, so
    the card shows interpretable features by default and reveals generic ones on
    expand. Two parts:
      - Global (whole-protein feature-set comparison): ``isoform_only`` /
        ``canonical_only`` (features present in only one form) and ``shared``
        (present in both, ranked by |Δ| activation), with the saved total count.
      - Differential coordinates: ``unique_region`` — features firing on just the
        isoform-unique residues.

    Scoring and LLM surfacing are intentionally out of scope for this card — it
    is descriptive over data already in the parquet. (The scored P2 criterion is
    the separate "Core Fold Perturbation" RMSD tile.)
    """
    raw = iso.raw or {}
    if raw.get("isoform_sae_status") != "ok":
        return None

    def _int(key: str) -> int | None:
        v = raw.get(key)
        return int(v) if v is not None else None

    def _top(prefix: str) -> dict[str, Any] | None:
        idx = _int(f"isoform_sae_top_{prefix}_feature_index")
        if idx is None:
            return None
        return {
            "index": idx,
            "label": raw.get(f"isoform_sae_top_{prefix}_feature_label"),
            "delta": _sae_num(raw.get(f"isoform_sae_top_{prefix}_delta_max")),
        }

    _n_iso_only = _int("isoform_sae_n_isoform_only")
    _n_canon_only = _int("isoform_sae_n_canonical_only")
    return {
        "counts": {
            "isoform_only": _n_iso_only,
            "canonical_only": _n_canon_only,
            "shared": _int("isoform_sae_n_shared"),
        },
        # SAE features unique to one form (gained + lost) — the face tagline.
        "n_differ": (_n_iso_only or 0) + (_n_canon_only or 0),
        "mean_abs_delta_shared": _sae_num(raw.get("isoform_sae_mean_abs_delta_shared")),
        "isoform_only": _sae_simple_rows(_sae_records(raw.get("isoform_sae_features_isoform_only"))),
        "canonical_only": _sae_simple_rows(
            _sae_records(raw.get("isoform_sae_features_canonical_only"))
        ),
        "shared": _sae_shared_rows(_sae_records(raw.get("isoform_sae_shared_feature_deltas"))),
        "unique_region": _sae_simple_rows(
            _sae_records(raw.get("isoform_sae_unique_region_top_features"))
        ),
        "n_unique_region_features": _int("isoform_sae_n_unique_region_features"),
        "unique_region_space": raw.get("isoform_sae_unique_region_space"),
        "top_gained": _top("gained"),
        "top_lost": _top("lost"),
    }


# Biophysical properties shown in the standalone Biophysics card (differential vs
# shared region) — mirrors the table formerly nested in the P1 structure modal.
_BIOPHYSICS_FEATURES = [
    ("Isoelectric point (pI)", "pI"),
    ("Hydropathy (GRAVY)", "gravy"),
    ("Fraction charged", "fraction_charged"),
    ("Disorder fraction", "disorder"),
    ("Disorder-promoting", "fraction_disorder_promoting"),
    ("Low-complexity fraction", "fraction_lcr"),
    ("Prion-like fraction", "prionlike_fraction"),
    ("LLPS score", "llps_score"),
    ("π–π propensity", "pipi_propensity"),
    ("Aromaticity", "aromaticity"),
    ("Instability index", "instability_index"),
    ("Shannon entropy", "shannon_entropy"),
    ("Normalized complexity", "normalized_complexity"),
]
# Whole-protein levers the S2 scorer tests: |isoform − canonical| ≥ its cutoff.
_S2_LEVERS = [
    ("Hydropathy (GRAVY)", "gravy", "s2_gravy_delta_min"),
    ("Fraction charged", "fraction_charged", "s2_fraction_charged_delta_min"),
    ("Disorder fraction", "disorder", "s2_disorder_delta_min"),
]


def biophysics_card_for_isoform(iso: "Isoform") -> dict[str, Any] | None:
    """Standalone Biophysics card data (the S2 popup).

    Two tables in the house grammar: the whole-protein canonical-vs-isoform
    levers S2 scores (GRAVY / charge / disorder deltas), then the descriptive
    per-region profile (``cmp_biophysics_*_unique`` vs ``_shared``). Returns a
    ``ce``-shaped payload the shared ``_criterion_evidence.html`` renderer
    consumes, or ``None`` when no biophysics columns are present.
    """
    raw = getattr(iso, "raw", None) or {}
    g = raw.get
    orf_type = getattr(iso, "orf_type", None)
    cfg = _active_scoring()

    lever_rows = []
    for label, feat, cut_field in _S2_LEVERS:
        can, alt = g(f"canonical_biophysics_{feat}"), g(f"isoform_biophysics_{feat}")
        if _fmt_num(can) is None and _fmt_num(alt) is None:
            continue
        delta = _maybe_float(g(f"cmp_biophysics_{feat}_delta"))
        lever_rows.append(
            _cmp_row(
                label,
                can,
                alt,
                orf_type=orf_type,
                drives=True,
                passes=delta is not None and abs(delta) >= getattr(cfg, cut_field),
            )
        )

    region_rows = []
    for label, feat in _BIOPHYSICS_FEATURES:
        shared, unique = g(f"cmp_biophysics_{feat}_shared"), g(f"cmp_biophysics_{feat}_unique")
        if _fmt_num(shared) is None and _fmt_num(unique) is None:
            continue
        region_rows.append(_cmp_row(label, shared, unique, orf_type=orf_type))
    if not lever_rows and not region_rows:
        return None

    # Directional headline: how the differential region compares to the shared
    # core for the three properties S2 keys off. Number = (differential − core);
    # the sign gives the direction word. Region-vs-core (not the whole-protein
    # isoform−canonical delta) so small regions aren't diluted and truncations
    # read the same way as extensions.
    headline_segments: list[dict[str, Any]] = []
    plain_bits: list[str] = []
    for feat, adjective, noun, eps in (
        ("gravy", "hydrophobic", "hydropathy", 0.1),
        ("fraction_charged", "charged", "charge", 0.03),
        ("disorder", "disordered", "disorder", 0.03),
    ):
        try:
            u = float(g(f"cmp_biophysics_{feat}_unique"))
            s = float(g(f"cmp_biophysics_{feat}_shared"))
        except (TypeError, ValueError):
            continue
        if math.isnan(u) or math.isnan(s):
            continue
        diff = u - s
        if abs(diff) < eps:
            word, num = f"similar {noun}", ""
        else:
            word = f"{'more' if diff > 0 else 'less'} {adjective}"
            num = f" ({diff:+.2f})"
        if headline_segments:
            headline_segments.append({"t": " · ", "strong": False})
        headline_segments.append({"t": word, "strong": True})
        if num:
            headline_segments.append({"t": num, "strong": False})
        plain_bits.append(word + num)
    headline = (
        " · ".join(plain_bits)
        if plain_bits
        else f"{len(lever_rows) + len(region_rows)} properties"
    )

    region = _region_noun(iso)
    sections = []
    if lever_rows:
        sections.append(
            _cmp_section(
                "Biophysics · canonical vs isoform (whole protein)",
                "Property",
                _PROTEIN_SIDES,
                lever_rows,
                subtitle="the S2 levers: a shift in hydropathy, charge or disorder",
            )
        )
    if region_rows:
        sections.append(
            _cmp_section(
                f"Biophysics · {region} vs shared",
                "Property",
                ("Shared", _region_noun(iso, cap=True)),
                region_rows,
                subtitle="descriptive profile of each region",
            )
        )
    _mark_driving_rows(sections, iso, "S2_biophysics")
    return {
        "headline": headline,
        "headline_segments": headline_segments or None,
        "evidence": {
            "about": (
                "Biophysical character of the isoform versus the canonical — pI, "
                "hydropathy, charge, disorder and related properties. S2 scores the "
                "whole-protein GRAVY / charge / disorder shifts; the per-region table "
                f"below profiles the {region} against the shared core."
            ),
            "sections": sections,
        },
    }


# ── One comparison grammar for every evidence table ─────────────────────────
#
# Label | reference | subject | Ratio. The reference (Canonical, or the Shared
# core) is muted on the left, the subject (the Isoform, or the stratum's own
# region) on the right, and Ratio = subject / reference is always present ("—"
# when it does not read as a ratio). A row is highlighted only when it is a row
# the criterion's scorer tests, the criterion fired, and that row passed.

_CMP_COL_CLASSES = ["dm-shared", "", "dm-ratio"]  # reference (muted) | subject | ratio
_PROTEIN_SIDES = ("Canonical", "Isoform")
_AUTO = object()  # sentinel: compute the ratio from the two sides
_REGION_PHRASE = re.compile(r"\b(?:[Dd]ifferential|[Dd]iff|[Uu]nique)[ -]region\b")


def _region_noun(iso: Any, cap: bool = False) -> str:
    """The stratum's name for the differential region (``contract.region_noun``)."""
    return region_noun(getattr(iso, "orf_type", None), cap=cap, default=DEFAULT_REGION_NOUN)


def _regionize(text: str, orf_type: str | None) -> str:
    """Name the differential region in ``text`` the way this ORF type does.

    Normalises the site's spellings ("differential region", "diff region",
    "unique-region") onto the pipeline's "unique region", then defers to
    ``contract.stratum_label`` — the same rewrite the tag labels use.
    """
    if region_noun(orf_type) is None:
        return text
    norm = _REGION_PHRASE.sub(
        lambda m: "Unique region" if m.group(0)[0].isupper() else "unique region", text
    )
    return stratum_label(norm, orf_type)


def _ratio(subject: Any, reference: Any) -> float | None:
    """subject / reference, or None where that does not read as a ratio.

    Undefined for a non-numeric side, a reference ≤ 0 (a negative denominator
    inverts it) or a negative subject — the same guard the tag layer applies to
    ``*_ratio`` metrics.
    """
    s, r = _maybe_float(subject), _maybe_float(reference)
    if s is None or r is None or not (math.isfinite(s) and math.isfinite(r)):
        return None
    if r <= 0 or s < 0:
        return None
    return s / r


def _fmt_ratio(r: float | None) -> str:
    """Two significant figures below 10× (``0.42×``, ``3.1×``), whole numbers above."""
    if r is None or not math.isfinite(r):
        return "—"
    return f"{r:.2g}×" if r < 10 else f"{r:,.0f}×"


def _cmp_row(
    label: str,
    reference: Any,
    subject: Any,
    ratio: Any = _AUTO,
    *,
    orf_type: str | None = None,
    term: str | None = None,
    drives: bool = False,
    passes: bool = True,
    fmt: Any = None,
) -> dict[str, Any]:
    """One comparison row: ``[reference, subject, ratio]`` display cells.

    ``reference`` / ``subject`` are raw values (formatted with ``fmt``, default
    ``_fmt_num``; a str passes through). ``ratio`` defaults to subject /
    reference; pass a float to use a precomputed one, or ``None`` for a row
    where a ratio means nothing (categorical, ordinal, single-sided).
    ``drives`` marks a row the criterion's scorer tests and ``passes`` whether
    this row met its own cutoff; ``_mark_driving_rows`` turns them into ``hot``.
    """
    fmt = fmt or _fmt_num
    if ratio is _AUTO:
        ratio = _ratio(subject, reference)
    return {
        "label": label,
        "cols": [fmt(reference) or "—", fmt(subject) or "—", _fmt_ratio(ratio)],
        "drives": drives,
        "passes": bool(passes),
        **_term(term or label, orf_type),
    }


def _cmp_section(
    title: str,
    label_header: str,
    sides: tuple[str, str],
    rows: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    """A render-ready comparison section in the house grammar."""
    return {
        "title": title,
        "cmp_headers": [label_header, sides[0], sides[1], "Ratio"],
        "col_classes": _CMP_COL_CLASSES,
        "compare_rows": rows,
        **extra,
    }


def _mark_driving_rows(sections: list[dict[str, Any]], iso: Any, cid: str) -> None:
    """Resolve each row's ``drives``/``passes`` into the one ``hot`` flag.

    Hot = the criterion fired (``iso.criteria[cid] is True``) AND the row is one
    its scorer tests AND that row passed — so a highlight always reads "this is
    why the criterion fired". The verdict itself is the scorer's, never redone
    here; per-row ``passes`` only picks out which of several drivers carried it.
    """
    fired = (getattr(iso, "criteria", None) or {}).get(cid) is True
    for s in sections:
        for r in s.get("compare_rows") or []:
            drives = r.pop("drives", False)
            passes = r.pop("passes", True)
            r["hot"] = bool(fired and drives and passes)


def _fmt_num(v, pct=False):
    """Format a metric value for display, or None if missing/NaN."""
    import math

    if v is None:
        return None
    if isinstance(v, str):
        return v or None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return None
    if pct:
        return f"{f * 100:.0f}%"
    if f == int(f) and abs(f) < 1e6:
        return str(int(f))
    return f"{f:.3g}"


# Plain-English "what this means" for each criterion — shown at the top of the
# criterion modal so a reader knows how to interpret the evidence below it.
CRITERION_ABOUT = {
    "C1_primate_conservation": (
        "How well is the isoform-unique region conserved at the amino-acid level "
        "across primates (mean percent identity)? High AA identity in close "
        "relatives argues the alternative protein is real and translated, not a "
        "sequencing or annotation artifact. (Reading-frame intactness is shown "
        "as context.)"
    ),
    "C2_mammalian_conservation": (
        "The same amino-acid identity test across mammals. Conservation over "
        "deeper evolutionary distance is stronger evidence the ORF is under "
        "selection to be translated. (Reading-frame intactness is shown as "
        "context.)"
    ),
    "C3_phylop_coding_selection": (
        "phyloP / phastCons measure per-base evolutionary constraint. A high "
        "absolute phyloP over the differential region means the sequence itself "
        "is under strong purifying selection — evidence it is coding and does "
        "something. (Enrichment over the shared core is shown as context only.)"
    ),
    "D1_multi_cell_line": (
        "Is the alternative start used in more than one cell line? Reproducible "
        "initiation across independent samples argues against a one-off ribosome "
        "artifact."
    ),
    "D2_initiation_efficiency": (
        "How efficiently ribosomes initiate at this start (TIS) relative to "
        "background, per cell line. Strong, reproducible initiation supports a "
        "genuine translation event."
    ),
    "D3_mass_spec": (
        "Were tryptic peptides unique to the differential region detected by "
        "mass spec? Direct peptide evidence is the strongest proof the isoform "
        "protein exists."
    ),
    "P1_structured_extension": (
        "Does the gained/lost region fold into ordered structure (pLDDT) and "
        "shift the protein's biophysical character (charge, hydropathy, "
        "disorder)? Structured, biophysically distinct regions are more likely "
        "to be functional."
    ),
    "L1_localization_change": (
        "Do the predicted localization features (DeepLoc prediction, sorting "
        "signals, or membrane association) differ between the canonical and the "
        "isoform? A protein with changed localization features acts in a "
        "different cellular context."
    ),
    "S1_domain_change": (
        "Does the isoform gain or lose a real InterPro functional domain in the "
        "differential region (disorder/structural-only signatures excluded)? "
        "Gaining or losing a domain changes function directly."
    ),
    "L2_targeting_change": (
        "Do N-terminal targeting signals (SignalP secretion, TargetP "
        "mitochondrial/chloroplast) differ between canonical and isoform? "
        "N-terminal changes most directly add or remove targeting peptides."
    ),
    "M1_pathogenic_variant_enrichment": (
        "Does healthy human germline variation (gnomAD) avoid this region "
        "(depletion ratio < 1×), and is it intrinsically constrained (ESM-C "
        "constraint delta > 0)? Depletion of population variation plus high "
        "sequence constraint mean the region resists change — it is functionally "
        "important. Scored only where the unique region is canonical coding "
        "sequence, i.e. on truncations. gnomAD is a tolerance catalogue, not a "
        "disease one; disease/cancer variants (ClinVar / COSMIC) live in M2."
    ),
    "M2_clinical_variant_overlap": (
        "Are disease (ClinVar / COSMIC) variants enriched per nucleotide in the "
        "differential region versus the shared core (disease enrichment ratio ≥ "
        "1×)? Disease variants concentrating in the unique region tie it to "
        "phenotype."
    ),
    "P3_secondary_structure": (
        "Does the differential region contain actual secondary structure — a helix "
        "or strand — rather than coil? ESMFold2 predicts coordinates but no "
        "secondary structure, so elements are assigned from those coordinates "
        "(P-SEA, Cα geometry). Because they are derived from a prediction, each "
        "element carries its own mean pLDDT: a geometrically clean helix running "
        "through a disordered stretch is geometry fitted to a guess, so BOTH "
        "length and confidence are required to score. The direction differs by ORF "
        "type — an extension GAINS the element (a candidate functional addition), "
        "a truncation LOSES one, and there the element is read off the canonical "
        "structure because the removed segment exists only in it. This says "
        "nothing about whether the element is integrated with the rest of the "
        "fold; the contact and PAE evidence in this same category answers that."
    ),
    "P2_shared_structural_change": (
        "The shared region is the stretch of protein identical in both the isoform "
        "and the canonical (the canonical body for an extension; the post-truncation "
        "body for a truncation). Folded in both contexts it normally comes out nearly "
        "identical, so its Cα backbone RMSD ≈ 0. A high shared-region RMSD (superposed "
        "on the shared residues only, from the ESMFold2 structures) means the "
        "extension or truncation reorganizes how the retained region folds — a rare, "
        "high-interest functional signal. TM-score is a length-normalized companion; "
        "the check only fires when both structures are confidently folded, and "
        "uORF/altORF isoforms (no shared region) are not evaluable."
    ),
    "S2_biophysics": (
        "Does the whole-protein biophysical character shift between the canonical "
        "and the isoform — hydropathy (GRAVY), charged fraction, or intrinsic "
        "disorder? A meaningful shift on any of these means the alternative region "
        "changes the protein's physicochemical behaviour (solubility, membrane "
        "affinity, condensate propensity), independent of whether it folds."
    ),
    "S3_sae": (
        "Do sparse-autoencoder (SAE) interpretability features of the ESM-C protein "
        "language model differ between the isoform and the canonical — features "
        "gained or lost in the model's learned feature space? A presence check: it "
        "fires when any interpretable feature is active in one protein but not the "
        "other. The score is a binary gained-or-lost check."
    ),
}


# Per-metric glossary: label -> (about-page anchor, one-line hover definition).
# Each metric in a criterion modal links to its About section and shows this tip
# on hover. Labels must match the strings produced in criterion_evidence_for.
METRIC_GLOSSARY: dict[str, tuple[str, str]] = {
    # phyloP / phastCons
    "phyloP mean": ("m-phylop", "Mean phyloP evolutionary constraint over the region (higher = more conserved)."),
    "phastCons mean": ("m-phylop", "Mean phastCons probability the region sits in a conserved element."),
    "phyloP at TIS codon": ("m-phylop", "phyloP score at the 3-nt start codon."),
    "phyloP Kozak window": ("m-phylop", "Mean phyloP over the Kozak window (−9..+4)."),
    "phastCons at TIS codon": ("m-phylop", "phastCons score at the start codon."),
    "phastCons Kozak window": ("m-phylop", "Mean phastCons over the Kozak window."),
    "phyloP at start codon": ("m-phylop", "phyloP score at the 3-nt start codon."),
    "phastCons at start codon": ("m-phylop", "phastCons score at the 3-nt start codon."),
    "phyloP over Kozak window": ("m-phylop", "Mean phyloP over the Kozak window (−9..+4)."),
    "phastCons over Kozak window": ("m-phylop", "Mean phastCons over the Kozak window (−9..+4)."),
    "Start codon": ("m-initiation", "The start codon (ATG or a near-cognate) at this initiation site."),
    "Kozak mismatch — full consensus": ("m-initiation", "Mismatches to the full Kozak consensus (0 = perfect)."),
    # Frame conservation
    "Frame intact (fraction of species)": ("m-frame", "Fraction of aligned species keeping an intact reading frame."),
    "Species aligned": ("m-frame", "Number of species with an alignment over the differential ORF."),
    "Species frame-intact": ("m-frame", "Number of species whose reading frame stays intact."),
    "Start codon conserved": ("m-frame", "Fraction of species conserving the start codon."),
    "Mean AA identity": ("m-frame", "Mean amino-acid identity across aligned species."),
    "Deepest intact species": ("m-frame", "Most evolutionarily distant species whose frame is still intact."),
    "Phylo depth (MRCA)": ("m-frame", "Phylogenetic depth of the deepest frame-intact species."),
    # Initiation / Kozak
    "Kozak context (−9..+4)": ("m-initiation", "Sequence window around the start codon (−9..+4)."),
    "Kozak Hamming — full consensus": ("m-initiation", "Mismatches to the full Kozak consensus."),
    "Kozak Hamming — major positions": ("m-initiation", "Mismatches at the key −3 and +4 Kozak positions."),
    "Kozak Hamming — partial": ("m-initiation", "Mismatches at the partial Kozak consensus positions."),
    "Kozak window GC content": ("m-initiation", "GC fraction of the Kozak window."),
    # Structure / ESMFold2
    "pLDDT (whole protein)": ("m-structure", "Mean ESMFold2 per-residue confidence (0–1) over the whole protein."),
    "pLDDT — differential region": ("m-structure", "Mean ESMFold2 confidence over the differential region only."),
    "pLDDT std — differential region": ("m-structure", "Spread of pLDDT within the differential region."),
    "pLDDT Δ (diff vs shared)": ("m-structure", "Differential-minus-shared mean pLDDT."),
    "TM-score (iso vs canonical)": ("m-structure", "Global fold similarity (0–1) between isoform and canonical."),
    "RMSD global (Å)": ("m-structure", "Backbone RMSD between isoform and canonical folds."),
    "Extension contacts": ("m-structure", "Inter-residue contacts the differential region makes with the core."),
    "Shared-region Cα RMSD": ("m-structure", "Backbone RMSD over the shared (identical) residues only, after superposing on them — 0 ≈ identical fold, high = the retained region refolds."),
    "Shared-region TM-score": ("m-structure", "Length-normalized fold similarity (0–1) of the shared region between isoform and canonical."),
    "Shared region length": ("m-structure", "Number of residues shared (identical) between the isoform and canonical proteins."),
    "Shared-region pLDDT": ("m-structure", "Mean ESMFold2 confidence over the shared residues, in each protein's own fold."),
    "Min shared-region pLDDT": ("m-structure", "Lower of the two mean shared-region ESMFold2 confidences; the RMSD is only trusted (scored) when this is high."),
    # Localization / targeting
    "Predicted location": ("m-localization", "DeepLoc predicted subcellular compartment."),
    "Sorting signals": ("m-localization", "DeepLoc-detected sorting / targeting signals."),
    "Membrane": ("m-localization", "DeepLoc membrane-association call."),
    "TargetP": ("m-localization", "TargetP N-terminal presequence (mitochondrial / chloroplast / none)."),
    "SignalP": ("m-localization", "SignalP secretory signal-peptide call."),
    # Domains / motifs / MS
    "InterPro domains": ("m-domains", "Count of annotated InterPro domains on the canonical vs the isoform protein."),
    "Short linear motifs": ("m-domains", "Count of short linear motifs on the canonical vs the isoform protein."),
    "InterPro domains in diff region": ("m-domains", "Annotated InterPro domains overlapping the differential region."),
    "Motifs in diff region": ("m-domains", "Short linear motifs in the differential region."),
    "Domains changed in region": ("m-domains", "Real (InterPro) domains gained, lost or cut by the differential region — the S1 basis."),
    "Tryptic peptides (in-silico)": ("m-massspec", "Predicted tryptic peptides from each protein (in-silico digest, not MS-detected)."),
    "Validated by mass-spec": ("m-massspec", "Peptides confirmed against real spectra by PepQuery."),
    "Isoform-unique peptides": ("m-massspec", "Peptides that map only to the isoform — direct existence evidence."),
    "Validated isoform-unique peptides": ("m-massspec", "Isoform-unique peptides PepQuery confirmed against real spectra — the D3 basis."),
    "Mass-spec peptides in diff region": ("m-massspec", "Detected tryptic peptides unique to the differential region."),
    # Constraint / variant effect
    "ESM-C mean LLR": ("m-esm2", "Mean ESM-C masked-marginal log-likelihood ratio; lower = more constrained."),
    "Constraint Δ": ("m-esm2", "How much more constrained (ESM-C) the differential region is than the shared core."),
    "Constrained positions": ("m-esm2", "Count of residues the language model flags as highly constrained."),
    "Mean ΔLLR — unique-region variants": ("m-esm2", "Mean ESM-C ΔLLR across variants in the unique region."),
    "Min ΔLLR — unique-region variants": ("m-esm2", "Most-constrained ESM-C ΔLLR among unique-region variants."),
    "Variants ESM-C-scored": ("m-esm2", "Number of variants scored by ESM-C."),
    "AlphaMissense-pathogenic in unique": ("m-alphamissense", "AlphaMissense-pathogenic missense variants in the unique region."),
    "Mean AlphaMissense — unique": ("m-alphamissense", "Mean AlphaMissense pathogenicity in the unique region."),
    "Variants AlphaMissense-scored": ("m-alphamissense", "Number of variants scored by AlphaMissense."),
    "Scorable variants in unique region": ("m-clinical", "Variants in the unique region that could be scored."),
    "Damaging variants in unique region": ("m-clinical", "Variants called damaging by AlphaMissense or ESM-C."),
    "Scorable variants": ("m-clinical", "Variants in the differential region that could be scored (ESM-C, AlphaMissense, or LoF)."),
    "Damaging variants": ("m-clinical", "Variants called damaging — AlphaMissense-pathogenic, ESM-C-constrained, or loss-of-function."),
    "— of which loss-of-function": ("m-clinical", "Damaging variants that are frameshift / stop-gained / splice / start-lost (missense predictors never see these)."),
    "AlphaMissense-pathogenic": ("m-alphamissense", "AlphaMissense-pathogenic missense variants in the differential region."),
    "Mean ΔLLR (ESM-C)": ("m-esm2", "Mean ESM-C ΔLLR across variants in the differential region (lower = more constrained)."),
    "Min ΔLLR (ESM-C)": ("m-esm2", "Most-constrained ESM-C ΔLLR among differential-region variants."),
    "Mean AlphaMissense": ("m-alphamissense", "Mean AlphaMissense pathogenicity in the differential region."),
    # Biophysics
    "Isoelectric point (pI)": ("m-biophysics", "pH at which the region carries no net charge."),
    "Hydropathy (GRAVY)": ("m-biophysics", "Grand average of hydropathy; positive = hydrophobic."),
    "Fraction charged": ("m-biophysics", "Fraction of charged residues (D/E/K/R)."),
    "Disorder fraction": ("m-biophysics", "Fraction of residues predicted intrinsically disordered."),
    "Disorder-promoting": ("m-biophysics", "Fraction of disorder-promoting residues."),
    "Low-complexity fraction": ("m-biophysics", "Fraction in low-complexity (repetitive) sequence."),
    "Prion-like fraction": ("m-biophysics", "Fraction in prion-like (Q/N-rich) sequence."),
    "LLPS score": ("m-biophysics", "Propensity to drive liquid–liquid phase separation."),
    "π–π propensity": ("m-biophysics", "Planar π-contact propensity (a phase-separation driver)."),
    "Aromaticity": ("m-biophysics", "Fraction of aromatic residues (F/W/Y)."),
    "Instability index": ("m-biophysics", "Predicted in-vivo instability; >40 suggests unstable."),
    "Shannon entropy": ("m-biophysics", "Compositional entropy (sequence diversity)."),
    "Normalized complexity": ("m-biophysics", "Compositional complexity normalized to length."),
    # Clinical burden
    "All variants": ("m-clinical", "ClinVar / gnomAD / COSMIC variants intersecting each region."),
    "Disease variants": ("m-clinical", "ClinVar + COSMIC (disease/cancer) variants in each region — gnomAD (population/tolerance) is excluded; it feeds M1's constraint, not disease burden."),
    "gnomAD variants": ("m-clinical", "gnomAD (healthy-population/germline) variants per nucleotide per region. Depletion in the differential region (depletion ratio <1×) = it resists germline variation = constrained."),
    "Pathogenic": ("m-clinical", "Pathogenic / likely-pathogenic variants in each region."),
    "Clinical/observed variants": ("m-clinical", "ClinVar / gnomAD / COSMIC variants intersecting the region."),
    "Pathogenic variants": ("m-clinical", "Pathogenic / likely-pathogenic variants in the region."),
    "Clinical variants in diff region": ("m-clinical", "Clinical-database variants inside the differential region."),
}


# P-SEA reports bare "helix"/"strand"; the UI names the actual structure types.
# helix/strand only: sse_elements emits an element solely for _STRUCTURED runs,
# so coil residues never reach here as elements (test_coil_is_never_an_element).
_SSE_LABEL = {"helix": "alpha helix", "strand": "beta strand"}


def _term(label: str, orf_type: str | None = None) -> dict[str, str]:
    """Glossary tip + about-page anchor for a metric label (empty if unknown).

    The tip names the differential region the way ``orf_type`` does.
    """
    ent = METRIC_GLOSSARY.get(label)
    if not ent:
        return {}
    anchor, tip = ent
    return {"tip": _regionize(tip, orf_type), "anchor": anchor}


def criterion_evidence_for(iso) -> dict:
    """Per-criterion differential evidence for the score modals (C1–D3, L, M, P, S1).

    Pulls the conservation / structure / localization / PLM-VEP / function /
    clinical evidence off the parquet row, sliced finer than a flat modality
    grouping (primate vs mammalian frame, DeepLoc vs targeting) and keyed by
    criterion id so each score popup hosts exactly its own evidence + a
    plain-English descriptor.

    Every comparison table uses the one house grammar (``_cmp_section``):
    ``Label | Canonical | Isoform | Ratio`` for whole-protein / per-start
    comparisons, ``Label | Shared | <region> | Ratio`` for region comparisons,
    where <region> is the stratum's own name (Extension / Lost region / ORF).
    Rows the criterion's scorer tests are highlighted when it fired.

    Returns ``{criterion_id: {"about": str, "sections": [section, ...]}}`` where
    every section is render-ready for ``_criterion_evidence.html``.
    """
    raw = getattr(iso, "raw", None) or {}
    g = raw.get
    orf_type = getattr(iso, "orf_type", None)
    diff_space = getattr(iso, "diff_space", "") or ""
    is_trunc = diff_space == "canonical" or (orf_type or "") == "truncated"
    cfg = _active_scoring()
    region = _region_noun(iso)  # "extension" / "lost region" / "ORF"
    REGION_SIDES = ("Shared", _region_noun(iso, cap=True))

    def row(label, reference, subject, ratio=_AUTO, **kw):
        return _cmp_row(label, reference, subject, ratio, orf_type=orf_type, **kw)

    def keyed(label, ref_key, subj_key, ratio=_AUTO, **kw):
        """A row read off two columns, or None when both sides are missing."""
        ref = g(ref_key) if ref_key else None
        subj = g(subj_key) if subj_key else None
        if _fmt_num(ref) is None and _fmt_num(subj) is None:
            return None
        return row(label, ref, subj, ratio, **kw)

    def section(title, label_header, sides, rows, **extra):
        rows = [r for r in rows if r]
        if not rows:
            return None
        return _cmp_section(title, label_header, sides, rows, **extra)

    def pct(v):
        return _fmt_num(v, pct=True)

    # ---- modality section builders (each → section dict or None) ----------

    def sec_frame(clade):
        # Canonical-ORF frame vs differential-ORF frame across the clade. The
        # canonical twin columns (``..._canonical_*``) score the full canonical
        # ORF; the bare columns score the isoform-unique region only.
        pre = f"isoform_conservation_frame_{clade}"
        specs = [
            # (label, suffix, percent, ratio meaningful, drives)
            ("Mean AA identity", "mean_pident", True, True, True),
            ("Frame intact (fraction of species)", "frac_intact", True, True, False),
            ("Species aligned", "n_species_aligned", False, True, False),
            ("Species frame-intact", "n_species_intact_frame", False, True, False),
            ("Start codon conserved", "start_codon_conserved", True, True, False),
            ("Deepest intact species", "deepest_species", False, False, False),
            ("Phylo depth (MRCA)", "max_depth", False, False, False),
        ]
        rows = [
            keyed(
                label,
                f"{pre}_canonical_{suf}",
                f"{pre}_{suf}",
                _AUTO if has_ratio else None,
                fmt=pct if is_pct else None,
                drives=drives,
            )
            for label, suf, is_pct, has_ratio, drives in specs
        ]
        return section(
            f"Sequence conservation · {clade}",
            "Conservation metric",
            _PROTEIN_SIDES,
            rows,
            subtitle=_regionize(
                "canonical ORF vs the isoform's differential ORF — mean AA identity is "
                "the score basis; frame intactness across the clade is context",
                orf_type,
            ),
        )

    def sec_phylop():
        rows = [
            keyed(
                "phyloP mean",
                "isoform_conservation_phylop_shared_region_mean",
                "isoform_conservation_phylop_unique_region_mean",
                drives=True,
            ),
            keyed(
                "phastCons mean",
                "isoform_conservation_phastcons_shared_region_mean",
                "isoform_conservation_phastcons_unique_region_mean",
            ),
        ]
        return section(
            f"Per-base conservation · {region} vs shared",
            "Track",
            REGION_SIDES,
            rows,
            subtitle=(
                f"signed mean phyloP over the {region} is the score basis (high = strong "
                "purifying selection, negative = accelerated); the shared column and ratio "
                "are context"
            ),
        )

    def sec_start_site():
        # Canonical-start vs alternative-start: initiation context + per-base
        # conservation at each start codon. The canonical column reads
        # ``canonical_*`` mirrors. One row per property.
        specs = [
            ("Start codon", "canonical_start_codon", "start_codon", False),
            (
                "Kozak context (−9..+4)",
                "canonical_initiation_context_kozak_context",
                "isoform_initiation_context_kozak_context",
                False,
            ),
            (
                "phyloP at start codon",
                "canonical_conservation_phylop_at_tis",
                "isoform_conservation_phylop_at_tis",
                True,
            ),
            (
                "phastCons at start codon",
                "canonical_conservation_phastcons_at_tis",
                "isoform_conservation_phastcons_at_tis",
                True,
            ),
            (
                "phyloP over Kozak window",
                "canonical_conservation_phylop_kozak_mean",
                "isoform_conservation_phylop_kozak_mean",
                True,
            ),
            (
                "phastCons over Kozak window",
                "canonical_conservation_phastcons_kozak_mean",
                "isoform_conservation_phastcons_kozak_mean",
                True,
            ),
            (
                "Kozak mismatch — full consensus",
                "canonical_initiation_context_kozak_hamming_full",
                "isoform_initiation_context_kozak_hamming_full",
                False,
            ),
            (
                "Kozak window GC content",
                "canonical_initiation_context_kozak_window_gc_content",
                "isoform_initiation_context_kozak_window_gc_content",
                True,
            ),
        ]
        if all(_fmt_num(g(alt_key)) is None for _, _, alt_key, _ in specs):
            return None
        rows = [
            row(label, g(can_key), g(alt_key), _AUTO if has_ratio else None)
            for label, can_key, alt_key, has_ratio in specs
        ]
        return section(
            "Start site · canonical vs isoform",
            "Property",
            _PROTEIN_SIDES,
            rows,
            subtitle="initiation context + per-base conservation at each start codon",
        )

    CELL_LINES = ("HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen")

    def sec_cell_lines():
        # Canonical-start vs alternative-start usage (CPM) per cell line. A line
        # counts toward D1 when the alternative start was called there (it carries
        # a p-value) — the p-value rides in the Isoform cell.
        rows = []
        for cl in CELL_LINES:
            can_cpm, alt_cpm = g(f"canonical_expr_{cl}_cpm"), g(f"expr_{cl}_cpm")
            pval = _fmt_num(g(f"expr_{cl}_p_value"))
            if _fmt_num(can_cpm) is None and _fmt_num(alt_cpm) is None and pval is None:
                continue
            alt_cell = _fmt_num(alt_cpm) or "—"
            if pval is not None:
                alt_cell = f"{alt_cell} (p {pval})"
            rows.append(
                row(
                    cl.replace("_", " "),
                    can_cpm,
                    alt_cell,
                    _ratio(alt_cpm, can_cpm),
                    drives=True,
                    passes=pval is not None,
                )
            )
        q = _fmt_num(g("fisher_qvalue"))
        return section(
            "Per-cell-line usage · canonical vs isoform",
            "Cell line",
            _PROTEIN_SIDES,
            rows,
            subtitle=(
                "start CPM per cell line; p = the alternative-start call in that line"
                + (f" · Fisher q = {q}" if q else "")
            ),
        )

    def sec_efficiency():
        # Canonical-start vs this alternative-start initiation efficiency, per
        # cell line. Canonical efficiency comes from the per-Tid Annotated row
        # (``canonical_expr_*``); the alternative start from ``expr_*``.
        cut = cfg.initiation_efficiency_min
        rows = []
        for cl in CELL_LINES:
            alt = _maybe_float(g(f"expr_{cl}_initiation_efficiency"))
            rows.append(
                keyed(
                    cl.replace("_", " "),
                    f"canonical_expr_{cl}_initiation_efficiency",
                    f"expr_{cl}_initiation_efficiency",
                    drives=True,
                    passes=alt is not None and alt >= cut,
                )
            )
        return section(
            "Start-Site Usage · canonical vs isoform",
            "Cell line",
            _PROTEIN_SIDES,
            rows,
            subtitle=(
                "ribosome initiation efficiency at the canonical start vs this "
                "alternative start, per cell line"
            ),
        )

    def sec_structure():
        # Whole-protein pLDDT, canonical vs isoform. Global fold-similarity
        # singletons (TM-score, RMSD, interface contacts) have no second side —
        # they are the canonical-vs-isoform comparison expressed as one number —
        # so they ride along in the caption rather than a Details list.
        status = g("isoform_structure_status")
        bits = []
        tm = _fmt_num(g("isoform_structure_tm_score"))
        rmsd = _fmt_num(g("isoform_structure_rmsd_global"))
        con = _fmt_num(g("isoform_structure_extension_contacts"))
        if tm is not None:
            bits.append(f"TM-score {tm}")
        if rmsd is not None:
            bits.append(f"RMSD {rmsd} Å")
        if con is not None:
            bits.append(f"{con} interface contacts")
        sub = " · ".join(bits) if bits else (f"status: {status}" if status else None)
        warn = None
        if status == "uniform_plddt":
            warn = (
                "Structure metrics are a uniform-pLDDT placeholder for this "
                "isoform — treat the per-residue confidence as unreliable."
            )
        return section(
            "Structure (ESMFold2) · canonical vs isoform",
            "Metric",
            _PROTEIN_SIDES,
            [
                keyed(
                    "pLDDT (whole protein)",
                    "isoform_structure_plddt_canonical_mean",
                    "isoform_structure_plddt_isoform_mean",
                )
            ],
            subtitle=sub,
            warn=warn,
        )

    def sec_structure_region():
        # Per-region fold confidence: differential region vs shared core — the P1
        # basis. The shared-region pLDDT isn't stored, but Δ (= diff − shared) is,
        # so we recover it (no pipeline change).
        diff_raw = _maybe_float(g("isoform_structure_plddt_diffregion_mean"))
        if diff_raw is None or not math.isfinite(diff_raw):
            return None
        delta_raw = _maybe_float(g("isoform_structure_plddt_delta_shared"))
        shared_raw = (
            diff_raw - delta_raw if delta_raw is not None and math.isfinite(delta_raw) else None
        )
        return section(
            f"Fold confidence · {region} vs shared",
            "Metric",
            REGION_SIDES,
            [row("pLDDT", shared_raw, diff_raw, term="pLDDT — differential region", drives=True)],
            subtitle="mean ESMFold2 pLDDT in each region",
        )

    def _sse_split():
        """Whole-protein SSE elements split into (unique, shared) lists.

        Reads ``isoform_structure_sse_all_elements``, where each element is
        tagged ``unique`` / ``shared`` / ``spans`` by the pipeline. A ``spans``
        element crosses the boundary and counts for the isoform: it has
        secondary structure in the unique region, which is what the card asks.
        Its full span is shown, so the extent into the shared core is visible
        without spending a line of the tile saying so.
        """
        raw_all = g("isoform_structure_sse_all_elements")
        els = [e for e in (raw_all if raw_all is not None else []) if isinstance(e, dict)]
        unique = [e for e in els if e.get("region") in ("unique", "spans")]
        shared = [e for e in els if e.get("region") == "shared"]
        return unique, shared

    def _sse_qualifies(e):
        """The P3 test: long enough AND confident enough.

        Thresholds come from ScoringConfig via site.evidence — the same values
        the scorer uses — so retuning the config moves this modal, the tile
        headline and the verdict together instead of leaving them disagreeing.
        """
        return (e.get("length") or 0) >= _p3_min_sse_length() and (
            e.get("plddt_mean") or 0
        ) >= _p3_min_sse_plddt()

    def _sse_stat(els, kind):
        return sum(1 for e in els if e.get("type") == kind)

    def _sse_longest(els):
        return max((e.get("length") or 0 for e in els), default=0)

    def _sse_mean_plddt(els):
        vals = [e["plddt_mean"] for e in els if e.get("plddt_mean") is not None]
        return sum(vals) / len(vals) if vals else None

    def sec_sse_summary():
        if g("isoform_structure_sse_status") != "ok":
            return None
        unique, shared = _sse_split()
        if not unique and not shared:
            return None
        # Counts describe the elements that COUNT — the same qualifying set the
        # tile headline and the P3 verdict use. Counting every element here
        # instead put two different answers to "how many" on one card (TRIP13:
        # headline 1, table 1 helix + 1 strand, because its 14 aa helix sits at
        # pLDDT 0.63). The sub-threshold ones are listed in their own section
        # below rather than dropped.
        unique = [e for e in unique if _sse_qualifies(e)]
        shared = [e for e in shared if _sse_qualifies(e)]

        def sse_row(label, fn, drives=False):
            u, s = fn(unique), fn(shared)
            return row(label, s, u, drives=drives, passes=bool(u))

        return section(
            f"Secondary structure · {region} vs shared",
            "Metric",
            REGION_SIDES,
            [
                sse_row("Alpha helices", lambda e: _sse_stat(e, "helix"), drives=True),
                sse_row("Beta strands", lambda e: _sse_stat(e, "strand"), drives=True),
                sse_row("Longest element (aa)", _sse_longest),
                sse_row("Mean pLDDT", _sse_mean_plddt),
            ],
            subtitle="helices and strands assigned from the predicted coordinates (P-SEA)",
        )

    def sec_sse_list(qualifying=True):
        """Elements with their coordinates, in two columns.

        Split into two sections so every count on the card agrees: the first
        lists what the headline and P3 count, the second the ones that fall
        short. Both keep the region | shared layout.
        """
        if g("isoform_structure_sse_status") != "ok":
            return None
        unique, shared = _sse_split()
        keep = _sse_qualifies if qualifying else (lambda e: not _sse_qualifies(e))
        unique = [e for e in unique if keep(e)]
        shared = [e for e in shared if keep(e)]
        if not unique and not shared:
            return None

        def cell(e):
            if e is None:
                return None
            span = f"{e.get('start')}–{e.get('end')}"
            bits = [f"{e.get('length')} aa"]
            if e.get("plddt_mean") is not None:
                bits.append(f"pLDDT {e['plddt_mean']:.2f}")
            kind = e.get("type") or ""
            return {
                # `kind` keys the CSS class and stays the bare P-SEA type name;
                # `label` is what the badge reads.
                "kind": kind,
                "label": _SSE_LABEL.get(kind, kind),
                "span": span,
                "detail": " · ".join(bits),
            }

        pairs = [
            {"left": cell(u), "right": cell(s)}
            for u, s in zip_longest(unique, shared, fillvalue=None)
        ]
        counts = f"{len(unique)} in the {region}, {len(shared)} in the shared core"
        if qualifying:
            title = "Elements and coordinates"
            subtitle = f"{counts} — residue numbering is 1-based on the protein holding the region"
        else:
            title = "Below threshold"
            # Named for both reasons: across cheeseman_test 62 of 69 sub-threshold
            # elements are short rather than low-confidence, so calling the
            # section "low confidence" would mislabel most of it.
            # Interpolated, not written out: this sentence states the same rule
            # _sse_qualifies applies, so a retune must move both together.
            subtitle = (
                f"{counts} — shorter than {_p3_min_sse_length()} aa or below "
                f"pLDDT {_p3_min_sse_plddt():.2f}, so not counted above"
            )
        return {
            "title": title,
            "subtitle": subtitle,
            # On a truncation the region exists only in the canonical protein, so
            # the column is named for the stratum, never "isoform-unique".
            "pair_headers": [_region_noun(iso, cap=True), "Shared core"],
            "pairs": pairs,
        }

    def sec_shared_rmsd():
        # P2 — strict shared-region Cα RMSD (Kabsch-superposed on the shared
        # residues only), the score basis, gated by shared length and pLDDT. RMSD
        # is single-valued (it IS the comparison), so it sits in the Isoform
        # column with no canonical side; shared pLDDT is genuinely two-sided.
        status = g("isoform_structure_rmsd_shared_status")
        rows = [
            keyed(
                "Shared-region Cα RMSD",
                None,
                "isoform_structure_rmsd_shared",
                None,
                drives=True,
            ),
            keyed(
                "Shared-region pLDDT",
                "isoform_structure_plddt_shared_mean_canonical",
                "isoform_structure_plddt_shared_mean_isoform",
            ),
        ]
        bits = []
        tm = _fmt_num(g("isoform_structure_tm_score_shared"))
        n = g("isoform_structure_shared_region_len")
        if tm is not None:
            bits.append(f"shared TM-score {tm}")
        if n is not None:
            bits.append(f"shared region {int(n)} aa")
        gtm = _fmt_num(g("isoform_structure_tm_score"))
        grmsd = _fmt_num(g("isoform_structure_rmsd_global"))
        if gtm is not None:
            bits.append(f"global TM-score {gtm}")
        if grmsd is not None:
            bits.append(f"global RMSD {grmsd} Å")

        if not bits and not any(rows):
            if status and status != "ok":
                _why = {
                    "no_shared_region": "no shared region (uORF/altORF, or region too short)",
                    "unverified_alignment": "isoform/canonical alignment unverified",
                    "no_cache": "predicted structures not available",
                }.get(status, status)
                return {
                    "title": "Shared-region structural change",
                    "subtitle": f"not evaluable — {_why}",
                }
            return None
        sub = "Cα RMSD (Å) superposed on the shared residues only; TM-score is length-normalized"
        if bits:
            sub += " · " + " · ".join(bits)
        sec = section(
            "Shared-region structural change", "Property", _PROTEIN_SIDES, rows, subtitle=sub
        )
        return sec or {"title": "Shared-region structural change", "subtitle": sub}

    def sec_deeploc():
        rows = [
            keyed(
                label,
                f"cmp_localization_deeploc_{feat}_canonical",
                f"cmp_localization_deeploc_{feat}_isoform",
                None,
                drives=True,
                passes=bool(g(f"cmp_localization_deeploc_{feat}_changed")),
            )
            for label, feat in (
                ("Predicted location", "prediction"),
                ("Sorting signals", "signals"),
                ("Membrane", "membrane"),
            )
        ]
        changed = any(
            bool(g(f"cmp_localization_deeploc_{feat}_changed"))
            for feat in ("prediction", "signals", "membrane")
        )
        return section(
            "Subcellular localization (DeepLoc)",
            "Property",
            _PROTEIN_SIDES,
            rows,
            subtitle=(
                "localization features changed (prediction/signals/membrane)"
                if changed
                else "no localization-feature change"
            ),
        )

    def sec_targeting():
        rows = [
            keyed(
                label,
                f"canonical_{tool}_{tool}_prediction",
                f"isoform_{tool}_{tool}_prediction",
                None,
                drives=True,
                passes=bool(g(f"cmp_{tool}_{tool}_prediction_changed")),
            )
            for label, tool in (("TargetP", "targetp"), ("SignalP", "signalp"))
        ]
        changed = any(
            bool(g(f"cmp_{tool}_{tool}_prediction_changed")) for tool in ("targetp", "signalp")
        )
        return section(
            "N-terminal targeting (SignalP / TargetP)",
            "Predictor",
            _PROTEIN_SIDES,
            rows,
            subtitle="sorting signal changed" if changed else "no sorting-signal change",
        )

    def _named_features(col, kind):
        """{signature_name: display_label} for an interproscan / motifs column."""
        out = {}
        for h in _to_record_list(g(col)):
            nm = h.get("name") or h.get("match") or h.get("description")
            if not nm:
                continue
            desc = h.get("description") or h.get("interpro_description")
            label = desc if (kind == "domain" and desc and desc not in ("-", "—")) else nm
            out[str(nm)] = str(label)[:60]
        return out

    def sec_domains_motifs():
        can_dom = _named_features("canonical_interproscan_hits", "domain")
        iso_dom = _named_features("isoform_interproscan_hits", "domain")
        can_mot = _named_features("canonical_motifs_hits", "motif")
        iso_mot = _named_features("isoform_motifs_hits", "motif")
        n_changed = _maybe_int(g("cmp_interproscan_n_real_domains_changed_in_diff_region"))
        if not (can_dom or iso_dom or can_mot or iso_mot) and n_changed is None:
            return None
        rows = [
            # The S1 basis: real domains gained, lost or cut by the region. One
            # number describing the comparison, so it has no canonical side.
            row(
                f"Domains changed in {region}",
                None,
                n_changed,
                None,
                term="Domains changed in region",
                drives=True,
            )
            if n_changed is not None
            else None,
            row("InterPro domains", len(can_dom), len(iso_dom)),
            row("Short linear motifs", len(can_mot), len(iso_mot)),
        ]
        hits = []
        for nm in sorted(set(iso_dom) - set(can_dom)):
            hits.append({"kind": "gained", "name": iso_dom[nm], "span": "domain"})
        for nm in sorted(set(can_dom) - set(iso_dom)):
            hits.append({"kind": "lost", "name": can_dom[nm], "span": "domain"})
        for nm in sorted(set(iso_mot) - set(can_mot)):
            hits.append({"kind": "gained", "name": iso_mot[nm], "span": "motif"})
        for nm in sorted(set(can_mot) - set(iso_mot)):
            hits.append({"kind": "lost", "name": can_mot[nm], "span": "motif"})
        return section(
            "Domains & motifs · canonical vs isoform",
            "Feature",
            _PROTEIN_SIDES,
            rows,
            subtitle="gained = only in the isoform; lost = only in the canonical",
            hits=hits,
        )

    def sec_massspec():
        can = _to_record_list(g("canonical_massspec_hits"))
        iso_hits = _to_record_list(g("isoform_massspec_hits"))
        if not can and not iso_hits:
            return None
        can_val = sum(1 for p in can if p.get("validated"))
        iso_val = sum(1 for p in iso_hits if p.get("validated"))
        iso_unique = [p for p in iso_hits if p.get("unique_to_isoform")]
        unique_val = sum(1 for p in iso_unique if p.get("validated"))
        rows = [
            row("Tryptic peptides (in-silico)", len(can), len(iso_hits)),
            row("Validated by mass-spec", can_val, iso_val),
            row("Isoform-unique peptides", None, len(iso_unique), None),
            # The D3 basis: isoform-unique AND spectrum-validated.
            row(
                "Validated isoform-unique peptides",
                None,
                unique_val,
                None,
                drives=True,
                passes=unique_val > 0,
            ),
        ]
        hits = []
        for p in iso_unique[:20]:
            pos, end = p.get("pos"), p.get("end")
            span = f"{pos}–{end}" if pos is not None and end is not None else ""
            kind = "validated" if p.get("validated") else "peptide"
            hits.append({"kind": kind, "name": str(p.get("peptide") or "?")[:40], "span": span})
        return section(
            "Peptide evidence · canonical vs isoform",
            "Feature",
            _PROTEIN_SIDES,
            rows,
            subtitle="isoform-unique peptides are direct evidence the alternative protein exists",
            hits=hits,
        )

    def _region_lengths_nt():
        """(unique, shared) region lengths in nt for density ratios.

        Emitted by the variant_intersection module; falls back to AA length × 3
        only when the nt columns predate the row. The shared region is whichever
        whole protein the region is shared with (canonical for extensions,
        isoform for truncations).
        """
        unique_nt = _maybe_int(g("isoform_variant_intersection_unique_region_nt"))
        shared_nt = _maybe_int(g("isoform_variant_intersection_shared_region_nt"))
        unique_len = unique_nt or (
            (getattr(iso, "diff_end", None) or len(getattr(iso, "differential_sequence", "") or ""))
            * 3
        )
        shared_aa = (
            getattr(iso, "isoform_len", None) if is_trunc else getattr(iso, "canonical_len", None)
        )
        shared_len = shared_nt or ((shared_aa or 0) * 3)
        return unique_len, shared_len

    def _density_ratio(n_unique, n_shared, lengths):
        """Per-length density ratio, region ÷ shared (None when undefined)."""
        unique_len, shared_len = lengths
        if n_unique is None or n_shared is None or not unique_len or not shared_len:
            return None
        return _ratio(n_unique / unique_len, n_shared / shared_len)

    def sec_germline_tolerance():
        # gnomAD = germline population variation, a *tolerance* readout. Depletion
        # in the region vs the shared core = healthy human variation avoids it =
        # the region is constrained. Ratio < 1× = constrained; > 1× = tolerated.
        # (Disease variants live in M2.)
        n_u = _maybe_float(g("isoform_variant_intersection_n_gnomad_in_unique_region"))
        n_s = _maybe_float(g("isoform_variant_intersection_n_gnomad_in_shared_region"))
        # M1 reads only the module's depletion ratio; a ratio worked out here from
        # the counts is shown for context but never marks the row as M1's basis.
        depletion = _maybe_float(g("isoform_variant_intersection_gnomad_depletion_ratio"))
        if n_u is None and n_s is None and depletion is None:
            return None
        scored = depletion is not None
        if depletion is None:
            depletion = _density_ratio(n_u, n_s, _region_lengths_nt())
        return section(
            f"Germline variants · {region} vs shared",
            "Variant set",
            REGION_SIDES,
            [
                row(
                    "gnomAD variants",
                    n_s,
                    n_u,
                    depletion,
                    drives=True,
                    passes=scored and depletion < cfg.m1_depletion_ratio_max,
                )
            ],
            subtitle=(
                "gnomAD (population) variants; ratio = density per nucleotide, "
                f"{region} ÷ shared — below {cfg.m1_depletion_ratio_max:g}× means "
                "healthy human variation avoids the region (constrained), an M1 basis"
            ),
        )

    def sec_constraint():
        # Region-resolved sequence constraint (ESM-C): is the region more
        # constrained than the shared core? The Δ is the M1 lever; LLRs are
        # negative, so their ratio is undefined and shows "—".
        delta = _maybe_float(g("isoform_plm_vep_constraint_delta"))
        # M1 drops the Δ unless the PLM VEP run is ok, so only then can it pass.
        plm_ok = g("isoform_plm_vep_status") == "ok"
        return section(
            f"Sequence constraint (ESM-C) · {region} vs shared",
            "Property",
            REGION_SIDES,
            [
                keyed(
                    "ESM-C mean LLR",
                    "isoform_plm_vep_mean_llr_shared_region",
                    "isoform_plm_vep_mean_llr_unique_region",
                ),
                row(
                    "Constraint Δ",
                    None,
                    delta,
                    None,
                    drives=True,
                    passes=plm_ok and delta >= cfg.m1_constraint_delta_min,
                )
                if delta is not None
                else None,
                keyed(
                    "Constrained positions",
                    "isoform_plm_vep_n_constrained_positions_shared",
                    "isoform_plm_vep_n_constrained_positions_unique",
                ),
            ],
            subtitle=(
                f"lower LLR = more constrained; Δ = how much more constrained the {region} "
                f"is than the shared core (an M1 basis at ≥ {cfg.m1_constraint_delta_min:g})"
            ),
        )

    def sec_variant_burden(source, pool_label):
        # Predicted-damaging variant counts for one source pool (§4): gnomad
        # (germline → M1) or disease (ClinVar+COSMIC → M2). The predictors are
        # source-independent; this counts their calls over the chosen pool, region
        # vs shared, with a length-normalized ratio. Context, not a score basis.
        unique_aa = getattr(iso, "diff_end", None) or len(
            getattr(iso, "differential_sequence", "") or ""
        )
        shared_aa = (
            getattr(iso, "isoform_len", None) if is_trunc else getattr(iso, "canonical_len", None)
        )
        metrics = [
            ("Scorable variants", "n_scorable_in"),
            ("Damaging variants", "n_damaging_in"),
            ("— of which loss-of-function", "n_lof_in"),
            ("AlphaMissense-pathogenic", "n_am_pathogenic_in"),
        ]
        if all(
            _fmt_num(g(f"isoform_varianteffect_{stem}_unique_{source}")) is None
            for _, stem in metrics
        ):
            return None
        rows = []
        for label, stem in metrics:
            n_u = _maybe_float(g(f"isoform_varianteffect_{stem}_unique_{source}"))
            n_s = _maybe_float(g(f"isoform_varianteffect_{stem}_shared_{source}"))
            rows.append(row(label, n_s, n_u, _density_ratio(n_u, n_s, (unique_aa, shared_aa))))
        return section(
            f"Predicted-damaging variants · {pool_label}",
            "Variant set",
            REGION_SIDES,
            rows,
            subtitle=(
                f"AlphaMissense / ESM-C / LoF calls per region; ratio = density, "
                f"{region} ÷ shared"
            ),
        )

    def sec_variant_scores(source, pool_label):
        # Per-variant predictor scores for one source pool (§4), region vs shared.
        # Predictor coverage (overall #scored) rides in the caption.
        metrics = [
            ("Mean ΔLLR (ESM-C)", "mean_delta_llr"),
            ("Min ΔLLR (ESM-C)", "min_delta_llr"),
            ("Mean AlphaMissense", "mean_am_pathogenicity"),
        ]
        if all(
            _fmt_num(g(f"isoform_varianteffect_{stem}_unique_{source}")) is None
            for _, stem in metrics
        ):
            return None
        rows = [
            row(
                label,
                g(f"isoform_varianteffect_{stem}_shared_{source}"),
                g(f"isoform_varianteffect_{stem}_unique_{source}"),
            )
            for label, stem in metrics
        ]
        cov = []
        plm = _fmt_num(g("isoform_varianteffect_n_scored_plm"))
        am = _fmt_num(g("isoform_varianteffect_n_scored_am"))
        if plm is not None:
            cov.append(f"{plm} ESM-C")
        if am is not None:
            cov.append(f"{am} AlphaMissense")
        return section(
            f"Predictor scores · {pool_label}",
            "Property",
            REGION_SIDES,
            rows,
            subtitle="scored: " + " · ".join(cov) if cov else f"{pool_label} variants",
        )

    def sec_clinical_burden():
        disease_ratio = _maybe_float(g("isoform_variant_intersection_disease_enrichment_ratio"))
        lengths = _region_lengths_nt()
        rows = []
        for label, stem, drives in [
            ("Disease variants", "disease", True),
            ("Pathogenic", "pathogenic", False),
        ]:
            n_u = _maybe_float(g(f"isoform_variant_intersection_n_{stem}_in_unique_region"))
            n_s = _maybe_float(g(f"isoform_variant_intersection_n_{stem}_in_shared_region"))
            if n_u is None and n_s is None:
                continue
            # The module's disease enrichment ratio is the M2 basis; the other
            # rows use the locally computed density ratio.
            r = (
                disease_ratio
                if drives and disease_ratio is not None
                else _density_ratio(n_u, n_s, lengths)
            )
            rows.append(row(label, n_s, n_u, r, drives=drives))
        return section(
            f"Clinical-variant burden · {region} vs shared",
            "Variant set",
            REGION_SIDES,
            rows,
            subtitle=(
                f"counts per region; ratio = density per nucleotide, {region} ÷ shared — "
                f"≥ {cfg.m2_disease_enrichment_min:g}× means disease variants concentrate "
                f"in the {region}, the M2 basis"
            ),
        )

    # ---- assign sections to criteria --------------------------------------
    assignments = {
        "C1_primate_conservation": [sec_frame("primate")],
        "C2_mammalian_conservation": [sec_frame("mammalian")],
        "C3_phylop_coding_selection": [sec_phylop(), sec_start_site()],
        "D1_multi_cell_line": [sec_cell_lines()],
        "D2_initiation_efficiency": [sec_efficiency()],
        "D3_mass_spec": [sec_massspec()],
        "P1_structured_extension": [sec_structure(), sec_structure_region()],
        "P3_secondary_structure": [
            sec_sse_summary(),
            sec_sse_list(qualifying=True),
            sec_sse_list(qualifying=False),
        ],
        "L1_localization_change": [sec_deeploc()],
        "S1_domain_change": [sec_domains_motifs()],
        "L2_targeting_change": [sec_targeting()],
        "M1_pathogenic_variant_enrichment": [
            sec_germline_tolerance(),
            sec_constraint(),
            sec_variant_burden("gnomad", "germline (gnomAD)"),
            sec_variant_scores("gnomad", "germline (gnomAD)"),
        ],
        "M2_clinical_variant_overlap": [
            sec_clinical_burden(),
            sec_variant_burden("disease", "disease (ClinVar/COSMIC)"),
            sec_variant_scores("disease", "disease (ClinVar/COSMIC)"),
        ],
        "P2_shared_structural_change": [sec_shared_rmsd()],
    }

    out = {}
    for cid, secs in assignments.items():
        secs = [s for s in secs if s]
        _mark_driving_rows(secs, iso, cid)
        out[cid] = {
            # Left generic: it describes the criterion across every stratum
            # ("scored only on truncations"), so per-isoform nouns would misread.
            "about": CRITERION_ABOUT.get(cid, ""),
            "sections": secs,
        }
    return out


def _isoform_view(iso, gene):
    """Adapter — turns an Isoform + GeneRecord into the dict the V2 template uses.

    The V1 Isoform dataclass uses ``isoform_len`` / ``canonical_len`` and stashes
    high_confidence flags in ``raw`` (the parquet row). The V2 template wants
    ``isoform_length_aa`` / ``canonical_length_aa`` and explicit
    ``*_high_confidence`` fields, so we translate here.
    """
    raw = getattr(iso, "raw", None) or {}
    return {
        "tis_id": iso.tis_id,
        "transcript_id": getattr(iso, "transcript_id", None),
        "gene_name": gene.name,
        "uniprot_id": getattr(gene, "uniprot_id", None),
        "orf_type": getattr(iso, "orf_type", None),
        "isoform_length_aa": getattr(iso, "isoform_len", None),
        "canonical_length_aa": getattr(iso, "canonical_len", None),
        "existence_score": getattr(iso, "existence_score", None),
        "existence_evaluable": getattr(iso, "existence_evaluable", None),
        "existence_high_confidence": raw.get("isoform_scoring_existence_high_confidence"),
        "functional_score": getattr(iso, "functional_score", None),
        "functional_evaluable": getattr(iso, "functional_evaluable", None),
        "functional_high_confidence": raw.get("isoform_scoring_functional_high_confidence"),
    }


def llm_for_isoform(gene: GeneRecord, tis_id: str) -> dict[str, Any] | None:
    """Pluck the per-isoform LLM narrative out of the gene-level LLM blob.

    The LLM JSON schema isn't fully nailed down yet. We try the two shapes the
    spec implies — a top-level ``isoforms`` dict keyed by ``tis_id``, or a
    list of dicts with a ``tis_id`` field — and fall back to ``None``.
    """
    if not gene.llm:
        return None
    raw = gene.llm.get("isoforms")
    if isinstance(raw, dict):
        return raw.get(tis_id)
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict) and entry.get("tis_id") == tis_id:
                return entry
    return None


# Drives the evidence-tile UI grid on the isoform page (C1..D3, P1..P2).
CRITERIA_FOR_PAGE = [
    {
        "id": "C1_primate_conservation",
        "axis": "E",
        "label": "Primate conservation",
        "short_label": "Primates",
    },
    {
        "id": "C2_mammalian_conservation",
        "axis": "E",
        "label": "Mammalian conservation",
        "short_label": "Mammals",
    },
    {
        "id": "C3_phylop_coding_selection",
        "axis": "E",
        "label": "Coding Selection",
        "short_label": "PhyloP",
    },
    {
        "id": "D1_multi_cell_line",
        "axis": "E",
        "label": "Expression Breadth",
        "short_label": "Cell lines",
    },
    {
        "id": "D2_initiation_efficiency",
        "axis": "E",
        "label": "Start-Site Usage",
        "short_label": "Init. eff.",
    },
    {
        "id": "D3_mass_spec",
        "axis": "E",
        "label": "Peptide Evidence",
        "short_label": "MS",
    },
    {
        "id": "P1_structured_extension",
        "headline_label": "pLDDT Differential Region",
        "axis": "F",
        "label": "Fold Confidence",
        "short_label": "Folding",
    },
    {
        "id": "L1_localization_change",
        "axis": "F",
        "label": "Compartment",
        "short_label": "Localization",
    },
    {
        "id": "S1_domain_change",
        "axis": "F",
        "label": "Domain change",
        "short_label": "Domains",
    },
    {
        "id": "L2_targeting_change",
        "axis": "F",
        "label": "Sorting Signals",
        "short_label": "Targeting",
    },
    {
        "id": "M1_pathogenic_variant_enrichment",
        "axis": "F",
        "label": "Germline Variants",
        "short_label": "Germline",
    },
    {
        "id": "M2_clinical_variant_overlap",
        "axis": "F",
        "label": "Clinical Variants",
        "short_label": "Variants",
    },
    {
        "id": "P2_shared_structural_change",
        "headline_label": "Shared-Region RMSD",
        "axis": "F",
        "label": "Core Fold Perturbation",
        "short_label": "Shared RMSD",
    },
    {
        "id": "P3_secondary_structure",
        "axis": "F",
        "label": "Secondary Structure",
        "short_label": "SSE",
    },
    # S2/S3 are first-class scored criteria (biophysics + SAE); the page renders
    # them via their bespoke cards (_biophysics_card.html / _sae_card.html), but
    # they must appear here to stay in sync with the backend CRITERIA registry.
    {
        "id": "S2_biophysics",
        "axis": "F",
        "label": "Biophysics",
        "short_label": "Biophysics",
    },
    {
        "id": "S3_sae",
        "axis": "F",
        "label": "SAE features",
        "short_label": "SAE",
    },
]


# Thematic card groups — the six-way organization that replaces the old E/F axis
# on the isoform page. Order is the display order. ``members`` reference
# ``CRITERIA_FOR_PAGE`` ids, plus the literals "biophysics" / "sae" for the two
# descriptive (non-scored) tiles. Each group's members are internally uniform in
# the old axis, so existing per-tile axis colouring stays coherent.
# Derived from the backend CATEGORIES (same {name, letter, members} shape) so the
# grouped tile layout and the per-category LLM pass share one definition.
CARD_GROUPS = [
    {"name": c["name"], "letter": c["letter"], "members": list(c["members"])}
    for c in _CATEGORIES
]

# member id -> displayed badge (group letter + 1-based index within its group).
# Covers the 13 criterion ids plus "biophysics" / "sae".
CARD_BADGES = {
    member: f"{group['letter']}{i + 1}"
    for group in CARD_GROUPS
    for i, member in enumerate(group["members"])
}

# by-id view of CRITERIA_FOR_PAGE, for the grouped template loop.
CRITERIA_BY_ID = {c["id"]: c for c in CRITERIA_FOR_PAGE}


# Verdicts written by the category pass before #41 dropped the field. Still
# honoured, so an LLM directory that predates the change renders as it did.
LEGACY_VERDICT_LABELS = {
    "interesting": "Interesting",
    "neutral": "Neutral",
    "not_interesting": "Not interesting",
}

# Flag states that light a lineup letter green.
FIRING_GREEN = frozenset({"interesting", "met"})


def category_flag(
    group: dict[str, Any], read: dict[str, Any] | None, criteria: dict[str, bool | None]
) -> dict[str, Any]:
    """The state shown for one category on its tile and lineup letter.

    A legacy ``verdict`` in the category read wins. Without one the state comes
    from the category's own criterion scores in the parquet, not from the LLM:
    ``met`` when any member criterion is True, ``unmet`` when members were
    evaluated and none is, ``pending`` when none could be evaluated.
    """
    values = [criteria.get(m) for m in group["members"]]
    n_met = sum(v is True for v in values)
    n_evaluable = sum(v is not None for v in values)
    verdict = (read or {}).get("verdict")
    if verdict in LEGACY_VERDICT_LABELS:
        state, label = verdict, LEGACY_VERDICT_LABELS[verdict]
    elif not n_evaluable:
        state, label = "pending", "Not assessed"
    else:
        state = "met" if n_met else "unmet"
        label = f"{n_met}/{n_evaluable} criteria met"
    return {"state": state, "label": label, "n_met": n_met, "n_evaluable": n_evaluable}


def category_verdicts_for_isoform(*, llm_dir: Path, tis_slug: str) -> dict:
    """Per-category LLM read, keyed by CARD_GROUPS category name.

    Reads ``<llm_dir>/<tis_slug>/categories.json`` — an object keyed by category
    ``name`` (e.g. "Conservation") → ``{"reasoning": str, ...}``, produced by the
    ``category`` LLM pass (``llm.py``); tags arms add ``tags_fired`` and older
    files a ``verdict``. Returns ``{}`` when the file is absent or unreadable, so
    the front end falls back to a placeholder reasoning line.
    """
    p = Path(llm_dir) / tis_slug / "categories.json"
    if not p.exists():
        return {}
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return blob if isinstance(blob, dict) else {}
