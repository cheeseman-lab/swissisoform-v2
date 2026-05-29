"""Load the paired-evidence parquet + LLM JSONs into per-gene records.

The website is a thin reader over the SwissIsoform v2 pipeline output. Data
layout under ``SWISSISOFORM_DATA_DIR`` (default ``./data``):

::

    <DATA_DIR>/all_paired.parquet     # one row per (gene, TIS)
    <DATA_DIR>/structures/*.cif       # baked AlphaFold/Boltz isoform structures
    <DATA_DIR>/llm/<gene>.json        # optional — Stage-2 interpretation

LLM JSONs are produced by a separate pipeline and may not be present yet.
Missing files degrade gracefully (the gene page renders with a placeholder).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from markupsafe import escape

logger = logging.getLogger(__name__)


# Evidence axis labels. Order is load-bearing — the templates iterate these.
EXISTENCE_CRITERIA = [
    ("E1_primate_conservation", "E1", "Primate frame conservation"),
    ("E2_mammalian_conservation", "E2", "Mammalian frame conservation"),
    ("E3_phylop_coding_selection", "E3", "PhyloP coding selection"),
    ("E4_multi_cell_line", "E4", "Detected in multiple cell lines"),
    ("E5_initiation_efficiency", "E5", "Initiation efficiency"),
    ("E6_mass_spec", "E6", "Mass-spec peptide support"),
]

FUNCTIONAL_CRITERIA = [
    ("F1_structured_extension", "F1", "Structured extension (pLDDT)"),
    ("F2_localization_change", "F2", "Localization change"),
    ("F3_domain_change", "F3", "Domain (InterProScan) change"),
    ("F4_targeting_change", "F4", "Targeting (SignalP/TargetP) change"),
    ("F5_pathogenic_variant_enrichment", "F5", "Pathogenic variant enrichment"),
    ("F6_clinical_variant_overlap", "F6", "Clinical variant overlap"),
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
    # All clinical variants in the isoform-unique region (any significance)
    variants_in_unique: list[dict[str, Any]] = field(default_factory=list)
    # Raw row (kept for /api/data.json)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneRecord:
    """One gene + all its alternative isoforms + the (optional) LLM blob."""

    name: str
    uniprot_id: str | None
    uniprot_url: str | None
    function: str | None
    location: str | None
    canonical_len: int | None
    isoforms: list[Isoform]
    llm: dict[str, Any] | None
    canonical_cif: str | None


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
    """Variant hit arrays come back as numpy arrays of dicts; normalize to list."""
    if row_value is None:
        return []
    try:
        return [dict(x) for x in row_value if x is not None]
    except TypeError:
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


# --------------------------------------------------------------------------- #
# Row -> Isoform
# --------------------------------------------------------------------------- #


def _build_isoform(row: pd.Series, struct_index: dict[tuple[str, str], str]) -> Isoform:
    gene = row["gene_name"]
    tis_id = str(row["tis_id"])

    # Pathogenic variants in the unique region — pull from the clinical hits
    # rather than the variant-intersection module so we get hgvsp + source.
    variants_all = _to_record_list(row.get("isoform_clinical_hits"))
    variants_in_unique = [v for v in variants_all if v.get("in_isoform_unique")]
    pathogenic = [
        v
        for v in variants_in_unique
        if (v.get("clinical_significance") or "").lower().startswith(("pathogenic", "likely"))
    ]

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
        isoform_cif=isoform_cif,
        canonical_cif=canonical_cif,
        raw=_clean_nan({k: row[k] for k in row.index if not k.startswith("cmp_biophysics_")}),
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

    df = pd.read_parquet(parquet_path)
    struct_index = _structure_index(str(structures_dir))

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

        isoforms = [_build_isoform(r, struct_index) for _, r in sub.iterrows()]
        out[gene_name] = GeneRecord(
            name=gene_name,
            uniprot_id=uniprot_id,
            uniprot_url=uniprot_url,
            function=_maybe_str(head.get("generef_uniprot_function")),
            location=_maybe_str(head.get("generef_subcellular_location")),
            canonical_len=_maybe_int(head.get("canonical_len")),
            isoforms=isoforms,
            llm=llm,
            canonical_cif=struct_index.get((gene_name, "canonical")),
        )

    n_iso = sum(len(g.isoforms) for g in out.values())
    logger.info("loaded %d genes (%d isoforms)", len(out), n_iso)
    return out


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


def llm_synthesis_for_isoform(*, llm_dir: Path, tis_slug: str) -> dict | None:
    """Return the per-isoform synthesis JSON, or None if missing."""
    p = Path(llm_dir) / tis_slug / "synthesis.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("narrative"):
        data["narrative_html"] = _markdown_to_html(data["narrative"])
    return data


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


# Drives the 12-tile UI grid on the isoform page (E1..E6, F1..F6).
CRITERIA_FOR_PAGE = [
    {
        "id": "E1_primate_conservation",
        "axis": "E",
        "label": "Primate conservation",
        "short_label": "Primates",
    },
    {
        "id": "E2_mammalian_conservation",
        "axis": "E",
        "label": "Mammalian conservation",
        "short_label": "Mammals",
    },
    {
        "id": "E3_phylop_coding_selection",
        "axis": "E",
        "label": "PhyloP coding selection",
        "short_label": "PhyloP",
    },
    {
        "id": "E4_multi_cell_line",
        "axis": "E",
        "label": "Multi cell line expression",
        "short_label": "Cell lines",
    },
    {
        "id": "E5_initiation_efficiency",
        "axis": "E",
        "label": "Initiation efficiency",
        "short_label": "Init. eff.",
    },
    {"id": "E6_mass_spec", "axis": "E", "label": "Mass spec", "short_label": "MS"},
    {
        "id": "F1_structured_extension",
        "axis": "F",
        "label": "Structured extension",
        "short_label": "Folding",
    },
    {
        "id": "F2_localization_change",
        "axis": "F",
        "label": "Localization change",
        "short_label": "Localization",
    },
    {"id": "F3_domain_change", "axis": "F", "label": "Domain change", "short_label": "Domains"},
    {
        "id": "F4_targeting_change",
        "axis": "F",
        "label": "Targeting change",
        "short_label": "Targeting",
    },
    {
        "id": "F5_pathogenic_variant_enrichment",
        "axis": "F",
        "label": "Pathogenic variant enrichment",
        "short_label": "Pathogenic",
    },
    {
        "id": "F6_clinical_variant_overlap",
        "axis": "F",
        "label": "Clinical variant overlap",
        "short_label": "Variants",
    },
]


def llm_criterion_for_isoform(*, llm_dir: Path, tis_slug: str, criterion_id: str) -> dict | None:
    """Return one criterion LLM read from disk, or None if missing.

    Reads ``<llm_dir>/<tis_slug>/criteria.json`` and indexes by criterion_id.
    """
    p = Path(llm_dir) / tis_slug / "criteria.json"
    if not p.exists():
        return None
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return blob.get(criterion_id)
