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
                                             #   (synthesis.json + criteria.json)

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
    ("F5_pathogenic_variant_enrichment", "F5", "Germline tolerance & constraint"),
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
    # Precomputed per-residue colour maps (dual pLDDT ramp; diff region recoloured)
    isoform_colors: str | None
    canonical_colors: str | None
    # All clinical variants in the isoform-unique region (any significance)
    variants_in_unique: list[dict[str, Any]] = field(default_factory=list)
    # Every clinical variant over the isoform's coding region (unique + shared)
    variants_all: list[dict[str, Any]] = field(default_factory=list)
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


# --------------------------------------------------------------------------- #
# Row -> Isoform
# --------------------------------------------------------------------------- #


def _build_isoform(
    row: pd.Series,
    struct_index: dict[tuple[str, str], str],
    colors: frozenset[str] = frozenset(),
) -> Isoform:
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
        variants_all=variants_all,
        isoform_cif=isoform_cif,
        canonical_cif=canonical_cif,
        isoform_colors=_lookup_colors(colors, gene, tis_id, "isoform"),
        canonical_colors=_lookup_colors(colors, gene, tis_id, "canonical"),
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

    df = pd.read_parquet(parquet_path)
    struct_index = _structure_index(str(structures_dir))
    colors = _colors_index(str(structures_dir / "colors"))

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

        isoforms = [_build_isoform(r, struct_index, colors) for _, r in sub.iterrows()]
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


# Display filters requested by collaborators: drop uninterpretable features and
# features that fire on only a single residue (noise). Applied at shaping time so
# the card tables show only interpretable, robustly-present features.
_SAE_MIN_PREVALENCE = 2
_SAE_GENERIC_LABELS = {"unknown generic feature", "unknown", ""}


def _sae_is_generic(label: Any) -> bool:
    """True when a feature label is the placeholder / uninterpretable one."""
    return (str(label).strip().lower() if label is not None else "") in _SAE_GENERIC_LABELS


def _sae_prevalent(v: Any) -> bool:
    """True when a prevalence value clears the min-prevalence display threshold."""
    n = _sae_num(v)
    return n is not None and n >= _SAE_MIN_PREVALENCE


def _sae_simple_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape a simple SAE feature list ({max,mean,prevalence}) into display rows.

    Used for the isoform-only, canonical-only, and differential-coordinate
    (unique-region) feature lists, which share the same per-feature schema. Drops
    generic-labelled features and any firing on < 2 residues.
    """
    out: list[dict[str, Any]] = []
    for rec in records:
        if _sae_is_generic(rec.get("label")) or not _sae_prevalent(rec.get("prevalence")):
            continue
        idx = rec.get("feature_index")
        out.append(
            {
                "feature_index": int(idx) if idx is not None else None,
                "label": rec.get("label") or None,
                "description": rec.get("description") or None,
                "activation": _sae_num(rec.get("max")),
                "prevalence": _sae_num(rec.get("prevalence")),
            }
        )
    return out


def _sae_shared_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape the shared-feature deltas list into display rows (already |Δ|-sorted).

    Drops generic-labelled features and features firing on < 2 residues in both
    forms (a feature prevalent in either the isoform or canonical is kept).
    """
    out: list[dict[str, Any]] = []
    for rec in records:
        if _sae_is_generic(rec.get("label")):
            continue
        if not (
            _sae_prevalent(rec.get("isoform_prevalence"))
            or _sae_prevalent(rec.get("canonical_prevalence"))
        ):
            continue
        idx = rec.get("feature_index")
        out.append(
            {
                "feature_index": int(idx) if idx is not None else None,
                "label": rec.get("label") or None,
                "description": rec.get("description") or None,
                "delta_max": _sae_num(rec.get("delta_max")),
                "isoform_max": _sae_num(rec.get("isoform_max")),
                "canonical_max": _sae_num(rec.get("canonical_max")),
                "isoform_prevalence": _sae_num(rec.get("isoform_prevalence")),
                "canonical_prevalence": _sae_num(rec.get("canonical_prevalence")),
            }
        )
    return out


def sae_card_for_isoform(iso: "Isoform") -> dict[str, Any] | None:
    """Everything the F8 "SAE features" card needs, or None when not computed.

    Reads the ``isoform_sae_*`` columns already carried on ``iso.raw`` (populated
    by the SAE SiteModule and flattened into all_paired.parquet). Returns None
    when the SAE step did not run for this protein pair (status != "ok").

    Two parts of the card:
      - Global (whole-protein feature-set comparison): ``isoform_only`` /
        ``canonical_only`` (features present in only one form) and ``shared``
        (present in both, ranked by |Δ| activation) — each capped at top 30, with
        the saved total count for the full category.
      - Differential coordinates: ``unique_region`` — features firing on just the
        isoform-unique residues (top 30).

    Scoring and LLM surfacing are intentionally out of scope for this card — it
    is descriptive over data already in the parquet. (The scored F7 criterion is
    the separate "Shared region structural changes" RMSD tile.)
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

    return {
        "counts": {
            "isoform_only": _int("isoform_sae_n_isoform_only"),
            "canonical_only": _int("isoform_sae_n_canonical_only"),
            "shared": _int("isoform_sae_n_shared"),
        },
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
    "E1_primate_conservation": (
        "How well is the isoform-unique region conserved at the amino-acid level "
        "across primates (mean percent identity)? High AA identity in close "
        "relatives argues the alternative protein is real and translated, not a "
        "sequencing or annotation artifact. (Reading-frame intactness is shown "
        "as context.)"
    ),
    "E2_mammalian_conservation": (
        "The same amino-acid identity test across mammals. Conservation over "
        "deeper evolutionary distance is stronger evidence the ORF is under "
        "selection to be translated. (Reading-frame intactness is shown as "
        "context.)"
    ),
    "E3_phylop_coding_selection": (
        "phyloP / phastCons measure per-base evolutionary constraint. A high "
        "absolute phyloP over the differential region means the sequence itself "
        "is under strong purifying selection — evidence it is coding and does "
        "something. (Enrichment over the shared core is shown as context only.)"
    ),
    "E4_multi_cell_line": (
        "Is the alternative start used in more than one cell line? Reproducible "
        "initiation across independent samples argues against a one-off ribosome "
        "artifact."
    ),
    "E5_initiation_efficiency": (
        "How efficiently ribosomes initiate at this start (TIS) relative to "
        "background, per cell line. Strong, reproducible initiation supports a "
        "genuine translation event."
    ),
    "E6_mass_spec": (
        "Were tryptic peptides unique to the differential region detected by "
        "mass spec? Direct peptide evidence is the strongest proof the isoform "
        "protein exists."
    ),
    "F1_structured_extension": (
        "Does the gained/lost region fold into ordered structure (pLDDT) and "
        "shift the protein's biophysical character (charge, hydropathy, "
        "disorder)? Structured, biophysically distinct regions are more likely "
        "to be functional."
    ),
    "F2_localization_change": (
        "Do the predicted localization features (DeepLoc prediction, sorting "
        "signals, or membrane association) differ between the canonical and the "
        "isoform? A protein with changed localization features acts in a "
        "different cellular context."
    ),
    "F3_domain_change": (
        "Does the isoform gain or lose a real InterPro functional domain in the "
        "differential region (disorder/structural-only signatures excluded)? "
        "Gaining or losing a domain changes function directly."
    ),
    "F4_targeting_change": (
        "Do N-terminal targeting signals (SignalP secretion, TargetP "
        "mitochondrial/chloroplast) differ between canonical and isoform? "
        "N-terminal changes most directly add or remove targeting peptides."
    ),
    "F5_pathogenic_variant_enrichment": (
        "Does healthy human germline variation (gnomAD) avoid this region "
        "(depletion ratio < 1×), and is it intrinsically constrained (ESM-C "
        "constraint enrichment)? Depletion of population variation plus high "
        "sequence constraint mean the region resists change — it is functionally "
        "important. gnomAD is a tolerance catalogue, not a disease one; "
        "disease/cancer variants (ClinVar / COSMIC) live in F6."
    ),
    "F6_clinical_variant_overlap": (
        "Are disease (ClinVar / COSMIC) variants enriched per nucleotide in the "
        "differential region versus the shared core (disease enrichment ratio ≥ "
        "1×)? Disease variants concentrating in the unique region tie it to "
        "phenotype."
    ),
    "F7_shared_structural_change": (
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
    "Tryptic peptides (in-silico)": ("m-massspec", "Predicted tryptic peptides from each protein (in-silico digest, not MS-detected)."),
    "Validated by mass-spec": ("m-massspec", "Peptides confirmed against real spectra by PepQuery."),
    "Isoform-unique peptides": ("m-massspec", "Peptides that map only to the isoform — direct existence evidence."),
    "Mass-spec peptides in diff region": ("m-massspec", "Detected tryptic peptides unique to the differential region."),
    # Constraint / variant effect
    "ESM-C mean LLR": ("m-esm2", "Mean ESM-C masked-marginal log-likelihood ratio; lower = more constrained."),
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
    "Disease variants": ("m-clinical", "ClinVar + COSMIC (disease/cancer) variants in each region — gnomAD (population/tolerance) is excluded; it feeds F5's constraint, not disease burden."),
    "gnomAD variants": ("m-clinical", "gnomAD (healthy-population/germline) variants per nucleotide per region. Depletion in the differential region (depletion ratio <1×) = it resists germline variation = constrained."),
    "Pathogenic": ("m-clinical", "Pathogenic / likely-pathogenic variants in each region."),
    "Clinical/observed variants": ("m-clinical", "ClinVar / gnomAD / COSMIC variants intersecting the region."),
    "Pathogenic variants": ("m-clinical", "Pathogenic / likely-pathogenic variants in the region."),
    "Clinical variants in diff region": ("m-clinical", "Clinical-database variants inside the differential region."),
}


def _term(label: str) -> dict[str, str]:
    """Glossary tip + about-page anchor for a metric label (empty if unknown)."""
    ent = METRIC_GLOSSARY.get(label)
    if not ent:
        return {}
    anchor, tip = ent
    return {"tip": tip, "anchor": anchor}


def criterion_evidence_for(iso) -> dict:
    """Per-criterion differential evidence for the score modals (E1–E6, F1–F7).

    Pulls the conservation / structure / biophysics / localization / PLM-VEP /
    function / clinical evidence off the parquet row, sliced finer than a flat
    modality grouping (primate vs mammalian frame, DeepLoc vs targeting) and
    keyed by criterion id so each score popup hosts exactly its own evidence +
    a plain-English descriptor.

    Returns ``{criterion_id: {"about": str, "sections": [section, ...]}}`` where
    every section is render-ready for ``_criterion_evidence.html``.
    """
    raw = getattr(iso, "raw", None) or {}
    g = raw.get
    diff_space = getattr(iso, "diff_space", "") or ""
    is_trunc = diff_space == "canonical" or (getattr(iso, "orf_type", "") or "") == "truncated"

    def rows(specs, pct_keys=()):
        out = []
        for label, key in specs:
            val = _fmt_num(g(key), pct=key in pct_keys)
            if val is not None:
                out.append({"label": label, "value": val, **_term(label)})
        return out

    def compare(specs):
        """Generic N-column compare table.

        Each spec is ``(label, [key, key, ...], hot_key?)``. A ``None`` key
        renders ``—``. Rows where every column is missing are dropped. Pairs
        with a ``hot_key`` flag rows for visual emphasis (enriched biophysics).
        """
        out = []
        for spec in specs:
            label, keys = spec[0], spec[1]
            hot_key = spec[2] if len(spec) > 2 else None
            vals = [_fmt_num(g(k)) if k else None for k in keys]
            if all(v is None for v in vals):
                continue
            out.append(
                {
                    "label": label,
                    "cols": [v if v is not None else "—" for v in vals],
                    "hot": bool(g(hot_key)) if hot_key else False,
                    **_term(label),
                }
            )
        return out

    # Per-column CSS classes for the two comparison framings. First entry styles
    # the column *after* the row-label column.
    DIFF_SHARED_3 = ["", "dm-shared", "dm-ratio"]  # differential | shared | enrichment
    DIFF_SHARED_2 = ["", "dm-shared"]  # differential | shared
    CANON_ISO = ["dm-shared", ""]  # canonical (muted) | isoform (focus)

    # ---- modality section builders (each → section dict or None) ----------

    def sec_frame(clade):
        # Canonical-ORF frame vs differential-ORF frame across the clade. The
        # canonical twin columns (``..._canonical_*``) score the full canonical
        # ORF; the bare columns score the isoform-unique region only.
        pre = f"isoform_conservation_frame_{clade}"
        specs = [
            ("Mean AA identity", "mean_pident", True),
            ("Frame intact (fraction of species)", "frac_intact", True),
            ("Species aligned", "n_species_aligned", False),
            ("Species frame-intact", "n_species_intact_frame", False),
            ("Start codon conserved", "start_codon_conserved", True),
            ("Deepest intact species", "deepest_species", False),
            ("Phylo depth (MRCA)", "max_depth", False),
        ]
        compare_rows = []
        for label, suf, pct in specs:
            can = _fmt_num(g(f"{pre}_canonical_{suf}"), pct=pct)
            iso = _fmt_num(g(f"{pre}_{suf}"), pct=pct)
            if can is None and iso is None:
                continue
            compare_rows.append(
                {
                    "label": label,
                    "cols": [can or "—", iso or "—"],
                    "hot": False,
                    **_term(label),
                }
            )
        if not compare_rows:
            return None
        return {
            "title": f"Sequence conservation · {clade}",
            "subtitle": (
                "mean AA identity over the differential ORF is the score basis; "
                "frame intactness across the clade is context"
            ),
            "cmp_headers": ["Conservation metric", "Canonical ORF", "Differential ORF"],
            "col_classes": CANON_ISO,
            "compare_rows": compare_rows,
        }

    def sec_phylop():
        # Baseline comparison: differential region vs conserved core (the column
        # convention used across every tile), one row per track.
        compare_rows = compare(
            [
                (
                    "phyloP mean",
                    [
                        "isoform_conservation_phylop_unique_region_mean",
                        "isoform_conservation_phylop_shared_region_mean",
                        "isoform_conservation_phylop_enrichment",
                    ],
                ),
                (
                    "phastCons mean",
                    [
                        "isoform_conservation_phastcons_unique_region_mean",
                        "isoform_conservation_phastcons_shared_region_mean",
                        None,
                    ],
                ),
            ]
        )
        if not compare_rows:
            return None
        return {
            "title": "Per-base conservation · differential region",
            "subtitle": (
                "absolute mean phyloP over the differential region is the score "
                "basis (high = strong purifying selection); the shared column and "
                "enrichment are context, not the claim"
            ),
            "cmp_headers": ["Track", "Differential", "Shared", "vs shared"],
            "col_classes": DIFF_SHARED_3,
            "compare_rows": compare_rows,
        }

    def sec_start_site():
        # Canonical-start vs alternative-start: initiation context + per-base
        # conservation at each start codon. The canonical column reads
        # ``canonical_*`` mirrors. One row per property.
        specs = [
            ("Start codon", "canonical_start_codon", "start_codon"),
            (
                "Kozak context (−9..+4)",
                "canonical_initiation_context_kozak_context",
                "isoform_initiation_context_kozak_context",
            ),
            (
                "phyloP at start codon",
                "canonical_conservation_phylop_at_tis",
                "isoform_conservation_phylop_at_tis",
            ),
            (
                "phastCons at start codon",
                "canonical_conservation_phastcons_at_tis",
                "isoform_conservation_phastcons_at_tis",
            ),
            (
                "phyloP over Kozak window",
                "canonical_conservation_phylop_kozak_mean",
                "isoform_conservation_phylop_kozak_mean",
            ),
            (
                "phastCons over Kozak window",
                "canonical_conservation_phastcons_kozak_mean",
                "isoform_conservation_phastcons_kozak_mean",
            ),
            (
                "Kozak mismatch — full consensus",
                "canonical_initiation_context_kozak_hamming_full",
                "isoform_initiation_context_kozak_hamming_full",
            ),
            (
                "Kozak window GC content",
                "canonical_initiation_context_kozak_window_gc_content",
                "isoform_initiation_context_kozak_window_gc_content",
            ),
        ]
        compare_rows = []
        any_alt = False
        for label, can_key, alt_key in specs:
            can = _fmt_num(g(can_key))
            alt = _fmt_num(g(alt_key))
            if alt is not None:
                any_alt = True
            compare_rows.append(
                {"label": label, "cols": [can or "—", alt or "—"], "hot": False, **_term(label)}
            )
        if not any_alt:
            return None
        return {
            "title": "Start site · canonical vs isoform",
            "subtitle": "initiation context + per-base conservation at each start codon",
            "cmp_headers": ["Property", "Canonical", "Isoform"],
            "col_classes": CANON_ISO,
            "compare_rows": compare_rows,
        }

    CELL_LINES = ("HeLa", "K562", "U2OS", "RPE1_Async", "RPE1_Que", "RPE1_Sen")

    def sec_cell_lines():
        # Canonical-start vs alternative-start usage (CPM) per cell line — the
        # differential is the alternative start; p-value is the significance of
        # the alternative-TIS call in that cell line.
        cell_rows = []
        for cl in CELL_LINES:
            can_cpm = _fmt_num(g(f"canonical_expr_{cl}_cpm"))
            alt_cpm = _fmt_num(g(f"expr_{cl}_cpm"))
            pval = _fmt_num(g(f"expr_{cl}_p_value"))
            if can_cpm is not None or alt_cpm is not None or pval is not None:
                cell_rows.append(
                    {
                        "label": cl.replace("_", " "),
                        "cols": [can_cpm or "—", alt_cpm or "—", pval or "—"],
                        "hot": False,
                    }
                )
        if not cell_rows:
            return None
        q = _fmt_num(g("fisher_qvalue"))
        return {
            "title": "Per-cell-line usage · canonical vs isoform",
            "subtitle": f"Fisher q = {q}" if q else "canonical vs isoform-start CPM per cell line",
            "cmp_headers": ["Cell line", "Canonical", "Isoform", "p-value"],
            "col_classes": ["dm-shared", "", ""],
            "compare_rows": cell_rows,
        }

    def sec_efficiency():
        # Canonical-start vs this alternative-start initiation efficiency, per
        # cell line. Canonical efficiency comes from the per-Tid Annotated row
        # (``canonical_expr_*``); the alternative start from ``expr_*``.
        eff_rows = []
        for cl in CELL_LINES:
            can = _fmt_num(g(f"canonical_expr_{cl}_initiation_efficiency"))
            alt = _fmt_num(g(f"expr_{cl}_initiation_efficiency"))
            if can is None and alt is None:
                continue
            eff_rows.append(
                {"label": cl.replace("_", " "), "cols": [can or "—", alt or "—"], "hot": False}
            )
        if not eff_rows:
            return None
        return {
            "title": "Initiation efficiency · canonical vs isoform",
            "subtitle": (
                "ribosome initiation efficiency at the canonical start vs this "
                "alternative start, per cell line"
            ),
            "cmp_headers": ["Cell line", "Canonical", "Isoform"],
            "col_classes": CANON_ISO,
            "compare_rows": eff_rows,
        }

    def sec_structure():
        # Whole-protein pLDDT, canonical vs isoform. Global fold-similarity
        # singletons (TM-score, RMSD, interface contacts) have no second side —
        # they are the canonical-vs-isoform comparison expressed as one number —
        # so they ride along in the caption rather than a Details list.
        status = g("isoform_structure_status")
        compare_rows = compare(
            [
                (
                    "pLDDT (whole protein)",
                    [
                        "isoform_structure_plddt_canonical_mean",
                        "isoform_structure_plddt_isoform_mean",
                    ],
                ),
            ]
        )
        if not compare_rows:
            return None
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
        return {
            "title": "Structure (ESMFold2) · canonical vs isoform",
            "subtitle": sub,
            "cmp_headers": ["Metric", "Canonical", "Isoform"],
            "col_classes": CANON_ISO,
            "compare_rows": compare_rows,
            "warn": warn,
        }

    def sec_structure_region():
        # Per-region fold confidence: differential region vs shared core. The
        # shared-region pLDDT isn't stored, but Δ (= diff − shared) is, so we
        # recover it (no pipeline change). Enrichment = differential / shared.
        diff_raw = g("isoform_structure_plddt_diffregion_mean")
        delta_raw = g("isoform_structure_plddt_delta_shared")
        if diff_raw is None or (isinstance(diff_raw, float) and not math.isfinite(diff_raw)):
            return None
        shared_raw = None
        try:
            if delta_raw is not None and math.isfinite(float(delta_raw)):
                shared_raw = float(diff_raw) - float(delta_raw)
        except (TypeError, ValueError):
            shared_raw = None
        enr = (float(diff_raw) / shared_raw) if (shared_raw and shared_raw != 0) else None
        row = {
            "label": "pLDDT",
            "cols": [
                _fmt_num(diff_raw) or "—",
                _fmt_num(shared_raw) or "—",
                f"{enr:.2g}" if enr is not None else "—",
            ],
            "hot": False,
            **_term("pLDDT — differential region"),
        }
        return {
            "title": "Fold confidence · differential vs shared region",
            "subtitle": "mean ESMFold2 pLDDT in each region",
            "cmp_headers": ["Metric", "Differential", "Shared", "Enrichment"],
            "col_classes": DIFF_SHARED_3,
            "compare_rows": [row],
        }

    def sec_shared_rmsd():
        # F7 — strict shared-region Cα RMSD (Kabsch-superposed on the shared
        # residues only). Global TM/RMSD ride in the caption; the shared-region
        # metrics are the score basis. When not evaluable, surface the status so
        # the modal explains itself rather than rendering blank.
        status = g("isoform_structure_rmsd_shared_status")
        rmsd = _fmt_num(g("isoform_structure_rmsd_shared"))
        tm = _fmt_num(g("isoform_structure_tm_score_shared"))
        n = g("isoform_structure_shared_region_len")
        pmin = None
        try:
            pvals = [
                float(x)
                for x in (
                    g("isoform_structure_plddt_shared_mean_isoform"),
                    g("isoform_structure_plddt_shared_mean_canonical"),
                )
                if x is not None and math.isfinite(float(x))
            ]
            pmin = min(pvals) if pvals else None
        except (TypeError, ValueError):
            pmin = None

        detail_rows = []
        if rmsd is not None:
            detail_rows.append(
                {"label": "Shared-region Cα RMSD", "value": f"{rmsd} Å",
                 **_term("Shared-region Cα RMSD")}
            )
        if tm is not None:
            detail_rows.append(
                {"label": "Shared-region TM-score", "value": tm,
                 **_term("Shared-region TM-score")}
            )
        if n is not None:
            detail_rows.append(
                {"label": "Shared region length", "value": f"{int(n)} aa",
                 **_term("Shared region length")}
            )
        pmin_fmt = _fmt_num(pmin)
        if pmin_fmt is not None:
            detail_rows.append(
                {"label": "Min shared-region pLDDT", "value": pmin_fmt,
                 **_term("Min shared-region pLDDT")}
            )

        # Global fold-similarity singletons ride in the caption for context.
        bits = []
        gtm = _fmt_num(g("isoform_structure_tm_score"))
        grmsd = _fmt_num(g("isoform_structure_rmsd_global"))
        if gtm is not None:
            bits.append(f"global TM-score {gtm}")
        if grmsd is not None:
            bits.append(f"global RMSD {grmsd} Å")

        if not detail_rows:
            if status and status != "ok":
                _why = {
                    "no_shared_region": "no shared region (uORF/altORF, or region too short)",
                    "unverified_alignment": "isoform/canonical alignment unverified",
                    "no_cache": "predicted structures not available",
                }.get(status, status)
                return {
                    "title": "Shared-region structural change",
                    "subtitle": f"not evaluable — {_why}",
                    "rows": [],
                }
            return None
        sub = "Cα RMSD superposed on the shared residues only; TM-score is length-normalized"
        if bits:
            sub += " · " + " · ".join(bits)
        return {
            "title": "Shared-region structural change",
            "subtitle": sub,
            "rows": detail_rows,
        }

    def sec_biophysics():
        feats = [
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
        compare_rows = compare(
            [
                (
                    label,
                    [
                        f"cmp_biophysics_{feat}_unique",
                        f"cmp_biophysics_{feat}_shared",
                        f"cmp_biophysics_{feat}_ratio",
                    ],
                    f"cmp_biophysics_{feat}_enriched",
                )
                for label, feat in feats
            ]
        )
        if not compare_rows:
            return None
        return {
            "title": "Biophysics · differential vs shared region",
            "subtitle": "highlighted rows are enriched in the differential region",
            "cmp_headers": ["Property", "Differential", "Shared", "Enrichment"],
            "col_classes": DIFF_SHARED_3,
            "compare_rows": compare_rows,
        }

    def sec_deeploc():
        compare_rows = compare(
            [
                (
                    "Predicted location",
                    [
                        "cmp_localization_deeploc_prediction_canonical",
                        "cmp_localization_deeploc_prediction_isoform",
                    ],
                ),
                (
                    "Sorting signals",
                    [
                        "cmp_localization_deeploc_signals_canonical",
                        "cmp_localization_deeploc_signals_isoform",
                    ],
                ),
                (
                    "Membrane",
                    [
                        "cmp_localization_deeploc_membrane_canonical",
                        "cmp_localization_deeploc_membrane_isoform",
                    ],
                ),
            ]
        )
        if not compare_rows:
            return None
        changed = (
            bool(g("cmp_localization_deeploc_signals_changed"))
            or bool(g("cmp_localization_deeploc_prediction_changed"))
            or bool(g("cmp_localization_deeploc_membrane_changed"))
        )
        return {
            "title": "Subcellular localization (DeepLoc)",
            "subtitle": (
                "localization features changed (prediction/signals/membrane)"
                if changed
                else "no localization-feature change"
            ),
            "cmp_headers": ["Property", "Canonical", "Isoform"],
            "col_classes": CANON_ISO,
            "compare_rows": compare_rows,
            "highlight": changed,
        }

    def sec_targeting():
        compare_rows = compare(
            [
                (
                    "TargetP",
                    ["canonical_targetp_targetp_prediction", "isoform_targetp_targetp_prediction"],
                ),
                (
                    "SignalP",
                    ["canonical_signalp_signalp_prediction", "isoform_signalp_signalp_prediction"],
                ),
            ]
        )
        if not compare_rows:
            return None
        changed = (
            bool(g("cmp_targetp_targetp_prediction_changed"))
            or bool(g("cmp_signalp_signalp_prediction_changed"))
        )
        return {
            "title": "N-terminal targeting (SignalP / TargetP)",
            "subtitle": "targeting signal changed" if changed else "no targeting change",
            "cmp_headers": ["Predictor", "Canonical", "Isoform"],
            "col_classes": CANON_ISO,
            "compare_rows": compare_rows,
            "highlight": changed,
        }

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
        if not (can_dom or iso_dom or can_mot or iso_mot):
            return None
        compare_rows = [
            {
                "label": "InterPro domains",
                "cols": [str(len(can_dom)), str(len(iso_dom))],
                "hot": len(iso_dom) != len(can_dom),
                **_term("InterPro domains"),
            },
            {
                "label": "Short linear motifs",
                "cols": [str(len(can_mot)), str(len(iso_mot))],
                "hot": len(iso_mot) != len(can_mot),
                **_term("Short linear motifs"),
            },
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
        return {
            "title": "Domains & motifs (canonical vs isoform)",
            "subtitle": "gained = only in the isoform; lost = only in the canonical",
            "cmp_headers": ["Feature", "Canonical", "Isoform"],
            "col_classes": CANON_ISO,
            "compare_rows": compare_rows,
            "hits": hits,
        }

    def sec_massspec():
        can = _to_record_list(g("canonical_massspec_hits"))
        iso = _to_record_list(g("isoform_massspec_hits"))
        if not can and not iso:
            return None
        can_val = sum(1 for p in can if p.get("validated"))
        iso_val = sum(1 for p in iso if p.get("validated"))
        iso_unique = [p for p in iso if p.get("unique_to_isoform")]
        compare_rows = [
            {
                "label": "Tryptic peptides (in-silico)",
                "cols": [str(len(can)), str(len(iso))],
                "hot": False,
                **_term("Tryptic peptides (in-silico)"),
            },
            {
                "label": "Validated by mass-spec",
                "cols": [str(can_val), str(iso_val)],
                "hot": bool(iso_val),
                **_term("Validated by mass-spec"),
            },
            {
                "label": "Isoform-unique peptides",
                "cols": ["—", str(len(iso_unique))],
                "hot": bool(iso_unique),
                **_term("Isoform-unique peptides"),
            },
        ]
        hits = []
        for p in iso_unique[:20]:
            pos, end = p.get("pos"), p.get("end")
            span = f"{pos}–{end}" if pos is not None and end is not None else ""
            kind = "validated" if p.get("validated") else "peptide"
            hits.append({"kind": kind, "name": str(p.get("peptide") or "?")[:40], "span": span})
        return {
            "title": "Mass-spec peptide support (canonical vs isoform)",
            "subtitle": "isoform-unique peptides are direct evidence the alternative protein exists",
            "cmp_headers": ["Feature", "Canonical", "Isoform"],
            "col_classes": CANON_ISO,
            "compare_rows": compare_rows,
            "hits": hits,
        }

    def sec_germline_tolerance():
        # gnomAD = germline population variation, a *tolerance* readout. Depletion
        # in the differential region vs the shared core = healthy human variation
        # avoids it = the region is constrained. Depletion ratio < 1× =
        # constrained; > 1× = tolerated. (Disease variants live in F6.)
        # Density denominator is the nt length of each region (emitted by the
        # variant_intersection module), not AA length — fall back to the AA-len
        # ratio only when the nt columns predate this row.
        unique_nt = _maybe_int(g("isoform_variant_intersection_unique_region_nt"))
        shared_nt = _maybe_int(g("isoform_variant_intersection_shared_region_nt"))
        depletion = _maybe_float(g("isoform_variant_intersection_gnomad_depletion_ratio"))

        unique_len = unique_nt or (
            (getattr(iso, "diff_end", None) or len(getattr(iso, "differential_sequence", "") or ""))
            * 3
        )
        shared_len = shared_nt or (
            ((getattr(iso, "isoform_len", None) if is_trunc
              else getattr(iso, "canonical_len", None)) or 0) * 3
        )

        def _enrich(nu, ns):
            if nu is None or ns is None or not unique_len or not shared_len or ns == 0:
                return None
            du, ds = nu / unique_len, ns / shared_len
            return None if ds == 0 else du / ds

        diff = _fmt_num(g("isoform_variant_intersection_n_gnomad_in_unique_region"))
        cons = _fmt_num(g("isoform_variant_intersection_n_gnomad_in_shared_region"))
        if diff is None and cons is None and depletion is None:
            return None
        if depletion is None:
            try:
                depletion = _enrich(
                    float(diff) if diff is not None else None,
                    float(cons) if cons is not None else None,
                )
            except (TypeError, ValueError):
                depletion = None
        row = {
            "label": "gnomAD variants",
            "cols": [
                diff or "—",
                cons or "—",
                f"{depletion:.2g}×" if depletion is not None else "—",
            ],
            "hot": bool(depletion is not None and depletion < 1),
            **_term("gnomAD variants"),
        }
        return {
            "title": "Germline tolerance · differential vs shared region",
            "subtitle": (
                "gnomAD (population) variant density per nucleotide — depletion "
                "ratio < 1× means healthy human variation avoids the region "
                "(constrained), the F5 basis alongside ESM-C constraint"
            ),
            "cmp_headers": ["Variant set", "Differential", "Shared", "Depletion ratio"],
            "col_classes": DIFF_SHARED_3,
            "compare_rows": [row],
        }

    def sec_constraint():
        # Region-resolved sequence constraint (ESM-C): is the differential region
        # more constrained than the shared region?
        compare_rows = compare(
            [
                (
                    "ESM-C mean LLR",
                    [
                        "isoform_plm_vep_mean_llr_unique_region",
                        "isoform_plm_vep_mean_llr_shared_region",
                        "isoform_plm_vep_constraint_enrichment",
                    ],
                ),
                (
                    "Constrained positions",
                    [
                        "isoform_plm_vep_n_constrained_positions_unique",
                        "isoform_plm_vep_n_constrained_positions_shared",
                        None,
                    ],
                ),
            ]
        )
        if not compare_rows:
            return None
        return {
            "title": "Sequence constraint (ESM-C) · differential vs shared region",
            "subtitle": "lower LLR = more constrained; enrichment = differential / conserved",
            "cmp_headers": ["Property", "Differential", "Shared", "Enrichment"],
            "col_classes": DIFF_SHARED_3,
            "compare_rows": compare_rows,
        }

    def sec_variant_burden(source, pool_label):
        # Predicted-damaging variant counts for one source pool (§4): gnomad
        # (germline → F5) or disease (ClinVar+COSMIC → F6). The predictors are
        # source-independent; this counts their calls over the chosen pool,
        # differential vs shared, length-normalized enrichment.
        unique_len = getattr(iso, "diff_end", None) or len(
            getattr(iso, "differential_sequence", "") or ""
        )
        shared_len = (getattr(iso, "isoform_len", None) if is_trunc
                      else getattr(iso, "canonical_len", None))

        def _enrich(nu, ns):
            if nu is None or ns is None or not unique_len or not shared_len or ns == 0:
                return None
            du, ds = nu / unique_len, ns / shared_len
            return None if ds == 0 else du / ds

        metrics = [
            ("Scorable variants", "n_scorable_in"),
            ("Damaging variants", "n_damaging_in"),
            ("— of which loss-of-function", "n_lof_in"),
            ("AlphaMissense-pathogenic", "n_am_pathogenic_in"),
        ]
        compare_rows = []
        any_diff = False
        for label, stem in metrics:
            diff = _fmt_num(g(f"isoform_varianteffect_{stem}_unique_{source}"))
            cons = _fmt_num(g(f"isoform_varianteffect_{stem}_shared_{source}"))
            if diff is not None:
                any_diff = True
            try:
                r = _enrich(float(diff) if diff is not None else None,
                            float(cons) if cons is not None else None)
            except (TypeError, ValueError):
                r = None
            compare_rows.append(
                {
                    "label": label,
                    "cols": [diff or "—", cons or "—", f"{r:.2g}×" if r is not None else "—"],
                    "hot": bool(r is not None and r > 1),
                    **_term(label),
                }
            )
        if not any_diff:
            return None
        return {
            "title": f"Predicted-damaging variants · {pool_label}",
            "subtitle": "AlphaMissense / ESM-C / LoF calls per region (length-normalized)",
            "cmp_headers": ["Variant set", "Differential", "Shared", "Enrichment"],
            "col_classes": DIFF_SHARED_3,
            "compare_rows": compare_rows,
        }

    def sec_variant_scores(source, pool_label):
        # Per-variant predictor scores for one source pool (§4), differential vs
        # shared. Predictor coverage (overall #scored) rides in the caption.
        metrics = [
            ("Mean ΔLLR (ESM-C)", "mean_delta_llr"),
            ("Min ΔLLR (ESM-C)", "min_delta_llr"),
            ("Mean AlphaMissense", "mean_am_pathogenicity"),
        ]
        compare_rows = []
        any_diff = False
        for label, stem in metrics:
            diff = _fmt_num(g(f"isoform_varianteffect_{stem}_unique_{source}"))
            shared = _fmt_num(g(f"isoform_varianteffect_{stem}_shared_{source}"))
            if diff is not None:
                any_diff = True
            compare_rows.append(
                {"label": label, "cols": [diff or "—", shared or "—"], "hot": False, **_term(label)}
            )
        if not any_diff:
            return None
        cov = []
        plm = _fmt_num(g("isoform_varianteffect_n_scored_plm"))
        am = _fmt_num(g("isoform_varianteffect_n_scored_am"))
        if plm is not None:
            cov.append(f"{plm} ESM-C")
        if am is not None:
            cov.append(f"{am} AlphaMissense")
        sub = "scored: " + " · ".join(cov) if cov else f"{pool_label} variants"
        return {
            "title": f"Predictor scores · {pool_label}",
            "subtitle": sub,
            "cmp_headers": ["Property", "Differential", "Shared"],
            "col_classes": DIFF_SHARED_2,
            "compare_rows": compare_rows,
        }

    def sec_clinical_burden():
        # Region lengths (nt) for length-normalized density — emitted by the
        # variant_intersection module. The differential region is the unique
        # N-terminus; the shared region is whichever whole protein it is shared
        # with (canonical for extensions, isoform for truncations). Fall back to
        # AA-length×3 only when the nt columns predate this row.
        unique_nt = _maybe_int(g("isoform_variant_intersection_unique_region_nt"))
        shared_nt = _maybe_int(g("isoform_variant_intersection_shared_region_nt"))
        unique_len = unique_nt or (
            (getattr(iso, "diff_end", None) or len(getattr(iso, "differential_sequence", "") or ""))
            * 3
        )
        can_len = getattr(iso, "canonical_len", None)
        iso_len = getattr(iso, "isoform_len", None)
        shared_len = shared_nt or (((iso_len if is_trunc else can_len) or 0) * 3)
        disease_ratio = _maybe_float(
            g("isoform_variant_intersection_disease_enrichment_ratio")
        )

        def _n(key):
            v = _fmt_num(g(key))
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        def _enrich(nu, ns):
            if nu is None or ns is None or not unique_len or not shared_len or ns == 0:
                return None
            du, ds = nu / unique_len, ns / shared_len
            return None if ds == 0 else du / ds

        crows = []
        for label, uk, sk in [
            (
                "Disease variants",
                "isoform_variant_intersection_n_disease_in_unique_region",
                "isoform_variant_intersection_n_disease_in_shared_region",
            ),
            (
                "Pathogenic",
                "isoform_variant_intersection_n_pathogenic_in_unique_region",
                "isoform_variant_intersection_n_pathogenic_in_shared_region",
            ),
        ]:
            nu, ns = _n(uk), _n(sk)
            if nu is None and ns is None:
                continue
            # Prefer the module's disease enrichment ratio for the disease row
            # (it is the F6 basis); fall back to the locally-computed density.
            if label == "Disease variants" and disease_ratio is not None:
                r = disease_ratio
            else:
                r = _enrich(nu, ns)
            crows.append(
                {
                    "label": label,
                    "cols": [
                        _fmt_num(nu) or "—",
                        _fmt_num(ns) or "—",
                        f"{r:.2g}×" if r is not None else "—",
                    ],
                    "hot": bool(r is not None and r > 1),
                    **_term(label),
                }
            )
        if not crows:
            return None
        return {
            "title": "Clinical-variant burden · differential vs shared region",
            "subtitle": (
                "counts per region; ratio is density-normalized (variants per "
                "nucleotide, differential ÷ shared — ≥1× = disease variants "
                "concentrate in the differential region, the F6 basis)"
            ),
            "cmp_headers": ["Variant set", "Differential", "Shared", "Enrichment"],
            "col_classes": DIFF_SHARED_3,
            "compare_rows": crows,
        }

    # ---- assign sections to criteria --------------------------------------
    assignments = {
        "E1_primate_conservation": [sec_frame("primate")],
        "E2_mammalian_conservation": [sec_frame("mammalian")],
        "E3_phylop_coding_selection": [sec_phylop(), sec_start_site()],
        "E4_multi_cell_line": [sec_cell_lines()],
        "E5_initiation_efficiency": [sec_efficiency()],
        "E6_mass_spec": [sec_massspec()],
        "F1_structured_extension": [sec_structure(), sec_structure_region(), sec_biophysics()],
        "F2_localization_change": [sec_deeploc()],
        "F3_domain_change": [sec_domains_motifs()],
        "F4_targeting_change": [sec_targeting()],
        "F5_pathogenic_variant_enrichment": [
            sec_germline_tolerance(),
            sec_constraint(),
            sec_variant_burden("gnomad", "germline (gnomAD)"),
            sec_variant_scores("gnomad", "germline (gnomAD)"),
        ],
        "F6_clinical_variant_overlap": [
            sec_clinical_burden(),
            sec_variant_burden("disease", "disease (ClinVar/COSMIC)"),
            sec_variant_scores("disease", "disease (ClinVar/COSMIC)"),
        ],
        "F7_shared_structural_change": [sec_shared_rmsd()],
    }

    out = {}
    for cid, secs in assignments.items():
        out[cid] = {
            "about": CRITERION_ABOUT.get(cid, ""),
            "sections": [s for s in secs if s],
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


# Drives the evidence-tile UI grid on the isoform page (E1..E6, F1..F7).
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
        "label": "Germline tolerance & constraint",
        "short_label": "Germline",
    },
    {
        "id": "F6_clinical_variant_overlap",
        "axis": "F",
        "label": "Clinical variant overlap",
        "short_label": "Variants",
    },
    {
        "id": "F7_shared_structural_change",
        "axis": "F",
        "label": "Shared region structural changes",
        "short_label": "Shared RMSD",
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
