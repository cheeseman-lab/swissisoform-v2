"""Module: Mass Spectrometry Validation — in-silico tryptic digest and peptide validation.

Performs in-silico tryptic digestion of isoform proteins to identify unique peptides,
optionally cross-referencing with pre-computed PepQuery2 validation results.

``precompute_pepquery`` runs the PepQuery2 standalone jar (staged by
``scripts/setup/setup_databases.py pepquery`` into
``data/reference/pepquery/pepquery-2.0.2/pepquery-2.0.2.jar``) over a
batch of candidate peptides via ``java -jar``.  MS/MS spectra come from
PepQueryDB on demand via the ``-b`` flag — no local spectral index.
The returned dict feeds ``MassSpecModule`` at init.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from swissisoform.config import PipelineConfig
from swissisoform.models import TranslationInitiationSite

logger = logging.getLogger(__name__)


DEFAULT_PEPQUERY_JAR = (
    Path(__file__).resolve().parents[3]
    / "data" / "reference" / "pepquery" / "pepquery-2.0.2" / "pepquery-2.0.2.jar"
)


class MassSpecModule:
    """Mass spectrometry validation module (``ProteinModule`` protocol).

    Performs in-silico tryptic digestion to find peptides with their positions
    in the protein, marks peptides unique to the isoform (not found in canonical
    digest), and optionally validates against pre-computed PepQuery2 results.

    Attributes:
        MODULE_NAME: Unique module identifier.
        OUTPUT_COLUMNS: Column names produced (prefixed with MODULE_NAME_).
        SCOPE: Module scope ('C' for per-site classification).
    """

    MODULE_NAME: str = "massspec"
    OUTPUT_COLUMNS: list[str] = ["massspec_hits", "massspec_summary"]
    SCOPE: str = "C"

    def __init__(
        self,
        config: PipelineConfig,
        validated_peptides: dict[str, set[str]] | None = None,
    ) -> None:
        """Initialize with pipeline configuration.

        Args:
            config: Pipeline configuration.
            validated_peptides: Optional pre-computed PepQuery results.
                Dict mapping gene_name -> set of validated peptide sequences.
                If provided, peptides found in this set are marked as validated.
        """
        self.config = config
        self.validated_peptides = validated_peptides or {}

    def _tryptic_digest(
        self,
        protein: str,
        missed_cleavages: int = 1,
        min_length: int = 7,
        max_length: int = 30,
    ) -> list[dict[str, Any]]:
        """In-silico tryptic digestion returning peptides with positions.

        Trypsin cleaves after K or R, except when followed by P.

        Args:
            protein: Protein sequence, optionally ending with '*'.
            missed_cleavages: Maximum number of missed cleavages (0 and 1).
            min_length: Minimum peptide length to keep.
            max_length: Maximum peptide length to keep.

        Returns:
            List of dicts with keys: peptide, pos, end, length.
        """
        seq = protein.rstrip("*").upper()
        if not seq:
            return []

        # Find cleavage sites: positions where seq[i] in (K, R) and seq[i+1] != P
        cleavage_sites: list[int] = []
        for i in range(len(seq)):
            if seq[i] in ("K", "R"):
                if i + 1 < len(seq) and seq[i + 1] == "P":
                    continue  # KP/RP exception
                cleavage_sites.append(i)

        # Build site boundaries: start positions of each fragment
        sites = [0] + [s + 1 for s in cleavage_sites] + [len(seq)]

        # Generate peptides for each missed cleavage count
        seen: set[tuple[str, int]] = set()
        peptides: list[dict[str, Any]] = []

        for mc in range(missed_cleavages + 1):
            for i in range(len(sites) - 1 - mc):
                start = sites[i]
                end = sites[i + 1 + mc]
                pep = seq[start:end]
                pep_len = len(pep)
                if pep_len < min_length or pep_len > max_length:
                    continue
                key = (pep, start)
                if key in seen:
                    continue
                seen.add(key)
                peptides.append(
                    {
                        "peptide": pep,
                        "pos": start,
                        "end": end,
                        "length": pep_len,
                    }
                )

        return peptides

    def annotate(
        self,
        protein: str,
        canonical_protein: str | None = None,
        gene_name: str | None = None,
    ) -> dict[str, Any]:
        """Compute mass spectrometry annotations for a protein.

        When *canonical_protein* is ``None`` (unknown), ``unique_to_isoform``
        is set to ``None`` for every peptide — NOT ``False`` — because we
        cannot tell whether a peptide is unique without the canonical digest.
        Summary ``unique_peptides`` is also ``None`` in that case.

        Args:
            protein: Isoform protein sequence.
            canonical_protein: Canonical protein sequence for uniqueness
                comparison. ``None`` means "unknown" (output will not claim
                uniqueness). Empty string is treated as unknown.
            gene_name: Gene name for PepQuery result lookup. ``None``/empty
                means "unknown" (no validation performed).

        Returns:
            Dict with keys 'hits' (list of peptide dicts) and 'summary' (stats dict).
        """
        isoform_peptides = self._tryptic_digest(protein)

        # Normalize missing values — empty string is treated as "unknown"
        canonical_known = bool(canonical_protein)
        gene_known = bool(gene_name)

        if not canonical_known:
            logger.debug(
                "MassSpec.annotate called without canonical_protein — "
                "unique_to_isoform will be None (unknown) for every peptide"
            )

        # pepquery_run is True only when a validated-peptide cache was
        # provided at init *and* contains an entry for this gene.  An
        # empty cache means PepQuery2 was never precomputed — downstream
        # consumers (scoring E6) should treat ``validated_peptides == 0``
        # as "cannot evaluate" rather than "no evidence".
        pepquery_run = gene_known and gene_name in self.validated_peptides

        if not isoform_peptides:
            return {
                "hits": [],
                "summary": {
                    "total_peptides": 0,
                    "unique_peptides": 0 if canonical_known else None,
                    "validated_peptides": 0 if pepquery_run else None,
                    "min_peptide_length": None,
                    "max_peptide_length": None,
                    "pepquery_run": pepquery_run,
                },
            }

        # Build canonical peptide set for uniqueness check
        canonical_pep_seqs: set[str] = set()
        if canonical_known:
            canonical_digested = self._tryptic_digest(canonical_protein)
            canonical_pep_seqs = {p["peptide"] for p in canonical_digested}

        # Get validated peptide set for this gene.  Only truthy when
        # PepQuery2 has actually been precomputed for this gene.
        gene_validated: set[str] = (
            self.validated_peptides.get(gene_name, set()) if gene_known else set()
        )

        # Annotate each peptide
        hits: list[dict[str, Any]] = []
        unique_count = 0
        validated_count = 0

        for pep in isoform_peptides:
            # unique is None (unknown) when canonical is missing — NOT False
            if canonical_known:
                unique = pep["peptide"] not in canonical_pep_seqs
            else:
                unique = None

            # validated is None (unknown) when PepQuery2 was never
            # precomputed for this gene — NOT False.  False would wrongly
            # claim evidence absence when we simply haven't searched.
            if pepquery_run:
                validated = pep["peptide"] in gene_validated
            else:
                validated = None

            if unique is True:
                unique_count += 1
            if validated is True:
                validated_count += 1

            hits.append(
                {
                    "peptide": pep["peptide"],
                    "pos": pep["pos"],
                    "end": pep["end"],
                    "length": pep["length"],
                    "unique_to_isoform": unique,
                    "validated": validated,
                }
            )

        lengths = [h["length"] for h in hits]
        summary = {
            "total_peptides": len(hits),
            "unique_peptides": unique_count if canonical_known else None,
            "validated_peptides": validated_count if pepquery_run else None,
            "min_peptide_length": min(lengths),
            "max_peptide_length": max(lengths),
            "pepquery_run": pepquery_run,
        }

        return {"hits": hits, "summary": summary}

    def run(self, tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]:
        """Compute mass spec annotations for each TIS site.

        Args:
            tis_sites: Input TIS sites with proteins set.

        Returns:
            The same sites with isoform_annotations["massspec"] populated.
        """
        for site in tis_sites:
            site.isoform_annotations[self.MODULE_NAME] = self.annotate(
                site.isoform_protein,
                canonical_protein=site.canonical_protein,
                gene_name=site.gene_name,
            )
        return tis_sites


# ---------------------------------------------------------------------------
# PepQuery2 precompute
# ---------------------------------------------------------------------------


def collect_unique_peptides(
    genes: list[Any],
    *,
    min_length: int = 7,
    max_length: int = 30,
    missed_cleavages: int = 1,
) -> dict[str, set[str]]:
    """Build ``{gene_name: {peptide, ...}}`` for all isoform-unique peptides.

    For each TIS in each gene, tryptic-digest the isoform protein, then
    set-difference against the TIS's own ``canonical_protein`` digest.
    The result feeds :func:`precompute_pepquery`.
    """
    probe = MassSpecModule.__new__(MassSpecModule)
    probe.validated_peptides = {}
    out: dict[str, set[str]] = {}
    for gene in genes:
        for site in gene.tis_sites:
            if not site.isoform_protein or not site.canonical_protein:
                continue
            iso = {
                p["peptide"]
                for p in probe._tryptic_digest(
                    site.isoform_protein,
                    missed_cleavages=missed_cleavages,
                    min_length=min_length,
                    max_length=max_length,
                )
            }
            can = {
                p["peptide"]
                for p in probe._tryptic_digest(
                    site.canonical_protein,
                    missed_cleavages=missed_cleavages,
                    min_length=min_length,
                    max_length=max_length,
                )
            }
            unique = iso - can
            if unique:
                out.setdefault(gene.gene_name, set()).update(unique)
    return out


def _parse_pepquery_output(outdir: Path) -> set[str]:
    """Return the set of peptide sequences PepQuery2 flagged as confident.

    PepQuery2 writes one ``psm_rank.txt`` **per searched dataset** under
    ``<outdir>/<dataset_name>/psm_rank.txt`` (alongside ``psm.txt``,
    ``psm_rank.mgf``, ``detail.txt``, ``ptm.txt``, etc.).  There is also
    a sibling ``<outdir>/database/`` directory holding the built FMIndex,
    which we skip.  A peptide is considered validated when at least one
    of its PSMs has ``confident == "Yes"``.

    Globs every matching file so comma-separated ``-b`` runs (which
    produce one subdir per dataset) aggregate cleanly.
    """
    rank_files = [
        p for p in outdir.glob("*/psm_rank.txt") if p.parent.name != "database"
    ]
    if not rank_files:
        logger.warning(
            "pepquery: no psm_rank.txt under %s/*/ — no peptides validated", outdir
        )
        return set()

    import pandas as pd

    validated: set[str] = set()
    for rank_file in rank_files:
        try:
            df = pd.read_csv(rank_file, sep="\t")
        except Exception as exc:  # noqa: BLE001
            logger.warning("pepquery: could not parse %s: %s", rank_file, exc)
            continue
        if "peptide" not in df.columns:
            logger.warning("pepquery: %s missing 'peptide' column", rank_file)
            continue
        if "confident" in df.columns:
            hit = df[df["confident"].astype(str).str.lower() == "yes"]
        elif "rank" in df.columns:
            hit = df[df["rank"] == 1]
        else:
            hit = df
        validated.update(hit["peptide"].astype(str).tolist())

    return validated


def precompute_pepquery(
    peptides_by_gene: dict[str, set[str]],
    *,
    dataset: str = "w",
    reference_db: str = "gencode:human",
    jar_path: Path | None = None,
    java_bin: str = "java",
    cache_dir: Path | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, set[str]]:
    """Run PepQuery2 over all unique peptides in one batched invocation.

    Pools every peptide across genes into a single flat input file,
    invokes ``java -jar <pepquery.jar> -b <dataset> -db <ref> -hc
    -i pep.txt -o outdir/`` exactly once, parses the output, then
    re-keys the validated set back to gene names.

    PepQueryDB ``-b`` tags (from PepQuery2 docs):

        ``all``  all MS/MS datasets in PepQueryDB (slow, broadest)
        ``w``    global proteome datasets (recommended default)
        ``p``    phosphoproteome datasets
        ``g``    glycosylation datasets
        ``a``    acetylation datasets
        ``u``    ubiquitination datasets
        ``CPTAC`` all CPTAC datasets
        ``CPTAC_TCGA_Colon_Cancer_Proteome_PDC000111``  a specific dataset

    Args:
        peptides_by_gene: ``{gene: {peptide, ...}}`` — build with
            :func:`collect_unique_peptides`.
        dataset: PepQueryDB ``-b`` tag.  ``"w"`` is the default because
            it covers global tumor + tissue proteomes without the
            modification-focused datasets that slow ``all`` down.
        reference_db: ``-db`` reference — ``gencode:human`` matches our
            isoform sequences.  Use ``swissprot:human`` for canonical
            peptides only.
        jar_path: Path to ``pepquery-2.0.2.jar``.  Defaults to the
            location staged by ``scripts/setup/setup_databases.py pepquery``
            (``data/reference/pepquery/pepquery-2.0.2/``).
        java_bin: ``java`` binary (>= 11).  Defaults to system PATH.
        cache_dir: Optional ``data/cache/pepquery/`` for result JSON.
            Keyed by sha1 of ``(dataset, reference_db, sorted peptide
            list)`` — re-runs with the same inputs skip the Java step.
        extra_args: Extra CLI flags forwarded to pepquery2 (e.g.
            ``["-tol", "10"]``).  Defaults keep PepQuery's high-confidence
            filter (``-hc``) on.

    Returns:
        ``{gene: {validated_peptide, ...}}``.  Empty dict (with WARN)
        when the conda env is missing or the subprocess fails — callers
        hand this into ``MassSpecModule`` which gracefully degrades
        ``pepquery_run`` to False.
    """
    if not peptides_by_gene:
        logger.info("precompute_pepquery: no peptides to validate")
        return {}

    # Deduplicate peptides while remembering which gene(s) each came from.
    peptide_to_genes: dict[str, set[str]] = {}
    for gene, peps in peptides_by_gene.items():
        for pep in peps:
            peptide_to_genes.setdefault(pep, set()).add(gene)

    peptides_sorted = sorted(peptide_to_genes)
    logger.info(
        "precompute_pepquery: %d unique peptides across %d genes "
        "(dataset=%s, db=%s)",
        len(peptides_sorted),
        len(peptides_by_gene),
        dataset,
        reference_db,
    )

    # ── Cache check ────────────────────────────────────────────────
    cache_key = _pepquery_cache_key(dataset, reference_db, peptides_sorted)
    cache_path = (Path(cache_dir) / f"{cache_key}.json") if cache_dir else None
    if cache_path and cache_path.exists():
        logger.info("precompute_pepquery: cache hit %s", cache_path)
        with open(cache_path) as fh:
            validated_peptides = set(json.load(fh))
        return _regroup_by_gene(validated_peptides, peptide_to_genes)

    jar = Path(jar_path) if jar_path else DEFAULT_PEPQUERY_JAR
    if not jar.exists():
        logger.warning(
            "precompute_pepquery: jar not found at %s — run "
            "`python scripts/setup/setup_databases.py pepquery` first",
            jar,
        )
        return {}

    if shutil.which(java_bin) is None:
        logger.warning(
            "precompute_pepquery: %r not on PATH — install openjdk >= 11",
            java_bin,
        )
        return {}

    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="pepquery_", dir="."))
    peptides_file = tmpdir / "peptides.txt"
    outdir = tmpdir / "out"
    outdir.mkdir()
    with open(peptides_file, "w") as fh:
        fh.write("\n".join(peptides_sorted) + "\n")

    cmd = [
        java_bin,
        "-jar",
        str(jar),
        "-b",
        dataset,
        "-db",
        reference_db,
        "-hc",
        "-i",
        str(peptides_file),
        "-o",
        str(outdir),
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("precompute_pepquery: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if proc.stdout:
            logger.info("pepquery stdout (tail):\n%s", proc.stdout[-1000:])
        if proc.stderr:
            logger.info("pepquery stderr (tail):\n%s", proc.stderr[-1000:])
    except subprocess.CalledProcessError as exc:
        logger.error(
            "precompute_pepquery: pepquery exited %d. stderr=%s",
            exc.returncode,
            (exc.stderr or "")[-1000:],
        )
        return {}

    validated_peptides = _parse_pepquery_output(outdir)
    logger.info(
        "precompute_pepquery: %d / %d peptides validated",
        len(validated_peptides),
        len(peptides_sorted),
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(sorted(validated_peptides), fh)
        logger.info("precompute_pepquery: cached → %s", cache_path)

    # Only clean up the tempdir when we successfully parsed validated
    # peptides.  On zero-hit runs (parser miss, broken Java, misconfigured
    # datasets) we keep the ``pepquery_*`` tree so the user can inspect
    # ``out/<dataset>/psm_rank.txt`` and ``detail.txt`` directly.
    if validated_peptides:
        try:
            shutil.rmtree(tmpdir)
        except OSError:
            logger.warning("precompute_pepquery: failed to remove %s", tmpdir)
    else:
        logger.info(
            "precompute_pepquery: 0 peptides validated — keeping %s for inspection",
            tmpdir,
        )

    return _regroup_by_gene(validated_peptides, peptide_to_genes)


def _regroup_by_gene(
    validated: set[str], peptide_to_genes: dict[str, set[str]]
) -> dict[str, set[str]]:
    """Map ``validated`` peptide set back to ``{gene: {peptide, ...}}``.

    Every gene that had peptides submitted to PepQuery2 is present in the
    output — with an empty set if none of its peptides matched a confident
    spectrum. This lets :class:`MassSpecModule` distinguish "PepQuery ran but
    found no MS evidence" (the gene IS in the dict, validated count is 0)
    from "PepQuery never queried this gene" (the gene is NOT in the dict).
    Without this guarantee, queried-but-unvalidated genes look identical to
    unqueried ones and E6 incorrectly reports ``None`` instead of ``False``.
    """
    out: dict[str, set[str]] = {}
    # Initialize every queried gene with an empty set so it shows up in the
    # output even when PepQuery validated zero of its peptides.
    for genes in peptide_to_genes.values():
        for gene in genes:
            out.setdefault(gene, set())
    for pep in validated:
        for gene in peptide_to_genes.get(pep, ()):
            out[gene].add(pep)
    return out


def _pepquery_cache_key(dataset: str, reference_db: str, peptides: list[str]) -> str:
    """Stable hash over ``(dataset, db, peptide list)`` for cache lookup."""
    import hashlib

    payload = "\n".join([dataset, reference_db, *peptides]).encode("ascii")
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:16]


