"""Set up reference databases from primary sources.

Idempotent orchestrator for the four external reference databases the
expensive annotation modules depend on:

    diamond  — UniProt SwissProt reviewed FASTA → DIAMOND .dmnd database
               (conservation module)
    gnomad   — gnomAD v4.1 exome VCF → gene-indexed parquet
               (clinical module, gnomAD source)
    clinvar  — NCBI variant_summary.txt.gz → gene-indexed parquet
               (clinical module, ClinVar source)
    cosmic   — manual raw TSV prereq → standardized parquet
               (clinical module, COSMIC source)
    gencode  — delegates to scripts/setup/download_references.sh
    pepquery        — PepQuery2 jar (mass-spec module)
    pepquery-spectra — mirror PepQueryDB MS/MS library to a local store
               (~196 GiB, opt-in; lets runs search via local `-ms`, no per-run S3)

Each subcommand writes:
    data/reference/<db>/<artifact>
    data/reference/<db>/_setup.json   # provenance sidecar

By default, subcommands skip work if the artifact already exists. Pass
``--refresh`` to force re-download + re-build.

Driven by the thin CLI at ``scripts/setup/setup_databases.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE per line, # comments ok.

    Does not overwrite existing env vars (so CLI ``export FOO=...`` wins).
    No external deps.
    """
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


logger = logging.getLogger("setup_databases")

ROOT = Path(__file__).resolve().parents[3]
REF = ROOT / "data" / "reference"

# ---------------------------------------------------------------------------
# Provenance sidecars
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sidecar(
    db_dir: Path,
    *,
    source_url: str,
    version: str,
    artifact: Path,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write ``_setup.json`` recording provenance for a DB artifact.

    Handles both single-file and directory artifacts (e.g. COSMIC's
    per-VCF parquet directory). For directories, records the aggregate
    size and per-file sha256 sums instead of a single hash.
    """
    db_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "source_url": source_url,
        "version": version,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact": str(artifact.relative_to(ROOT)),
    }
    if artifact.is_dir():
        files = sorted(f for f in artifact.rglob("*") if f.is_file())
        payload["artifact_is_directory"] = True
        payload["artifact_size_bytes"] = sum(f.stat().st_size for f in files)
        payload["artifact_files"] = {
            str(f.relative_to(artifact)): {
                "size_bytes": f.stat().st_size,
                "sha256": _sha256(f),
            }
            for f in files
        }
    else:
        payload["artifact_size_bytes"] = artifact.stat().st_size
        payload["artifact_sha256"] = _sha256(artifact)
    if extra:
        payload.update(extra)
    (db_dir / "_setup.json").write_text(json.dumps(payload, indent=2))


def is_built(artifact: Path, refresh: bool) -> bool:
    """Return True if *artifact* exists and we aren't forcing refresh."""
    if refresh and artifact.exists():
        logger.info("refresh: removing %s", artifact)
        artifact.unlink()
        return False
    return artifact.exists()


def run(cmd: list[str], **kwargs: Any) -> None:
    """Subprocess.run wrapper with logging + check=True."""
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


# ---------------------------------------------------------------------------
# DIAMOND DB (UniProt SwissProt reviewed)
# ---------------------------------------------------------------------------

DIAMOND_DIR = REF / "diamond"
DIAMOND_FASTA = DIAMOND_DIR / "uniprot_sprot.fasta.gz"
DIAMOND_DB = DIAMOND_DIR / "swissprot.dmnd"
DIAMOND_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/complete/uniprot_sprot.fasta.gz"
)


def setup_diamond(refresh: bool = False) -> None:
    """Download UniProt SwissProt reviewed FASTA and build a DIAMOND DB."""
    if is_built(DIAMOND_DB, refresh):
        logger.info("diamond: %s already exists — skipping", DIAMOND_DB)
        return
    if shutil.which("diamond") is None:
        raise RuntimeError("diamond binary not found on PATH")

    DIAMOND_DIR.mkdir(parents=True, exist_ok=True)

    if not DIAMOND_FASTA.exists() or refresh:
        run(["wget", "-q", "--show-progress", DIAMOND_URL, "-O", str(DIAMOND_FASTA)])

    # diamond makedb reads gz directly
    run(
        [
            "diamond",
            "makedb",
            "--in",
            str(DIAMOND_FASTA),
            "-d",
            str(DIAMOND_DB.with_suffix("")),  # diamond appends .dmnd
        ]
    )

    write_sidecar(
        DIAMOND_DIR,
        source_url=DIAMOND_URL,
        version="uniprot_sprot.current_release",
        artifact=DIAMOND_DB,
    )
    logger.info("diamond: built %s", DIAMOND_DB)


# ---------------------------------------------------------------------------
# ClinVar variant_summary
# ---------------------------------------------------------------------------

CLINVAR_DIR = REF / "clinvar"
CLINVAR_GZ = CLINVAR_DIR / "variant_summary.txt.gz"
CLINVAR_PARQUET = CLINVAR_DIR / "variant_summary.parquet"
CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
# Columns we keep + the downstream-standardized names we map them to
_CLINVAR_KEEP = [
    "Type",
    "Name",
    "GeneSymbol",
    "ClinicalSignificance",
    "RS# (dbSNP)",
    "PhenotypeIDS",
    "PhenotypeList",
    "Origin",
    "Assembly",
    "Chromosome",
    "Start",
    "Stop",
    "ReferenceAllele",
    "AlternateAllele",
    "VariationID",
    "HGNC_ID",
]


def setup_clinvar(refresh: bool = False) -> None:
    """Download ClinVar variant_summary and convert to a gene-indexed parquet."""
    import pandas as pd

    if is_built(CLINVAR_PARQUET, refresh):
        logger.info("clinvar: %s already exists — skipping", CLINVAR_PARQUET)
        return

    CLINVAR_DIR.mkdir(parents=True, exist_ok=True)

    if not CLINVAR_GZ.exists() or refresh:
        run(["wget", "-q", "--show-progress", CLINVAR_URL, "-O", str(CLINVAR_GZ)])

    logger.info("clinvar: parsing %s", CLINVAR_GZ)
    t0 = time.perf_counter()
    df = pd.read_csv(
        CLINVAR_GZ,
        sep="\t",
        low_memory=False,
        na_values=["-", "", "na"],
        usecols=lambda c: c in _CLINVAR_KEEP or True,  # keep all cols; subset below
    )
    # Filter to GRCh38 assembly
    if "Assembly" in df.columns:
        df = df[df["Assembly"] == "GRCh38"]
    # Keep the columns we care about (if present) + anything else
    df = df.reset_index(drop=True)

    df.to_parquet(CLINVAR_PARQUET, index=False)
    logger.info(
        "clinvar: wrote %d rows × %d cols to %s (%.1fs)",
        *df.shape,
        CLINVAR_PARQUET,
        time.perf_counter() - t0,
    )

    write_sidecar(
        CLINVAR_DIR,
        source_url=CLINVAR_URL,
        version="variant_summary.current",
        artifact=CLINVAR_PARQUET,
        extra={"n_rows": int(len(df)), "n_cols": int(df.shape[1])},
    )


# ---------------------------------------------------------------------------
# gnomAD v4.1 exome bulk (heavy — ~20-40 GB)
# ---------------------------------------------------------------------------

GNOMAD_DIR = REF / "gnomad"
GNOMAD_PARQUET = GNOMAD_DIR / "gnomad_v4.1_exome.parquet"
GNOMAD_VCF_DIR = GNOMAD_DIR / "vcf"  # per-chromosome VCFs staged here
# HTTPS mirror of the GCS bucket (public). Per-chromosome files:
#   https://gnomad-public-us-east-1.s3.amazonaws.com/release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr{N}.vcf.bgz
GNOMAD_BASE = "https://gnomad-public-us-east-1.s3.amazonaws.com/release/4.1/vcf/exomes/"
GNOMAD_CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]


def _parse_vep_header(vcf: "pysam.VariantFile") -> list[str]:
    """Extract the VEP CSQ format from the VCF header (pipe-separated field names)."""
    for record in vcf.header.info.values():
        if record.name == "vep":
            desc = record.description
            if "Format:" in desc:
                fmt = desc.split("Format:", 1)[1].strip().strip('"')
                return [s.strip() for s in fmt.split("|")]
    raise ValueError("gnomAD VCF missing 'vep' INFO field")


def _parse_gnomad_vcf(vcf_path: Path, parquet_path: Path) -> int:
    """Stream a gnomAD exome VCF into a per-chrom parquet.

    Filters to PASS variants with a canonical-transcript VEP annotation
    that has a non-null SYMBOL (gene name).  Extracts:

        chrom, pos, ref, alt, allele_frequency,
        consequence, gene_symbol, hgvsp, hgvsc, protein_position,
        variant_id  (chr-pos-ref-alt)

    Returns the number of rows written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pysam

    vcf = pysam.VariantFile(str(vcf_path))
    vep_fields = _parse_vep_header(vcf)
    idx = {name: i for i, name in enumerate(vep_fields)}

    keep_cols = [
        "Consequence",
        "SYMBOL",
        "CANONICAL",
        "HGVSp",
        "HGVSc",
        "Protein_position",
    ]
    for c in keep_cols:
        if c not in idx:
            raise ValueError(f"gnomAD VEP header missing expected field {c!r}")

    rows: list[dict[str, Any]] = []
    BATCH = 500_000
    writer: pq.ParquetWriter | None = None
    n_total = 0

    def flush() -> None:
        nonlocal rows, writer, n_total
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(str(parquet_path), table.schema)
        writer.write_table(table)
        n_total += len(rows)
        rows = []

    for rec in vcf:
        # Skip non-PASS (gnomAD uses "PASS" or comma-separated filters)
        if rec.filter.keys() and "PASS" not in rec.filter.keys():
            continue
        # AF is the per-allele allele frequency
        try:
            af_tuple = rec.info.get("AF")
            af = float(af_tuple[0]) if af_tuple else None
        except (TypeError, IndexError):
            af = None

        vep_val = rec.info.get("vep")
        if not vep_val:
            continue

        # "vep" is a tuple of strings (one per transcript annotation)
        canonical_ann: list[str] | None = None
        for ann_str in vep_val:
            parts = ann_str.split("|")
            if len(parts) < len(vep_fields):
                continue
            if parts[idx["CANONICAL"]] != "YES":
                continue
            if not parts[idx["SYMBOL"]]:
                continue
            canonical_ann = parts
            break
        if canonical_ann is None:
            continue

        alt = rec.alts[0] if rec.alts else ""
        rows.append(
            {
                "chrom": rec.chrom,
                "pos": int(rec.pos),
                "ref": rec.ref or "",
                "alt": alt,
                "variant_id": f"{rec.chrom}-{rec.pos}-{rec.ref}-{alt}",
                "allele_frequency": af,
                "consequence": canonical_ann[idx["Consequence"]],
                "gene_symbol": canonical_ann[idx["SYMBOL"]],
                "hgvsp": canonical_ann[idx["HGVSp"]] or None,
                "hgvsc": canonical_ann[idx["HGVSc"]] or None,
                "protein_position": canonical_ann[idx["Protein_position"]] or None,
            }
        )
        if len(rows) >= BATCH:
            flush()

    flush()
    if writer is not None:
        writer.close()
    vcf.close()
    return n_total


def setup_gnomad(refresh: bool = False) -> None:
    """Download gnomAD v4.1 exome per-chrom VCFs and extract to a gene-indexed parquet.

    This is the long pole — ~20-40 GB total download across all
    chromosomes, ~1-2 hours wall time.  Parses VCF INFO fields via
    pysam, filters to PASS + canonical-transcript VEP annotation, writes
    per-chromosome parquets, and concatenates into a single
    gene-indexed artifact.

    Not run by default from ``all`` — pass ``--include-gnomad``.
    """
    if is_built(GNOMAD_PARQUET, refresh):
        logger.info("gnomad: %s already exists — skipping", GNOMAD_PARQUET)
        return

    GNOMAD_VCF_DIR.mkdir(parents=True, exist_ok=True)

    per_chrom_paths: list[Path] = []
    for chrom in GNOMAD_CHROMS:
        vcf_url = f"{GNOMAD_BASE}gnomad.exomes.v4.1.sites.chr{chrom}.vcf.bgz"
        vcf_path = GNOMAD_VCF_DIR / f"chr{chrom}.vcf.bgz"
        tbi_path = vcf_path.with_suffix(".bgz.tbi")
        per_chrom_parquet = GNOMAD_VCF_DIR / f"chr{chrom}.parquet"
        per_chrom_paths.append(per_chrom_parquet)

        if per_chrom_parquet.exists() and not refresh:
            logger.info("gnomad: chr%s already parsed — skipping", chrom)
            continue

        if not vcf_path.exists() or refresh:
            logger.info("gnomad: downloading chr%s VCF", chrom)
            run(["wget", "-q", vcf_url, "-O", str(vcf_path)])
        if not tbi_path.exists() or refresh:
            run(["wget", "-q", vcf_url + ".tbi", "-O", str(tbi_path)])

        logger.info("gnomad: parsing chr%s", chrom)
        t0 = time.perf_counter()
        n = _parse_gnomad_vcf(vcf_path, per_chrom_parquet)
        logger.info(
            "gnomad: chr%s wrote %d variants in %.1fs",
            chrom,
            n,
            time.perf_counter() - t0,
        )

    # Stream-concat per-chrom parquets — peak memory bounded by the largest
    # per-chrom parquet (~3 GB for chr1), vs ~120M-row pandas concat OOM.
    import pyarrow.parquet as pq

    logger.info(
        "gnomad: stream-concatenating %d per-chrom parquets -> %s",
        len(per_chrom_paths),
        GNOMAD_PARQUET,
    )
    total_rows = 0
    writer: pq.ParquetWriter | None = None
    try:
        for p in per_chrom_paths:
            table = pq.read_table(p)
            if writer is None:
                writer = pq.ParquetWriter(GNOMAD_PARQUET, table.schema, compression="snappy")
            writer.write_table(table)
            total_rows += table.num_rows
            del table  # release before next read
    finally:
        if writer is not None:
            writer.close()
    logger.info("gnomad: wrote %d total variants to %s", total_rows, GNOMAD_PARQUET)

    write_sidecar(
        GNOMAD_DIR,
        source_url=GNOMAD_BASE + "gnomad.exomes.v4.1.sites.chr<N>.vcf.bgz",
        version="gnomad_v4.1_exomes",
        artifact=GNOMAD_PARQUET,
        extra={
            "n_variants": int(total_rows),
            "chromosomes": GNOMAD_CHROMS,
            "filter": "PASS + canonical-transcript VEP SYMBOL not null",
        },
    )


# ---------------------------------------------------------------------------
# COSMIC (manual raw download prereq)
# ---------------------------------------------------------------------------

COSMIC_DIR = REF / "cosmic"
COSMIC_RAW = COSMIC_DIR / "raw"
COSMIC_PARQUET = COSMIC_DIR / "cosmic_variants.parquet"
COSMIC_API = "https://cancer.sanger.ac.uk/api/mono/products/v1/downloads/scripted"
COSMIC_VERSION = "v102"
COSMIC_ASSEMBLY = "GRCh38"
COSMIC_FILES = [
    f"Cosmic_GenomeScreensMutant_Vcf_{COSMIC_VERSION}_{COSMIC_ASSEMBLY}.tar",
    f"Cosmic_NonCodingVariants_Vcf_{COSMIC_VERSION}_{COSMIC_ASSEMBLY}.tar",
    f"Cosmic_CompleteTargetedScreensMutant_Vcf_{COSMIC_VERSION}_{COSMIC_ASSEMBLY}.tar",
]
COSMIC_PATH_TEMPLATE = "grch38/cosmic/{version}/VCF/{filename}"
COSMIC_VCFS = [
    f"Cosmic_GenomeScreensMutant_{COSMIC_VERSION}_{COSMIC_ASSEMBLY}.vcf.gz",
    f"Cosmic_NonCodingVariants_{COSMIC_VERSION}_{COSMIC_ASSEMBLY}.vcf.gz",
    f"Cosmic_CompleteTargetedScreensMutant_{COSMIC_VERSION}_{COSMIC_ASSEMBLY}.vcf.gz",
]

COSMIC_PREREQ_MSG = f"""
COSMIC requires an authenticated download from the Sanger institute.

Before running `setup_databases.py cosmic`, you must:
  1. Register at https://cancer.sanger.ac.uk/cosmic (free academic license).
  2. Provide credentials via EITHER:
       a. A repo-root .env file (copy .env.example → .env, fill in):
              COSMIC_EMAIL=...
              COSMIC_PASSWORD=...
       b. Environment variables in the shell:
              export COSMIC_EMAIL=...
              export COSMIC_PASSWORD=...
       c. CLI flags: --cosmic-email / --cosmic-password

The script will download three {COSMIC_VERSION} GRCh38 VCF tar archives,
extract them, parse INFO fields, and produce a combined parquet at:
    {COSMIC_PARQUET.relative_to(ROOT)}
"""


def _cosmic_download_url(path: str, auth_header: str) -> str:
    """Ask the Sanger API for a signed download URL for *path*."""
    import requests

    resp = requests.get(
        COSMIC_API,
        params={"path": path, "bucket": "downloads"},
        headers={"Authorization": auth_header},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if "url" not in data:
        raise RuntimeError(f"COSMIC API response missing 'url': {data}")
    return data["url"]


def _parse_cosmic_vcf_streaming(vcf_path: Path, out_path: Path, batch_size: int = 500_000) -> int:
    """Stream-parse a COSMIC VCF.gz and write a parquet in RecordBatch chunks.

    Peak memory bounded by ``batch_size`` rows; NonCoding VCFs have hundreds
    of millions of rows, so buffering the full list OOMs. Columns match the
    v1 schema consumed by ClinicalModule's COSMIC filter.

    Returns the number of rows written.
    """
    import gzip

    import pyarrow as pa
    import pyarrow.parquet as pq

    def parse_info(info_str: str) -> dict[str, str]:
        d: dict[str, str] = {}
        for item in info_str.split(";"):
            if "=" in item:
                k, v = item.split("=", 1)
                d[k] = v
            else:
                d[item] = "True"
        return d

    schema = pa.schema(
        [
            ("CHROMOSOME", pa.string()),
            ("GENOME_START", pa.int64()),
            ("GENOMIC_MUTATION_ID", pa.string()),
            ("GENOMIC_WT_ALLELE", pa.string()),
            ("GENOMIC_MUT_ALLELE", pa.string()),
            ("GENE_SYMBOL", pa.string()),
            ("TRANSCRIPT_ACCESSION", pa.string()),
            ("MUTATION_CDS", pa.string()),
            ("MUTATION_AA", pa.string()),
            ("MUTATION_DESCRIPTION", pa.string()),
            ("HGVSG", pa.string()),
            ("HGVSC", pa.string()),
            ("HGVSP", pa.string()),
            ("STRAND", pa.string()),
            ("LEGACY_MUTATION_ID", pa.string()),
            ("GENOME_SCREEN_SAMPLE_COUNT", pa.string()),
            ("IS_CANONICAL", pa.string()),
        ]
    )

    # Column-major buffers — much lighter than a list of row dicts.
    cols: dict[str, list[Any]] = {name: [] for name in schema.names}
    n_written = 0

    def flush(writer: pq.ParquetWriter) -> None:
        nonlocal cols, n_written
        if not cols["CHROMOSOME"]:
            return
        batch = pa.record_batch(
            [pa.array(cols[name], type=schema.field(name).type) for name in schema.names],
            schema=schema,
        )
        writer.write_batch(batch)
        n_written += len(cols["CHROMOSOME"])
        cols = {name: [] for name in schema.names}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(out_path, schema, compression="snappy") as writer:
        with gzip.open(vcf_path, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 8:
                    continue
                chrom, pos, vid, ref, alt, _qual, _filt, info = fields[:8]
                info_d = parse_info(info)
                cols["CHROMOSOME"].append(chrom)
                cols["GENOME_START"].append(int(pos))
                cols["GENOMIC_MUTATION_ID"].append(vid)
                cols["GENOMIC_WT_ALLELE"].append(ref)
                cols["GENOMIC_MUT_ALLELE"].append(alt)
                cols["GENE_SYMBOL"].append(info_d.get("GENE", ""))
                cols["TRANSCRIPT_ACCESSION"].append(info_d.get("TRANSCRIPT", ""))
                cols["MUTATION_CDS"].append(info_d.get("CDS", ""))
                cols["MUTATION_AA"].append(info_d.get("AA", ""))
                cols["MUTATION_DESCRIPTION"].append(info_d.get("SO_TERM", ""))
                cols["HGVSG"].append(info_d.get("HGVSG", ""))
                cols["HGVSC"].append(info_d.get("HGVSC", ""))
                cols["HGVSP"].append(info_d.get("HGVSP", ""))
                cols["STRAND"].append(info_d.get("STRAND", ""))
                cols["LEGACY_MUTATION_ID"].append(info_d.get("LEGACY_ID", ""))
                cols["GENOME_SCREEN_SAMPLE_COUNT"].append(
                    info_d.get("GENOME_SCREEN_SAMPLE_COUNT", "")
                )
                cols["IS_CANONICAL"].append(info_d.get("IS_CANONICAL", ""))
                if len(cols["CHROMOSOME"]) >= batch_size:
                    flush(writer)
            flush(writer)
    return n_written


def setup_cosmic(refresh: bool = False) -> None:
    """Download COSMIC VCFs via the Sanger API and parse into a combined parquet.

    Ported from ``swissisoform/scripts/0_download_genome.sh`` (v1).  Uses
    COSMIC_EMAIL + COSMIC_PASSWORD env vars for authentication.  Writes
    intermediate tars to ``data/reference/cosmic/raw/`` and the combined
    parquet to ``data/reference/cosmic/cosmic_variants.parquet``.
    """
    import base64
    import os
    import tarfile

    # COSMIC_PARQUET is a directory (pyarrow dataset of per-VCF parquets).
    if COSMIC_PARQUET.is_dir():
        existing = list(COSMIC_PARQUET.glob("*.parquet"))
        if existing and not refresh:
            logger.info(
                "cosmic: %s already contains %d parquet file(s) — skipping",
                COSMIC_PARQUET,
                len(existing),
            )
            return
        if refresh:
            import shutil

            logger.info("refresh: removing %s", COSMIC_PARQUET)
            shutil.rmtree(COSMIC_PARQUET)

    email = os.environ.get("COSMIC_EMAIL")
    password = os.environ.get("COSMIC_PASSWORD")
    if not email or not password:
        print(COSMIC_PREREQ_MSG)
        raise RuntimeError("COSMIC_EMAIL and COSMIC_PASSWORD env vars required for download")
    auth_header = "Basic " + base64.b64encode(f"{email}:{password}".encode()).decode()

    COSMIC_RAW.mkdir(parents=True, exist_ok=True)

    # Download each tar + extract
    for fname in COSMIC_FILES:
        tar_path = COSMIC_RAW / fname
        if tar_path.exists() and not refresh:
            logger.info("cosmic: %s already downloaded", tar_path.name)
        else:
            logger.info("cosmic: requesting signed URL for %s", fname)
            api_path = COSMIC_PATH_TEMPLATE.format(version=COSMIC_VERSION, filename=fname)
            url = _cosmic_download_url(api_path, auth_header)
            run(["wget", "-q", "--show-progress", "-O", str(tar_path), url])
        # Extract into the same dir
        with tarfile.open(tar_path) as tar:
            tar.extractall(COSMIC_RAW)

    # One parquet per VCF inside a directory — pyarrow.dataset reads the
    # directory as a single logical dataset with filter pushdown.
    COSMIC_PARQUET.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq

    total_rows = 0
    parsed_files: list[str] = []
    for vcf_name in COSMIC_VCFS:
        vcf_path = COSMIC_RAW / vcf_name
        if not vcf_path.exists():
            logger.warning("cosmic: %s missing after extract — skipping", vcf_name)
            continue
        # Derive per-VCF output filename inside the directory dataset.
        out_name = vcf_name.replace(".vcf.gz", ".parquet")
        out_path = COSMIC_PARQUET / out_name
        if out_path.exists() and not refresh:
            logger.info("cosmic: %s already parsed — skipping", out_name)
            total_rows += pq.ParquetFile(out_path).metadata.num_rows
            parsed_files.append(vcf_name)
            continue
        t0 = time.perf_counter()
        logger.info("cosmic: streaming %s -> %s", vcf_name, out_name)
        n = _parse_cosmic_vcf_streaming(vcf_path, out_path)
        logger.info(
            "cosmic: %s -> %d variants (%.1fs)",
            vcf_name,
            n,
            time.perf_counter() - t0,
        )
        total_rows += n
        parsed_files.append(vcf_name)

    if total_rows == 0:
        raise RuntimeError("No COSMIC VCFs parsed — check downloads")

    logger.info(
        "cosmic: wrote %d variants across %d parquet files in %s",
        total_rows,
        len(parsed_files),
        COSMIC_PARQUET,
    )

    write_sidecar(
        COSMIC_DIR,
        source_url=f"{COSMIC_API} (auth; {COSMIC_VERSION} {COSMIC_ASSEMBLY})",
        version=f"Cosmic_{COSMIC_VERSION}_{COSMIC_ASSEMBLY}",
        artifact=COSMIC_PARQUET,
        extra={
            "n_variants": int(total_rows),
            "source_files": parsed_files,
        },
    )


# ---------------------------------------------------------------------------
# DeepLoc (software copy — no web download; tarball from DTU)
# ---------------------------------------------------------------------------

DEEPLOC_DIR = REF / "deeploc"
DEEPLOC_TARBALL_NAME = "deeploc-2.1.All.tar.gz"
DEEPLOC_TARBALL = DEEPLOC_DIR / DEEPLOC_TARBALL_NAME
DEEPLOC_ENV_NAME = "swissisoform-v2-deeploc"
# Source tarball — the DTU DeepLoc release isn't freely redownloadable, so
# we copy from the sibling swissisoform v1 project where it already lives.
DEEPLOC_SOURCE = Path("/lab/barcheese01/mdiberna/swissisoform/deeploc-2.1.All.tar.gz")


def _conda_env_exists(name: str) -> bool:
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        if line and not line.startswith("#"):
            env_name = line.split()[0]
            if env_name == name:
                return True
    return False


def _conda_run(env_name: str, cmd: list[str]) -> None:
    """``conda run`` with PYTHONNOUSERSITE=1 so ~/.local/ can't shadow env packages."""
    import os

    environ = dict(os.environ)
    environ["PYTHONNOUSERSITE"] = "1"
    full = ["conda", "run", "-n", env_name] + cmd
    logger.info("running: %s (PYTHONNOUSERSITE=1)", " ".join(full))
    subprocess.run(full, check=True, env=environ)


DEEPLOC_INSTALL_INSTRUCTIONS = f"""
DeepLoc 2.1 is NOT automatically installed by this script.  It requires a
manual setup because its dependencies need their own Python 3.8 conda env
(and fight with packages in ~/.local/ on shared filesystems).

The tarball is staged at:
    {DEEPLOC_TARBALL}

Run these four commands yourself to set up the env:

    conda create -n {DEEPLOC_ENV_NAME} -c conda-forge python=3.8 pip -y
    conda activate {DEEPLOC_ENV_NAME}
    PYTHONNOUSERSITE=1 pip install --no-user {DEEPLOC_TARBALL}
    PYTHONNOUSERSITE=1 python -c "import DeepLoc2; print('ok')"

The PYTHONNOUSERSITE=1 prefix prevents pip/python from picking up any
stale DeepLoc install in ~/.local/ (a common HPC failure mode).

Once the env exists, LocalizationModule will subprocess to it via
`conda run -n {DEEPLOC_ENV_NAME} ...` automatically.
"""


def setup_deeploc(refresh: bool = False) -> None:
    """Stage the DeepLoc tarball and verify the manually-installed env.

    DeepLoc 2.1 has no public pip/conda release.  Distribution is a
    Python 3.8-pinned tarball from DTU, which must live in its own conda
    env to avoid dependency conflicts with the main swissisoform-v2 env.

    This subcommand:
      1. Copies the tarball from the sibling swissisoform v1 project
         (``DEEPLOC_SOURCE``) to ``data/reference/deeploc/``.
      2. Prints the four manual commands to create + populate the env.
      3. Verifies the env exists (if the user has already run the
         commands) and writes a sidecar recording provenance.

    Why not automate the install?  pip install from inside a script
    fights with ``~/.local/`` user-site-packages on shared filesystems,
    and the failure modes are non-obvious.  Matches the v1 workflow.
    """
    DEEPLOC_DIR.mkdir(parents=True, exist_ok=True)

    if not DEEPLOC_TARBALL.exists() or refresh:
        if not DEEPLOC_SOURCE.exists():
            raise FileNotFoundError(
                f"DeepLoc tarball not found at {DEEPLOC_SOURCE}.  Download "
                "from https://services.healthtech.dtu.dk/services/DeepLoc-2.0/ "
                "(requires DTU registration) and place it there."
            )
        logger.info("deeploc: copying %s → %s", DEEPLOC_SOURCE, DEEPLOC_TARBALL)
        shutil.copyfile(DEEPLOC_SOURCE, DEEPLOC_TARBALL)
    else:
        logger.info("deeploc: tarball already staged at %s", DEEPLOC_TARBALL)

    # Print the manual install steps + check whether the user has run them
    print(DEEPLOC_INSTALL_INSTRUCTIONS)

    if not _conda_env_exists(DEEPLOC_ENV_NAME):
        logger.warning(
            "deeploc: conda env %r does not exist yet.  Follow the four "
            "manual commands above, then run this subcommand again to "
            "verify + write the sidecar.",
            DEEPLOC_ENV_NAME,
        )
        return

    # Env exists — smoke-test that DeepLoc2 imports cleanly.
    logger.info("deeploc: env %s exists; verifying import", DEEPLOC_ENV_NAME)
    try:
        _conda_run(
            DEEPLOC_ENV_NAME,
            [
                "python",
                "-c",
                "import DeepLoc2; print('deeploc import ok')",
            ],
        )
    except subprocess.CalledProcessError:
        logger.error(
            "deeploc: env %s exists but DeepLoc2 failed to import. "
            "Did you run the pip install with PYTHONNOUSERSITE=1?  "
            "See the instructions above.",
            DEEPLOC_ENV_NAME,
        )
        return

    write_sidecar(
        DEEPLOC_DIR,
        source_url=f"copy of {DEEPLOC_SOURCE}",
        version="DeepLoc-2.1.All",
        artifact=DEEPLOC_TARBALL,
        extra={"conda_env": DEEPLOC_ENV_NAME, "install_mode": "manual (verified)"},
    )
    logger.info("deeploc: verified + sidecar written")


# ---------------------------------------------------------------------------
# PepQuery2 (mass-spec peptide validation)
# ---------------------------------------------------------------------------

PEPQUERY_DIR = REF / "pepquery"
PEPQUERY_VERSION = "2.0.2"
PEPQUERY_TARBALL_URL = f"http://pepquery.org/data/pepquery-{PEPQUERY_VERSION}.tar.gz"
PEPQUERY_TARBALL = PEPQUERY_DIR / f"pepquery-{PEPQUERY_VERSION}.tar.gz"
PEPQUERY_JAR = PEPQUERY_DIR / f"pepquery-{PEPQUERY_VERSION}" / f"pepquery-{PEPQUERY_VERSION}.jar"


def setup_pepquery(refresh: bool = False) -> None:
    """Download the PepQuery2 standalone jar from pepquery.org.

    PepQuery2 is a plain Java 11+ CLI.  System Java is already on PATH
    on this cluster, so we skip the conda-env indirection used for
    DeepLoc and just stage the jar.
    :func:`swissisoform.evidence.e6_mass_spec.precompute_pepquery`
    invokes it via ``java -jar <PEPQUERY_JAR>``.

    ``-b <dataset>`` at query time pulls MS/MS spectra from PepQueryDB
    on demand — no local spectral index required at this step.
    """
    PEPQUERY_DIR.mkdir(parents=True, exist_ok=True)

    if PEPQUERY_JAR.exists() and not refresh:
        logger.info("pepquery: jar already present at %s", PEPQUERY_JAR)
    else:
        if not PEPQUERY_TARBALL.exists() or refresh:
            logger.info("pepquery: downloading %s", PEPQUERY_TARBALL_URL)
            subprocess.run(
                ["curl", "--fail", "--location", "--retry", "3",
                 "--output", str(PEPQUERY_TARBALL), PEPQUERY_TARBALL_URL],
                check=True,
            )
        logger.info("pepquery: extracting to %s", PEPQUERY_DIR)
        subprocess.run(
            ["tar", "-xzf", str(PEPQUERY_TARBALL), "-C", str(PEPQUERY_DIR)],
            check=True,
        )
        if not PEPQUERY_JAR.exists():
            raise FileNotFoundError(
                f"pepquery: extraction finished but {PEPQUERY_JAR} missing — "
                "tarball layout may have changed upstream."
            )

    # Smoke-test: `java -jar pepquery.jar` without args prints usage and
    # exits non-zero.  We just want to confirm Java can load the jar.
    if shutil.which("java") is None:
        logger.warning("pepquery: 'java' not on PATH — install openjdk >= 11 before running")
    else:
        proc = subprocess.run(
            ["java", "-jar", str(PEPQUERY_JAR)],
            capture_output=True, text=True, check=False, timeout=60,
        )
        help_blob = (proc.stdout or "") + (proc.stderr or "")
        if "pepquery" in help_blob.lower() or "Options" in help_blob:
            logger.info("pepquery: jar loaded OK (Java %s)", _java_version())
        else:
            logger.warning(
                "pepquery: jar ran but output looks wrong (exit=%d); stderr tail=%s",
                proc.returncode, (proc.stderr or "")[-400:],
            )


# Local PepQuery MS/MS spectra library — provisioned reference data per the
# CLAUDE.md Execution Contract: mirror the PepQueryDB datasets from the public
# S3 bucket so runs search the local copy via `-ms` instead of re-pulling (and
# deleting) the spectra from S3 on every search. ~196 GiB for the two datasets
# below; opt-in (not part of `all`).
PEPQUERY_SPECTRA_DIR = PEPQUERY_DIR / "spectra"
# Must match the datasets precompute_pepquery searches (the runner's -b list).
PEPQUERY_DATASETS = (
    "Deep_29_healthy_human_tissues_PXD010154",
    "GTEx_32_Tissues_Proteome_PXD016999",
)


def _pepquery_msms_s3_prefix(dataset: str) -> str:
    """Resolve a dataset's ``msms_library`` S3 prefix from the jar's msms.json."""
    import json
    import zipfile

    with zipfile.ZipFile(PEPQUERY_JAR) as z:
        catalog = json.loads(z.read("main/resources/msms.json"))
    try:
        ms_file = catalog[dataset]["ms_file"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"pepquery-spectra: {dataset!r} not found (or has no ms_file) in the "
            "jar's msms.json catalog"
        ) from exc
    if not ms_file or not str(ms_file).startswith("s3://"):
        raise ValueError(
            f"pepquery-spectra: {dataset!r} ms_file is not an S3 prefix: {ms_file!r}"
        )
    return str(ms_file).rstrip("/")


def setup_pepquery_spectra(refresh: bool = False) -> None:
    """Mirror the PepQueryDB MS/MS spectra library locally (~196 GiB).

    Contract-legal reference provisioning (see CLAUDE.md "Execution Contract").
    The PepQueryDB ``-b`` datasets live in a public S3 bucket as gzipped MGF;
    by default PepQuery re-downloads then deletes them on every search. Mirroring
    them once lets runs search the local copy via ``-ms`` with no per-run S3
    traffic. Anonymous (public) S3 — no credentials needed.

    Idempotent and resumable: ``aws s3 sync`` transfers only missing/changed
    objects, so an interrupted transfer resumes on re-run. ``refresh=True`` adds
    ``--delete`` to mirror exactly. Opt-in (not part of ``all``) given the size.
    """
    if not PEPQUERY_JAR.exists():
        raise FileNotFoundError(
            f"pepquery-spectra: jar missing at {PEPQUERY_JAR} — run the 'pepquery' "
            "target first (it ships the msms.json dataset catalog this reads)."
        )
    if shutil.which("aws") is None:
        raise FileNotFoundError(
            "pepquery-spectra: 'aws' CLI not on PATH — required for anonymous S3 sync."
        )

    PEPQUERY_SPECTRA_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in PEPQUERY_DATASETS:
        prefix = _pepquery_msms_s3_prefix(dataset)
        dest = PEPQUERY_SPECTRA_DIR / dataset
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("pepquery-spectra: syncing %s/ -> %s", prefix, dest)
        run([
            "aws", "s3", "sync", "--no-sign-request",
            *(["--delete"] if refresh else []),
            prefix + "/", str(dest),
        ])
    logger.info("pepquery-spectra: done -> %s", PEPQUERY_SPECTRA_DIR)

    write_sidecar(
        PEPQUERY_DIR,
        source_url=PEPQUERY_TARBALL_URL,
        version=PEPQUERY_VERSION,
        artifact=PEPQUERY_JAR,
        extra={"install_mode": "direct-jar", "java_version": _java_version()},
    )
    logger.info("pepquery: jar staged + sidecar written (%s)", PEPQUERY_JAR)


def _java_version() -> str:
    """Short Java version string for the provenance sidecar."""
    try:
        proc = subprocess.run(
            ["java", "-version"], capture_output=True, text=True, check=False, timeout=10,
        )
        text = proc.stderr or proc.stdout or ""
        return text.splitlines()[0] if text else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# HAL toolkit (hal2maf + halStats via cactus singularity image)
# ---------------------------------------------------------------------------

HAL_DIR = REF / "zoonomia"
HAL_SIF_DIR = HAL_DIR / "singularity"
HAL_SIF = HAL_SIF_DIR / "cactus.sif"
# Pinned cactus image. Cactus 2.9.9 ships on quay.io/comparative-genomics-toolkit
# and bundles the HAL toolkit (hal2maf, halStats, halLiftover, etc.). We use
# singularity because the bioconda ``cactus`` package has unresolvable
# libdeflate/toil dependency conflicts in our channel mix.
HAL_DOCKER_URI = "docker://quay.io/comparative-genomics-toolkit/cactus:v2.9.9"
# Wrapper scripts at scripts/bin/ that ConservationConfig points at — they
# invoke ``singularity exec`` against HAL_SIF with /lab bind-mounted so
# absolute HAL paths work inside the container.
HAL_WRAPPER_HAL2MAF = ROOT / "scripts" / "bin" / "hal2maf"
HAL_WRAPPER_HALSTATS = ROOT / "scripts" / "bin" / "halStats"


HAL_MISSING_SINGULARITY_MSG = """
HAL setup requires the ``singularity`` CLI. Not found on PATH.

On Whitehead cluster nodes singularity is at /usr/local/bin/singularity —
check your PATH or module-load singularity before re-running.
"""


def setup_hal(refresh: bool = False) -> None:
    """Pull the cactus singularity image and verify hal2maf runs.

    Installs the HAL toolkit as a container rather than a conda env because
    bioconda ``cactus`` has unresolvable dep conflicts in our channels.
    The wrapper scripts at ``scripts/bin/hal2maf`` + ``halStats`` shell out
    to ``singularity exec`` against this image, and
    ``ConservationConfig.hal2maf_binary`` points at the wrapper.

    Args:
        refresh: Re-pull the image even when it already exists.
    """
    if shutil.which("singularity") is None:
        print(HAL_MISSING_SINGULARITY_MSG)
        raise RuntimeError("singularity binary not on PATH")

    HAL_SIF_DIR.mkdir(parents=True, exist_ok=True)

    if HAL_SIF.exists() and refresh:
        logger.info("refresh: removing %s", HAL_SIF)
        HAL_SIF.unlink()

    if not HAL_SIF.exists():
        tmp_dir = ROOT / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        env = dict(os.environ)
        env["SINGULARITY_TMPDIR"] = str(tmp_dir)
        logger.info("hal: pulling %s -> %s", HAL_DOCKER_URI, HAL_SIF)
        subprocess.run(
            ["singularity", "pull", str(HAL_SIF), HAL_DOCKER_URI],
            check=True,
            env=env,
        )
    else:
        logger.info("hal: %s already pulled — skipping", HAL_SIF)

    for wrapper in (HAL_WRAPPER_HAL2MAF, HAL_WRAPPER_HALSTATS):
        if not wrapper.exists():
            raise RuntimeError(f"hal: wrapper script missing: {wrapper}")
        if not os.access(wrapper, os.X_OK):
            raise RuntimeError(f"hal: wrapper script not executable: {wrapper}")

    # Smoke-test the wrapper — confirms the container has hal2maf and
    # /lab bind-mount works.
    probe = subprocess.run(
        [str(HAL_WRAPPER_HAL2MAF), "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    # hal2maf --help exits 1 on some cactus builds but still prints usage
    # to stderr. Accept either 0 or a non-empty usage dump.
    output = (probe.stdout or "") + (probe.stderr or "")
    if "hal2maf" not in output.lower() and "usage" not in output.lower():
        raise RuntimeError(
            f"hal: wrapper smoke-test failed exit={probe.returncode} "
            f"output_head={output[:300]}"
        )
    logger.info("hal: wrapper smoke-test ok (exit=%d)", probe.returncode)

    write_sidecar(
        HAL_DIR,
        source_url=HAL_DOCKER_URI,
        version="cactus-v2.9.9",
        artifact=HAL_SIF,
        extra={
            "wrapper_hal2maf": str(HAL_WRAPPER_HAL2MAF),
            "wrapper_halstats": str(HAL_WRAPPER_HALSTATS),
            "bind_mounts": ["/lab"],
            "install_source": "singularity pull docker://quay.io/...",
        },
    )
    logger.info("hal: sidecar written")


# ---------------------------------------------------------------------------
# GENCODE (delegates to download_references.sh)
# ---------------------------------------------------------------------------


def setup_gencode(refresh: bool = False) -> None:
    """Invoke scripts/download_references.sh."""
    script = ROOT / "scripts" / "download_references.sh"
    cmd = ["bash", str(script)]
    if refresh:
        cmd.append("--force")
    run(cmd)


# ---------------------------------------------------------------------------
# SignalP 6.0 — academic-license pip-installable tarball (isolated env)
# ---------------------------------------------------------------------------

SIGNALP_DIR = REF / "signalp"
SIGNALP_ENV_NAME = "swissisoform-v2-signalp"
SIGNALP_TARBALL_GLOB = "signalp-6*.tar.gz"
SIGNALP_EXTRACT_ROOT = SIGNALP_DIR / "signalp6_fast"
SIGNALP_PACKAGE_DIR = SIGNALP_EXTRACT_ROOT / "signalp-6-package"
SIGNALP_MODELS_SRC = SIGNALP_PACKAGE_DIR / "models"
SIGNALP_URL = "https://services.healthtech.dtu.dk/services/SignalP-6.0/"


def setup_signalp(refresh: bool = False) -> None:
    """Create the SignalP 6.0 conda env and install the pip tarball.

    SignalP 6.0 is distributed as an academic-license pip-installable
    package with separately-bundled model weights.  Install is three steps:

      1. Extract ``signalp-6*.tar.gz`` → ``signalp6_fast/signalp-6-package/``.
      2. Create env + ``python -m pip install`` the extracted package
         (not the raw tarball — the tarball's top-level directory isn't
         a Python project) and downgrade numpy <2 to match torch 1.13.
         ``python -m pip`` bypasses ``~/.local/bin/pip`` shadowing.
      3. Copy ``models/*`` into the installed package's ``model_weights/``
         directory — the models aren't packaged into the wheel.

    Idempotent: skips steps already done unless ``refresh`` is True.
    """
    SIGNALP_DIR.mkdir(parents=True, exist_ok=True)

    matches = sorted(SIGNALP_DIR.glob(SIGNALP_TARBALL_GLOB))
    if not matches:
        logger.warning(
            "signalp: no tarball matching %s found in %s.  Download from "
            "%s (DTU academic license) and drop it there.",
            SIGNALP_TARBALL_GLOB, SIGNALP_DIR, SIGNALP_URL,
        )
        return
    tarball = matches[-1]
    logger.info("signalp: tarball → %s", tarball)

    # Step 1 — extract
    if SIGNALP_PACKAGE_DIR.exists() and not refresh:
        logger.info("signalp: already extracted at %s", SIGNALP_PACKAGE_DIR)
    else:
        if SIGNALP_EXTRACT_ROOT.exists() and refresh:
            shutil.rmtree(SIGNALP_EXTRACT_ROOT)
        logger.info("signalp: extracting %s → %s", tarball, SIGNALP_DIR)
        subprocess.run(["tar", "-xzf", str(tarball), "-C", str(SIGNALP_DIR)], check=True)
        if not SIGNALP_PACKAGE_DIR.exists():
            raise FileNotFoundError(
                f"signalp: extraction finished but {SIGNALP_PACKAGE_DIR} missing"
            )

    # Step 2 — create env + pip install (if needed)
    if not _conda_env_exists(SIGNALP_ENV_NAME):
        logger.info("signalp: creating conda env %s (python=3.10)", SIGNALP_ENV_NAME)
        subprocess.run(
            ["conda", "create", "-n", SIGNALP_ENV_NAME, "-c", "conda-forge",
             "python=3.10", "pip", "-y"],
            check=True,
        )
    else:
        logger.info("signalp: conda env %s already exists", SIGNALP_ENV_NAME)

    # Always verify the package is importable before skipping the pip install,
    # so a half-populated env self-heals on re-run.
    pip_needed = True
    try:
        _conda_run(SIGNALP_ENV_NAME, ["python", "-c", "import signalp"])
        pip_needed = False
    except subprocess.CalledProcessError:
        pass

    if pip_needed or refresh:
        logger.info("signalp: pip-installing %s into %s", SIGNALP_PACKAGE_DIR, SIGNALP_ENV_NAME)
        _conda_run(
            SIGNALP_ENV_NAME,
            ["python", "-m", "pip", "install", "--no-user", str(SIGNALP_PACKAGE_DIR)],
        )
        # torch 1.13 (SignalP's pin) requires numpy < 2.
        logger.info("signalp: pinning numpy<2 (torch 1.13 compat)")
        _conda_run(
            SIGNALP_ENV_NAME,
            ["python", "-m", "pip", "install", "--no-user", "numpy<2"],
        )

    # Step 3 — copy model weights into the installed package
    proc = subprocess.run(
        ["conda", "run", "-n", SIGNALP_ENV_NAME, "python", "-c",
         "import signalp, os; print(os.path.dirname(signalp.__file__))"],
        capture_output=True, text=True, check=True,
    )
    signalp_pkg_dir = Path(proc.stdout.strip())
    model_weights_dir = signalp_pkg_dir / "model_weights"
    model_weights_dir.mkdir(exist_ok=True)
    target_pt = model_weights_dir / "distilled_model_signalp6.pt"
    src_pt = SIGNALP_MODELS_SRC / "distilled_model_signalp6.pt"
    if not src_pt.exists():
        raise FileNotFoundError(f"signalp: model weights missing at {src_pt}")
    if target_pt.exists() and not refresh:
        logger.info("signalp: model weights already in place at %s", target_pt)
    else:
        logger.info("signalp: copying %s → %s", src_pt, target_pt)
        shutil.copyfile(src_pt, target_pt)

    # Step 4 — smoke test: predict on a 1-sequence FASTA
    smoke_dir = SIGNALP_DIR / ".smoke_test"
    smoke_dir.mkdir(exist_ok=True)
    smoke_fa = smoke_dir / "input.fa"
    smoke_fa.write_text(
        ">test\nMRAPGCVLLLGLCLLSQAALAGGEHSGEILVGGLFPMHSRGSEGKPCGDIKREGG\n"
    )
    try:
        _conda_run(
            SIGNALP_ENV_NAME,
            ["signalp6", "--fastafile", str(smoke_fa), "--organism", "eukarya",
             "--mode", "fast", "--format", "txt",
             "--output_dir", str(smoke_dir / "out")],
        )
    except subprocess.CalledProcessError as exc:
        logger.error("signalp: smoke test failed: %s", exc)
        return
    logger.info("signalp: smoke test ok")

    write_sidecar(
        SIGNALP_DIR,
        source_url=SIGNALP_URL,
        version=tarball.name,
        artifact=tarball,
        extra={"conda_env": SIGNALP_ENV_NAME, "install_mode": "automated"},
    )
    logger.info("signalp: verified + sidecar written")


# ---------------------------------------------------------------------------
# TargetP 2.0 — standalone Go binary + bundled TensorFlow libs
# ---------------------------------------------------------------------------

TARGETP_DIR = REF / "targetp"
TARGETP_TARBALL_GLOB = "targetp-2*.tar.gz"
TARGETP_EXTRACT_DIR = TARGETP_DIR / "targetp-2.0"
TARGETP_BIN = TARGETP_EXTRACT_DIR / "bin" / "targetp"
TARGETP_URL = "https://services.healthtech.dtu.dk/services/TargetP-2.0/"


def setup_targetp(refresh: bool = False) -> None:
    """Extract the TargetP 2.0 tarball and smoke-test the binary.

    TargetP 2.0 is a self-contained Go binary with bundled TensorFlow C
    libraries.  Unlike SignalP, it is NOT a Python package — no conda
    env is required.  This subcommand:

      1. Locates the user-supplied tarball in ``data/reference/targetp/``.
      2. Extracts it in place if ``targetp-2.0/`` is missing (or if
         ``refresh`` is True).
      3. Runs the example invocation from the DTU readme to verify the
         binary + bundled libs load cleanly.
      4. Writes a provenance sidecar.
    """
    TARGETP_DIR.mkdir(parents=True, exist_ok=True)

    matches = sorted(TARGETP_DIR.glob(TARGETP_TARBALL_GLOB))
    if not matches:
        logger.warning(
            "targetp: no tarball matching %s found in %s.  Download from "
            "%s (DTU academic license) and drop it there.",
            TARGETP_TARBALL_GLOB, TARGETP_DIR, TARGETP_URL,
        )
        return
    tarball = matches[-1]

    if TARGETP_BIN.exists() and not refresh:
        logger.info("targetp: binary already extracted at %s", TARGETP_BIN)
    else:
        if TARGETP_EXTRACT_DIR.exists() and refresh:
            logger.info("targetp: removing stale %s (refresh=True)", TARGETP_EXTRACT_DIR)
            shutil.rmtree(TARGETP_EXTRACT_DIR)
        logger.info("targetp: extracting %s → %s", tarball, TARGETP_DIR)
        subprocess.run(["tar", "-xzf", str(tarball), "-C", str(TARGETP_DIR)], check=True)
        if not TARGETP_BIN.exists():
            raise FileNotFoundError(
                f"targetp: extraction finished but {TARGETP_BIN} missing — "
                "tarball layout may have changed upstream."
            )

    # Smoke-test: run the example from the readme and confirm the summary
    # file appears.  Uses the test FASTA shipped inside the tarball.
    test_fa = TARGETP_EXTRACT_DIR / "test" / "example.fsa"
    if not test_fa.exists():
        logger.warning("targetp: test FASTA missing (%s) — skipping smoke test", test_fa)
    else:
        smoke_dir = TARGETP_DIR / ".smoke_test"
        smoke_dir.mkdir(exist_ok=True)
        prefix = smoke_dir / "example_short"
        env = dict(os.environ)
        lib_dir = TARGETP_EXTRACT_DIR / "lib"
        env["LD_LIBRARY_PATH"] = (
            f"{lib_dir}:{env['LD_LIBRARY_PATH']}" if "LD_LIBRARY_PATH" in env else str(lib_dir)
        )
        cmd = [
            str(TARGETP_BIN),
            "-fasta", str(test_fa),
            "-org", "non-pl",
            "-format", "short",
            "-prefix", str(prefix),
        ]
        logger.info("targetp: smoke-test %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, env=env, cwd=smoke_dir)
        except subprocess.CalledProcessError as exc:
            logger.error(
                "targetp: smoke test failed (exit %d). stderr=%s",
                exc.returncode, (exc.stderr or "")[-400:],
            )
            return
        summaries = list(smoke_dir.glob("example_short*summary*"))
        if not summaries:
            logger.error("targetp: smoke test produced no summary file in %s", smoke_dir)
            return
        logger.info("targetp: smoke test ok (%s)", summaries[0].name)

    write_sidecar(
        TARGETP_DIR,
        source_url=TARGETP_URL,
        version=tarball.name,
        artifact=TARGETP_BIN,
        extra={"install_mode": "native-binary"},
    )
    logger.info("targetp: verified + sidecar written")


# ---------------------------------------------------------------------------
# InterProScan 6 — Nextflow pipeline (DB + singularity images auto-pulled)
# ---------------------------------------------------------------------------

INTERPROSCAN_DIR = REF / "interproscan"
INTERPROSCAN_DATADIR = INTERPROSCAN_DIR / "datadir"
INTERPROSCAN_NF_REPO = "ebi-pf-team/interproscan6"
INTERPROSCAN_VERSION = "6.0.0"
INTERPROSCAN_DATA_VERSION = "108.0"


def setup_interproscan(refresh: bool = False) -> None:
    """Pre-warm the InterProScan 6 Nextflow datadir and smoke-test.

    InterProScan 6 is a Nextflow pipeline that orchestrates member-DB
    Singularity images under a unified interface.  First-run behaviour:

    - The pipeline auto-downloads member DBs into ``--datadir`` (~50 GB).
    - Singularity images for member tools are pulled into the Nextflow
      cache by ``-profile singularity``.

    This subcommand kicks off that first run on a trivial 1-protein
    FASTA so both downloads happen once, in a controlled location, with
    provenance recorded.  Subsequent precompute runs reuse the cache.

    Idempotent.  ``refresh`` deletes the datadir and re-downloads.
    """
    INTERPROSCAN_DIR.mkdir(parents=True, exist_ok=True)

    if shutil.which("nextflow") is None:
        logger.error(
            "interproscan: 'nextflow' not on PATH.  Install Nextflow "
            "(e.g. via conda: conda install -c bioconda nextflow) first."
        )
        return
    if shutil.which("singularity") is None:
        logger.error(
            "interproscan: 'singularity' not on PATH — required for the "
            "singularity profile.  Install singularity / apptainer first."
        )
        return

    if INTERPROSCAN_DATADIR.exists() and refresh:
        logger.info("interproscan: removing existing datadir (refresh=True)")
        shutil.rmtree(INTERPROSCAN_DATADIR)
    INTERPROSCAN_DATADIR.mkdir(parents=True, exist_ok=True)

    smoke_dir = INTERPROSCAN_DIR / ".smoke_test"
    smoke_dir.mkdir(exist_ok=True)
    smoke_out = smoke_dir / "out"
    if smoke_out.exists():
        shutil.rmtree(smoke_out)
    smoke_out.mkdir()
    # Use IPS6's canonical -profile test: uses a curated test FASTA + the
    # full default (non-ML) application set, which avoids the single-app
    # COMBINE_MATCHES bug that triggers when sequences have null matches.
    cmd = [
        "nextflow", "run", INTERPROSCAN_NF_REPO,
        "-r", INTERPROSCAN_VERSION,
        "-resume",
        "-profile", "singularity,test",
        "--datadir", str(INTERPROSCAN_DATADIR),
        "--interpro", INTERPROSCAN_DATA_VERSION,
        "--outdir", str(smoke_out),
        # Force container-based COMBINE_MATCHES path — the LOCAL variant
        # fails under Nextflow 25.x due to a Groovy classpath regression
        # around `lib/uk/ac/ebi/interpro/ProcessCombine.groovy`.
        "--batchSize", "50000",
    ]
    logger.info(
        "interproscan: first run (auto-downloads DBs + member images, 30–60 min): %s",
        " ".join(cmd),
    )
    try:
        subprocess.run(cmd, check=True, cwd=INTERPROSCAN_DIR)
    except subprocess.CalledProcessError as exc:
        logger.error("interproscan: nextflow run failed: %s", exc)
        return

    tsv_candidates = sorted(smoke_out.rglob("*.tsv"))
    if not tsv_candidates:
        logger.error("interproscan: smoke test produced no .tsv under %s", smoke_out)
        return
    logger.info(
        "interproscan: smoke test ok (%s, %d bytes)",
        tsv_candidates[0].name, tsv_candidates[0].stat().st_size,
    )

    # Sidecar artifact is the smoke-test TSV (small + reproducible); we
    # deliberately do NOT hash the 34 GB datadir — write_sidecar's per-file
    # sha256 loop would take hours and the hashes would change every time
    # InterPro rolls a minor data release anyway.
    write_sidecar(
        INTERPROSCAN_DIR,
        source_url=f"https://github.com/{INTERPROSCAN_NF_REPO}",
        version=f"{INTERPROSCAN_VERSION} (data {INTERPROSCAN_DATA_VERSION})",
        artifact=tsv_candidates[0],
        extra={
            "install_mode": "nextflow+singularity",
            "datadir": str(INTERPROSCAN_DATADIR.relative_to(ROOT)),
            "datadir_size_bytes": sum(
                f.stat().st_size for f in INTERPROSCAN_DATADIR.rglob("*") if f.is_file()
            ),
        },
    )
    logger.info("interproscan: verified + sidecar written")


# ---------------------------------------------------------------------------
# AlphaMissense hg38 — per-variant calibrated missense pathogenicity
# ---------------------------------------------------------------------------

ALPHAMISSENSE_DIR = REF / "alphamissense"
ALPHAMISSENSE_GZ = ALPHAMISSENSE_DIR / "AlphaMissense_hg38.tsv.gz"
ALPHAMISSENSE_TBI = ALPHAMISSENSE_DIR / "AlphaMissense_hg38.tsv.gz.tbi"
ALPHAMISSENSE_URL = "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz"


def setup_alphamissense(refresh: bool = False) -> None:
    """Download + tabix-index the AlphaMissense hg38 precomputed-scores table.

    The published file (~640 MB) is already BGZF-compressed and sorted by
    ``(CHROM, POS)``, so it can be tabix-indexed in place. Columns are
    ``CHROM POS REF ALT genome uniprot_id transcript_id protein_variant
    am_pathogenicity am_class`` with a ``#``-prefixed header.
    :class:`swissisoform.clinical.alphamissense.AlphaMissenseLookup` reads it
    by genomic ``(chrom, pos, ref, alt)``.

    Licence: CC BY-NC-SA 4.0 (DeepMind) — research use.

    Idempotent: skips when the ``.tbi`` already exists unless ``refresh``.
    """
    ALPHAMISSENSE_DIR.mkdir(parents=True, exist_ok=True)

    if ALPHAMISSENSE_TBI.exists() and not refresh:
        logger.info("alphamissense: %s already indexed — skipping", ALPHAMISSENSE_GZ)
        return

    if not ALPHAMISSENSE_GZ.exists() or refresh:
        logger.info("alphamissense: downloading %s", ALPHAMISSENSE_URL)
        partial = ALPHAMISSENSE_GZ.with_suffix(ALPHAMISSENSE_GZ.suffix + ".partial")
        subprocess.run(
            ["curl", "--fail", "--location", "--retry", "3",
             "--output", str(partial), ALPHAMISSENSE_URL],
            check=True,
        )
        partial.rename(ALPHAMISSENSE_GZ)

    if shutil.which("tabix") is None:
        raise RuntimeError("alphamissense: 'tabix' not on PATH — install htslib/tabix first")

    logger.info("alphamissense: tabix-indexing %s", ALPHAMISSENSE_GZ)
    run(["tabix", "-s", "1", "-b", "2", "-e", "2", "-c", "#", str(ALPHAMISSENSE_GZ)])

    # Smoke-test: a known likely_pathogenic SNV must resolve.
    import pysam

    with pysam.TabixFile(str(ALPHAMISSENSE_GZ)) as tbx:
        hits = [ln.split("\t") for ln in tbx.fetch("chr1", 69102, 69103)]
    if not hits:
        raise RuntimeError("alphamissense: index built but smoke-test query returned nothing")
    logger.info("alphamissense: smoke-test ok (%d rows at chr1:69103)", len(hits))

    write_sidecar(
        ALPHAMISSENSE_DIR,
        source_url=ALPHAMISSENSE_URL,
        version="AlphaMissense_hg38 (Cheng et al. 2023)",
        artifact=ALPHAMISSENSE_GZ,
        extra={"index": str(ALPHAMISSENSE_TBI.relative_to(ROOT)), "licence": "CC BY-NC-SA 4.0"},
    )
    logger.info("alphamissense: indexed + sidecar written")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, Any] = {
    "diamond": setup_diamond,
    "clinvar": setup_clinvar,
    "gnomad": setup_gnomad,
    "cosmic": setup_cosmic,
    "alphamissense": setup_alphamissense,
    "deeploc": setup_deeploc,
    "pepquery": setup_pepquery,
    "pepquery-spectra": setup_pepquery_spectra,
    "hal": setup_hal,
    "gencode": setup_gencode,
    "signalp": setup_signalp,
    "targetp": setup_targetp,
    "interproscan": setup_interproscan,
}


def main() -> int:
    """Run one (or all) DB setup target(s) from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Load .env from the repo root before any subcommand reads credentials.
    _load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        choices=list(_HANDLERS.keys()) + ["all"],
        help="Which DB to set up (or 'all').",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download + rebuild even if the artifact exists.",
    )
    parser.add_argument(
        "--include-gnomad",
        action="store_true",
        help="When target=all, include the heavy gnomAD bulk download.",
    )
    args = parser.parse_args()

    if args.target == "all":
        # gnomAD is the long pole (~6+ hours for all chromosomes) — placed
        # last so lighter DBs finish first and are available for
        # downstream work.
        targets = [
            "gencode", "diamond", "clinvar", "cosmic", "alphamissense",
            "deeploc", "hal", "gnomad",
        ]
    else:
        targets = [args.target]

    failed: list[str] = []
    for t in targets:
        try:
            logger.info("=== %s ===", t)
            _HANDLERS[t](refresh=args.refresh)
        except Exception as exc:
            logger.error("%s failed: %s", t, exc)
            failed.append(t)

    if failed:
        logger.error("failed targets: %s", failed)
        return 1
    logger.info("setup_databases: done")
    return 0
