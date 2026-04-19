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
    gencode  — delegates to scripts/download_references.sh

Each subcommand writes:
    data/reference/<db>/<artifact>
    data/reference/<db>/_setup.json   # provenance sidecar

By default, subcommands skip work if the artifact already exists. Pass
``--refresh`` to force re-download + re-build.

Usage:
    python scripts/setup_databases.py all
    python scripts/setup_databases.py diamond
    python scripts/setup_databases.py clinvar --refresh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


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


# Load .env from repo root (one level up from scripts/) before any subcmd.
_load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("setup_databases")

ROOT = Path(__file__).parent.parent
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
    """Write ``_setup.json`` recording provenance for a DB artifact."""
    db_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "source_url": source_url,
        "version": version,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact": str(artifact.relative_to(ROOT)),
        "artifact_size_bytes": artifact.stat().st_size,
        "artifact_sha256": _sha256(artifact),
    }
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
    run([
        "diamond", "makedb",
        "--in", str(DIAMOND_FASTA),
        "-d", str(DIAMOND_DB.with_suffix("")),  # diamond appends .dmnd
    ])

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
CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)
# Columns we keep + the downstream-standardized names we map them to
_CLINVAR_KEEP = [
    "Type", "Name", "GeneSymbol", "ClinicalSignificance", "RS# (dbSNP)",
    "PhenotypeIDS", "PhenotypeList", "Origin", "Assembly", "Chromosome",
    "Start", "Stop", "ReferenceAllele", "AlternateAllele",
    "VariationID", "HGNC_ID",
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
        CLINVAR_GZ, sep="\t", low_memory=False, na_values=["-", "", "na"],
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
        *df.shape, CLINVAR_PARQUET, time.perf_counter() - t0,
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
GNOMAD_BASE = (
    "https://gnomad-public-us-east-1.s3.amazonaws.com/release/4.1/vcf/exomes/"
)
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
    import pysam
    import pyarrow as pa
    import pyarrow.parquet as pq

    vcf = pysam.VariantFile(str(vcf_path))
    vep_fields = _parse_vep_header(vcf)
    idx = {name: i for i, name in enumerate(vep_fields)}

    keep_cols = [
        "Consequence", "SYMBOL", "CANONICAL", "HGVSp", "HGVSc", "Protein_position",
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
        rows.append({
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
        })
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
    import pandas as pd

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
            chrom, n, time.perf_counter() - t0,
        )

    logger.info("gnomad: concatenating %d per-chrom parquets", len(per_chrom_paths))
    df = pd.concat([pd.read_parquet(p) for p in per_chrom_paths], ignore_index=True)
    df.to_parquet(GNOMAD_PARQUET, index=False)
    logger.info("gnomad: wrote %d total variants to %s", len(df), GNOMAD_PARQUET)

    write_sidecar(
        GNOMAD_DIR,
        source_url=GNOMAD_BASE + "gnomad.exomes.v4.1.sites.chr<N>.vcf.bgz",
        version="gnomad_v4.1_exomes",
        artifact=GNOMAD_PARQUET,
        extra={
            "n_variants": int(len(df)),
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
COSMIC_PATH_TEMPLATE = (
    "grch38/cosmic/{version}/VCF/{filename}"
)
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


def _parse_cosmic_vcf(vcf_path: Path) -> "pd.DataFrame":
    """Parse a COSMIC VCF.gz into a standardized DataFrame.

    Columns match the v1 schema consumed by the ClinicalModule COSMIC
    filter (``GENE_SYMBOL``, ``CHROMOSOME``, ``GENOME_START``,
    ``GENOMIC_WT_ALLELE``, ``GENOMIC_MUT_ALLELE``, HGVSP/HGVSC, etc.).
    """
    import gzip
    import pandas as pd

    def parse_info(info_str: str) -> dict[str, str]:
        d: dict[str, str] = {}
        for item in info_str.split(";"):
            if "=" in item:
                k, v = item.split("=", 1)
                d[k] = v
            else:
                d[item] = "True"
        return d

    rows: list[dict[str, Any]] = []
    with gzip.open(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom, pos, vid, ref, alt, _qual, _filt, info = fields[:8]
            info_d = parse_info(info)
            rows.append({
                "CHROMOSOME": chrom,
                "GENOME_START": int(pos),
                "GENOMIC_MUTATION_ID": vid,
                "GENOMIC_WT_ALLELE": ref,
                "GENOMIC_MUT_ALLELE": alt,
                "GENE_SYMBOL": info_d.get("GENE", ""),
                "TRANSCRIPT_ACCESSION": info_d.get("TRANSCRIPT", ""),
                "MUTATION_CDS": info_d.get("CDS", ""),
                "MUTATION_AA": info_d.get("AA", ""),
                "MUTATION_DESCRIPTION": info_d.get("SO_TERM", ""),
                "HGVSG": info_d.get("HGVSG", ""),
                "HGVSC": info_d.get("HGVSC", ""),
                "HGVSP": info_d.get("HGVSP", ""),
                "STRAND": info_d.get("STRAND", ""),
                "LEGACY_MUTATION_ID": info_d.get("LEGACY_ID", ""),
                "GENOME_SCREEN_SAMPLE_COUNT": info_d.get(
                    "GENOME_SCREEN_SAMPLE_COUNT", ""
                ),
                "IS_CANONICAL": info_d.get("IS_CANONICAL", ""),
            })
    return pd.DataFrame(rows)


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
    import pandas as pd

    if is_built(COSMIC_PARQUET, refresh):
        logger.info("cosmic: %s already exists — skipping", COSMIC_PARQUET)
        return

    email = os.environ.get("COSMIC_EMAIL")
    password = os.environ.get("COSMIC_PASSWORD")
    if not email or not password:
        print(COSMIC_PREREQ_MSG)
        raise RuntimeError(
            "COSMIC_EMAIL and COSMIC_PASSWORD env vars required for download"
        )
    auth_header = "Basic " + base64.b64encode(
        f"{email}:{password}".encode()
    ).decode()

    COSMIC_RAW.mkdir(parents=True, exist_ok=True)

    # Download each tar + extract
    for fname in COSMIC_FILES:
        tar_path = COSMIC_RAW / fname
        if tar_path.exists() and not refresh:
            logger.info("cosmic: %s already downloaded", tar_path.name)
        else:
            logger.info("cosmic: requesting signed URL for %s", fname)
            api_path = COSMIC_PATH_TEMPLATE.format(
                version=COSMIC_VERSION, filename=fname
            )
            url = _cosmic_download_url(api_path, auth_header)
            run(["wget", "-q", "--show-progress", "-O", str(tar_path), url])
        # Extract into the same dir
        with tarfile.open(tar_path) as tar:
            tar.extractall(COSMIC_RAW)

    # Parse each VCF → DataFrame → combine
    frames: list[pd.DataFrame] = []
    for vcf_name in COSMIC_VCFS:
        vcf_path = COSMIC_RAW / vcf_name
        if not vcf_path.exists():
            logger.warning("cosmic: %s missing after extract — skipping", vcf_name)
            continue
        t0 = time.perf_counter()
        logger.info("cosmic: parsing %s", vcf_name)
        df = _parse_cosmic_vcf(vcf_path)
        logger.info(
            "cosmic: %s → %d variants (%.1fs)",
            vcf_name, len(df), time.perf_counter() - t0,
        )
        frames.append(df)

    if not frames:
        raise RuntimeError("No COSMIC VCFs parsed — check downloads")

    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["CHROMOSOME", "GENOME_START"]
    )
    combined.to_parquet(COSMIC_PARQUET, compression="snappy", index=False)
    logger.info(
        "cosmic: wrote %d variants to %s", len(combined), COSMIC_PARQUET
    )

    write_sidecar(
        COSMIC_DIR,
        source_url=f"{COSMIC_API} (auth; {COSMIC_VERSION} {COSMIC_ASSEMBLY})",
        version=f"Cosmic_{COSMIC_VERSION}_{COSMIC_ASSEMBLY}",
        artifact=COSMIC_PARQUET,
        extra={
            "n_variants": int(len(combined)),
            "source_files": COSMIC_FILES,
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
DEEPLOC_SOURCE = Path(
    "/lab/barcheese01/mdiberna/swissisoform/deeploc-2.1.All.tar.gz"
)


def _conda_env_exists(name: str) -> bool:
    result = subprocess.run(
        ["conda", "env", "list"], capture_output=True, text=True, check=True
    )
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
        _conda_run(DEEPLOC_ENV_NAME, [
            "python", "-c", "import DeepLoc2; print('deeploc import ok')",
        ])
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
# CLI
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, Any] = {
    "diamond": setup_diamond,
    "clinvar": setup_clinvar,
    "gnomad": setup_gnomad,
    "cosmic": setup_cosmic,
    "deeploc": setup_deeploc,
    "gencode": setup_gencode,
}


def main() -> int:
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
        targets = ["gencode", "diamond", "clinvar", "cosmic", "deeploc", "gnomad"]
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


if __name__ == "__main__":
    sys.exit(main())
