#!/usr/bin/env python
"""Merge one campaign's per-shard annotate outputs into data/output/<campaign>/.

Concatenates ``<campaign>_shard_<k>/all_paired.parquet`` for **exactly** the
shards ``0..n_shards-1`` named by the campaign's ``split_manifest.txt`` — never a
bare glob, which would sweep in leftovers from a previous split. Gene-sharding is
safe for scoring (every criterion is per-TIS/per-gene; ``min_cell_lines`` reads
``present_*`` columns already present in each row), so no cross-shard recompute
is needed.

Every shard is accounted for against the manifest and its gene list, and the
verdict is written to ``<campaign>/merge_report.tsv``:

    ok           merged, gene membership matches shard_meta.json
    unverified   merged, but the shard predates shard_meta.json
    stale        shard_meta.json disagrees with shards/shard_<k>.txt — NOT merged
    missing      no all_paired.parquet (never ran, or still running)
    unreadable   parquet present but failed to load (truncated mid-write)

The report also carries each shard's GPU-precompute hole rate. Phase C depends on
Phase B with ``afterany``, so a dead GPU chunk still lets annotation run: the
structure/PLM/SAE modules read the cache miss and emit ``status="no_cache"``.
Those fractions are the only way that failure is visible in the merged parquet —
a dead chunk is protein-hash-ordered, so its holes land in a handful of
gene shards and stand out against a campaign-wide baseline of zero.

A merge that is not complete does **not** overwrite ``all_paired.parquet``: it
lands at ``all_paired.partial.parquet`` and the command exits nonzero, so no
downstream stage can mistake a partial catalog for the real one. Refill the
missing shards and re-run, or pass ``--allow-partial`` to promote it deliberately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd

OUTPUT_ROOT = Path("data/output")

# Every GPU-fed module writes this literal when the hash-keyed precompute is
# missing, which is exactly the afterany GPU-hole signature — and distinct from
# its legitimate non-ok statuses (too_long, no_diff_region, uniform_plddt, …).
NO_CACHE = "no_cache"
GPU_STATUS_COLUMNS = {
    "structure": "isoform_structure_status",
    "plm": "isoform_plm_vep_status",
    "sae": "isoform_sae_status",
}

MERGEABLE = ("ok", "unverified")
NEEDS_REFILL = ("stale", "missing", "unreadable")


def read_manifest(path: Path) -> dict[str, str]:
    """Parse the tab-separated key/value ``split_manifest.txt``."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "\t" in line:
            k, v = line.split("\t", 1)
            out[k] = v
    return out


def shard_status(shard_dir: Path, shard_list: Path) -> tuple[str, pd.DataFrame | None]:
    """Classify one shard output and return its frame when it is usable."""
    paired = shard_dir / "all_paired.parquet"
    if not paired.exists():
        return "missing", None
    meta_path = shard_dir / "shard_meta.json"
    status = "unverified"
    if meta_path.exists():
        expected = hashlib.sha1(shard_list.read_bytes()).hexdigest()
        recorded = json.loads(meta_path.read_text()).get("shard_list_sha1")
        if recorded != expected:
            return "stale", None
        status = "ok"
    try:
        df = pd.read_parquet(paired)
    except Exception as exc:  # truncated / half-written parquet
        print(f"  shard {shard_dir.name}: unreadable ({exc})")
        return "unreadable", None
    return status, df


def gpu_hole_fractions(df: pd.DataFrame) -> dict[str, float | str]:
    """Fraction of rows whose GPU-fed modules reported a missing precompute."""
    out: dict[str, float | str] = {}
    for label, col in GPU_STATUS_COLUMNS.items():
        out[f"frac_no_cache_{label}"] = (
            round(float((df[col] == NO_CACHE).mean()), 4) if col in df.columns else ""
        )
    return out


def merge_campaign(
    campaign: str,
    output_root: Path = OUTPUT_ROOT,
    allow_partial: bool = False,
) -> tuple[int, pd.DataFrame | None]:
    """Merge one campaign. Returns ``(exit_code, per-shard report)``."""
    out = output_root / campaign
    manifest_path = out / "split_manifest.txt"
    if not manifest_path.exists():
        print(f"ERROR: no split_manifest.txt for campaign {campaign} ({manifest_path})")
        return 2, None
    manifest = read_manifest(manifest_path)
    stamped = manifest.get("campaign")
    if stamped and stamped != campaign:
        print(f"ERROR: manifest belongs to campaign {stamped!r}, not {campaign!r}")
        return 2, None
    n_shards = int(manifest["n_shards"])

    rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    ok_dirs: list[Path] = []
    expected_genes: set[str] = set()
    merged_genes: set[str] = set()
    gene_origin: dict[str, int] = {}
    duplicates: set[str] = set()

    for k in range(n_shards):
        shard_list = out / "shards" / f"shard_{k}.txt"
        shard_genes = {g for g in shard_list.read_text().split() if g}
        expected_genes |= shard_genes
        shard_dir = output_root / f"{campaign}_shard_{k}"
        status, df = shard_status(shard_dir, shard_list)
        present: set[str] = set()
        row: dict[str, object] = {
            "shard": k,
            "status": status,
            "n_rows": 0,
            "n_genes_expected": len(shard_genes),
            "n_genes_present": 0,
            "n_genes_dropped": "",
            "n_genes_unexpected": "",
            **{f"frac_no_cache_{label}": "" for label in GPU_STATUS_COLUMNS},
        }
        if df is not None:
            frames.append(df)
            ok_dirs.append(shard_dir)
            present = set(df["gene_name"].dropna().unique()) if "gene_name" in df else set()
            merged_genes |= present
            for g in present:
                if g in gene_origin and gene_origin[g] != k:
                    duplicates.add(g)
                gene_origin.setdefault(g, k)
            row.update(
                n_rows=len(df),
                n_genes_present=len(present),
                n_genes_dropped=len(shard_genes - present),
                n_genes_unexpected=len(present - shard_genes),
                **gpu_hole_fractions(df),
            )
        rows.append(row)

    report = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "merge_report.tsv"
    report.to_csv(report_path, sep="\t", index=False)

    counts = report["status"].value_counts().to_dict()
    orphans = sorted(
        d.name for d in output_root.glob(f"{campaign}_shard_*")
        if d.name.rsplit("_shard_", 1)[1].isdigit()
        and int(d.name.rsplit("_shard_", 1)[1]) >= n_shards
    )
    unexpected = merged_genes - expected_genes
    holed = [
        (int(r["shard"]), label)
        for _, r in report.iterrows()
        for label in GPU_STATUS_COLUMNS
        if isinstance(r[f"frac_no_cache_{label}"], float) and r[f"frac_no_cache_{label}"] > 0
    ]

    print(f"campaign {campaign}: {n_shards} shards in manifest")
    for status in (*MERGEABLE, *NEEDS_REFILL):
        if counts.get(status):
            print(f"  {status:<11} {counts[status]}")
    if any(counts.get(s) for s in NEEDS_REFILL):
        bad = report[report["status"].isin(NEEDS_REFILL)]["shard"].tolist()
        print(f"  shards needing a refill: {bad}")
    if orphans:
        print(f"  ORPHAN output dirs beyond shard {n_shards - 1} (stale split): {orphans}")
    if duplicates:
        print(
            f"  DUPLICATE genes merged from >1 shard: {len(duplicates)} "
            f"e.g. {sorted(duplicates)[:5]}"
        )
    if unexpected:
        print(
            f"  UNEXPECTED genes not in any shard list: {len(unexpected)} "
            f"e.g. {sorted(unexpected)[:5]}"
        )
    if holed:
        print(f"  GPU PRECOMPUTE HOLES ({NO_CACHE}) in {len(holed)} shard/module pairs:")
        for k, label in holed[:10]:
            frac = report.loc[report["shard"] == k, f"frac_no_cache_{label}"].iloc[0]
            print(f"    shard {k} {label}: {frac:.1%} of rows")
        if len(holed) > 10:
            print(f"    … and {len(holed) - 10} more (see {report_path})")
        print("    → re-run the GPU array for the affected chunks, delete those shards'")
        print("      all_paired.parquet, resubmit the annotate array, then merge again.")
    print(
        f"  genes: {len(merged_genes)} merged / {len(expected_genes)} expected "
        f"({len(expected_genes - merged_genes)} dropped at collapse or lost with a failed shard)"
    )
    print(f"  per-shard report: {report_path}")

    if not frames:
        print("ERROR: no usable shard outputs — nothing merged")
        return 1, report

    # GPU holes are reported, not fatal: they are refilled by re-running Phase B,
    # and blocking on them would strand catalogs whose proteins legitimately fail
    # to fold. Missing rows, by contrast, cannot be reasoned about downstream.
    complete = not (
        any(counts.get(s) for s in NEEDS_REFILL) or orphans or duplicates or unexpected
    )
    df = pd.concat(frames, ignore_index=True)

    if complete or allow_partial:
        target = out / "all_paired.parquet"
        df.to_parquet(target)
        (out / "all_paired.partial.parquet").unlink(missing_ok=True)  # superseded
        n_gene_files = 0
        for d in ok_dirs:
            for g in d.glob("*_paired.parquet"):
                if g.name == "all_paired.parquet":
                    continue
                shutil.copyfile(g, out / g.name)
                n_gene_files += 1
        print(f"wrote {target} ({len(df)} rows), {n_gene_files} per-gene parquets copied")
        if not complete:
            print("WARNING: published under --allow-partial — this catalog has known gaps")
        return 0, report

    target = out / "all_paired.partial.parquet"
    df.to_parquet(target)
    print(
        f"INCOMPLETE: wrote {target} ({len(df)} rows) and left all_paired.parquet untouched.\n"
        "Refill the shards above and re-run this merge, or pass --allow-partial to publish as-is."
    )
    return 1, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--campaign", default=os.environ.get("SWISSISO_CAMPAIGN"),
        help="Campaign name (default: $SWISSISO_CAMPAIGN).",
    )
    ap.add_argument(
        "--output-root", type=Path, default=OUTPUT_ROOT,
        help="Root holding <campaign>/ and <campaign>_shard_<k>/ (default: data/output).",
    )
    ap.add_argument(
        "--allow-partial", action="store_true",
        help="Publish all_paired.parquet even when shards are missing or stale.",
    )
    a = ap.parse_args()
    if not a.campaign:
        print("ERROR: --campaign (or SWISSISO_CAMPAIGN) is required")
        return 2
    code, _ = merge_campaign(a.campaign, a.output_root, a.allow_partial)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
