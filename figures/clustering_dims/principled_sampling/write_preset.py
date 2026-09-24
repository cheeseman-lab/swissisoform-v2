#!/usr/bin/env python
"""Emit ``presets/cheeseman50.toml`` from the coreset selection.

`coreset_50.csv` identifies isoforms by ``tis_id`` — ``chrom:orf_start:strand:
codon:tid`` — but a preset's ``[[isoforms]]`` entries join the combined catalog
on ``Tid`` / ``GenomePos`` / ``StartCodon``, and ``GenomePos`` is a *range*
(``chr17:48071434-48101392:-``). The range cannot be reconstructed from a
``tis_id``, so each pick is looked up in the combined catalog and its real
``GenomePos`` copied out.

The ORF start sits at the high end of the range on the minus strand and the low
end on the plus strand, which is what makes the join exact rather than fuzzy.

Generated rather than hand-written so the preset stays derivable if the coreset
is rebuilt.

Usage:
    python figures/clustering_dims/principled_sampling/write_preset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "feature_space"))
from export_feature_catalog import ROOT  # noqa: E402

HERE = Path(__file__).resolve().parent
CORESET_CSV = HERE / "coreset_50.csv"
COMBINED = ROOT / "data" / "output" / "filtered" / "all_samples_combined.parquet"
OUT_TOML = ROOT / "presets" / "cheeseman50.toml"

ORF_LABEL = {
    "extended": "Extended",
    "truncated": "Truncated",
    "uorf": "uORF",
    "uoorf": "uoORF",
    "internal_oof": "internal out-of-frame",
    "3utr_orf": "3'UTR ORF",
}
SOURCE_LABEL = {
    "anchor": "anchor",
    "rare_fill": "rare-type fill",
    "sampler": "sampler",
}


def parse_tis_id(tis_id: str) -> dict[str, str]:
    """Split ``chrom:pos:strand:codon:tid`` into its parts."""
    chrom, pos, strand, codon, tid = tis_id.split(":")
    return {"chrom": chrom, "pos": int(pos), "strand": strand, "codon": codon, "tid": tid}


def resolve(coreset: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    """Attach each pick's catalog ``GenomePos`` by exact coordinate match."""
    parts = pd.DataFrame([parse_tis_id(t) for t in coreset["tis_id"]], index=coreset.index)
    picks = pd.concat([coreset, parts], axis=1)

    cat = combined.copy()
    split = cat["GenomePos"].str.split(":", expand=True)
    span = split[1].str.split("-", expand=True)
    cat["chrom"], cat["strand"] = split[0], split[2]
    cat["lo"], cat["hi"] = span[0].astype(int), span[1].astype(int)
    # The ORF start is the high coordinate on minus strand, low on plus.
    cat["orf_start"] = cat["hi"].where(cat["strand"] == "-", cat["lo"])

    merged = picks.merge(
        cat[
            ["Tid", "StartCodon", "chrom", "strand", "orf_start", "GenomePos", "RecatTISType"]
        ].drop_duplicates(),
        left_on=["tid", "codon", "chrom", "strand", "pos"],
        right_on=["Tid", "StartCodon", "chrom", "strand", "orf_start"],
        how="left",
    )
    missing = merged[merged["GenomePos"].isna()]
    if len(missing):
        raise SystemExit(
            f"{len(missing)} picks did not resolve to a combined-catalog row:\n"
            + missing[["gene_name", "tis_id"]].to_string(index=False)
        )
    if len(merged) != len(coreset):
        raise SystemExit(f"join produced {len(merged)} rows from {len(coreset)} picks")
    return merged


def render(picks: pd.DataFrame) -> str:
    """Render the preset, grouped by gene, in the cheeseman13 style."""
    counts = picks["orf_type"].value_counts()
    by_source = picks["source"].value_counts()
    lines = [
        f"# The LLM-tuning coreset: 50 isoforms across {picks['gene_name'].nunique()} genes.",
        "#",
        "# Explicit [[isoforms]] picks, like cheeseman13 — each entry's",
        "# (tid, genome_pos, start_codon) is matched against the combined catalog to pull",
        "# its sequence, and the canonical Annotated rows for these genes are kept",
        "# automatically. Runs multi-sample.",
        "#",
        "# Composition:",
        f"#   {by_source.get('anchor', 0)} anchors        hand-curated, static",
        f"#   {by_source.get('rare_fill', 0)} rare-type fill  n=1 sampler within each of",
        "#                  uorf / internal_oof / 3utr_orf / uoorf — the four types",
        "#                  the global sampler never reaches, being 0.3-4.9% of the",
        "#                  pool. Yields each type's max, min and most typical.",
        f"#   {by_source.get('sampler', 0)} sampler        n=6 sampler over the MFA feature",
        "#                  space (all-ORF, 6,462 x 391), one isoform per gene.",
        "#",
        "# ORF types: " + ", ".join(f"{ORF_LABEL[t]} {n}" for t, n in counts.items()),
        "#",
        "# Regenerate with figures/clustering_dims/principled_sampling/write_preset.py",
        "# after rebuilding coreset_50.csv.",
        'run_name = "cheeseman50"',
        "",
        "# Production reproducibility bar, matching cheeseman13. At 1 every TIS passes D1",
        "# by construction, handing the LLM a criterion that is True everywhere.",
        "min_cell_lines = 3",
        "",
    ]

    for gene in sorted(picks["gene_name"].unique()):
        for _, r in picks[picks["gene_name"] == gene].sort_values("pos").iterrows():
            lines += [
                "[[isoforms]]",
                f'gene = "{gene}"',
                f'tid = "{r["tid"]}"',
                f'genome_pos = "{r["GenomePos"]}"',
                f'start_codon = "{r["codon"]}"'
                f"  # {ORF_LABEL.get(r['orf_type'], r['orf_type'])}"
                f" — {SOURCE_LABEL.get(r['source'], r['source'])}",
                "",
            ]
    return "\n".join(lines)


def main() -> None:
    """Resolve the coreset against the catalog and write the preset."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT_TOML), help="preset path to write")
    args = ap.parse_args()

    coreset = pd.read_csv(CORESET_CSV)
    combined = pd.read_parquet(
        COMBINED, columns=["Symbol", "Tid", "GenomePos", "StartCodon", "RecatTISType"]
    )
    picks = resolve(coreset, combined)

    # Every pick must be an alternative TIS; an Annotated row here would mean the
    # coreset picked a canonical, which the preset keeps automatically anyway.
    annotated = picks[picks["RecatTISType"] == "Annotated"]
    if len(annotated):
        raise SystemExit(
            f"{len(annotated)} picks are Annotated rows:\n"
            + annotated[["gene_name", "tis_id"]].to_string(index=False)
        )

    Path(args.out).write_text(render(picks))
    print(f"wrote {len(picks)} isoforms across {picks['gene_name'].nunique()} genes to {args.out}")
    print(picks["source"].value_counts().to_string())
    print(picks["orf_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
