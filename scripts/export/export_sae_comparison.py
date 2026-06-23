"""Export the per-pair isoform↔canonical SAE feature comparison.

Re-assembles a preset's genes (same load path as the runner), runs
the cross-protein :class:`~swissisoform.plm.sae_module.SAEFeatureModule` against
the cached 6B SAE features (``data/cache/sae_esmc/<SIZE>/``), and writes:

  1. ``sae_comparison.parquet`` — one row per (gene, tis_id): summary scalars
     (incl. the 4 counts) plus the FULL nested ``features_isoform_only`` /
     ``features_canonical_only`` / ``shared_feature_deltas`` lists (exhaustive).
  2. ``sae_feature_changes.parquet`` / ``.csv`` — the curated, ranked,
     metric-rich table: **top 30 per change_type per isoform**
     (isoform_only / canonical_only by max activation; shared by |delta_max|),
     one row per feature with rank + all metrics (max, mean, delta, prevalence
     for both sides + label/description).

No model inference — pure SAE-cache read + set ops. Reads everything else
read-only; writes only the NEW files above.

Usage:
    python scripts/export/export_sae_comparison.py --preset cheeseman13
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd  # noqa: E402

from swissisoform import runner  # noqa: E402, I001
from swissisoform.assembly import assemble_genes  # noqa: E402
from swissisoform.pipeline import UpstreamReference  # noqa: E402
from swissisoform.plm.embed import DEFAULT_MODEL_SIZE, protein_hash  # noqa: E402
from swissisoform.plm.sae import sae_cache_dir  # noqa: E402
from swissisoform.plm.sae_module import SAEFeatureModule  # noqa: E402
from swissisoform.references import (  # noqa: E402
    ALL_CELL_LINES,
    GENOME,
    GTF,
    PRESETS,
    PROTEIN,
    build_config,
)

# Long-form CSV columns (one row per feature × change_type).
# Top-N features per change_type, per isoform — the curated, ranked, metric-rich
# view (sae_feature_changes). Ranking: isoform_only / canonical_only by peak
# (max) activation; shared by |delta_max|. The lists in `ann` are already sorted
# that way, so we slice + enumerate.
TOP_N = 30
CHANGE_FIELDS = [
    "gene", "tis_id", "orf_type", "change_type", "rank", "feature_index", "label",
    "description",
    "isoform_max", "canonical_max", "delta_max",
    "isoform_mean", "canonical_mean", "delta_mean",
    "isoform_prevalence", "canonical_prevalence", "label_source",
]


def _change_rows(
    gene: str, tis_id: str, orf_type: str, ann: dict, top_n: int = TOP_N
) -> list[dict]:
    """Top-N features per change_type for one isoform, with full metrics + rank."""
    base = {"gene": gene, "tis_id": tis_id, "orf_type": orf_type}
    rows: list[dict] = []
    for rank, e in enumerate(ann["features_isoform_only"][:top_n], 1):
        rows.append({**base, "change_type": "isoform_only", "rank": rank,
                     "feature_index": e["feature_index"], "label": e["label"],
                     "description": e["description"],
                     "isoform_max": e["max"], "canonical_max": None, "delta_max": None,
                     "isoform_mean": e["mean"], "canonical_mean": None, "delta_mean": None,
                     "isoform_prevalence": e["prevalence"], "canonical_prevalence": None,
                     "label_source": e["label_source"]})
    for rank, e in enumerate(ann["features_canonical_only"][:top_n], 1):
        rows.append({**base, "change_type": "canonical_only", "rank": rank,
                     "feature_index": e["feature_index"], "label": e["label"],
                     "description": e["description"],
                     "isoform_max": None, "canonical_max": e["max"], "delta_max": None,
                     "isoform_mean": None, "canonical_mean": e["mean"], "delta_mean": None,
                     "isoform_prevalence": None, "canonical_prevalence": e["prevalence"],
                     "label_source": e["label_source"]})
    for rank, e in enumerate(ann["shared_feature_deltas"][:top_n], 1):
        rows.append({**base, "change_type": "shared", "rank": rank,
                     "feature_index": e["feature_index"], "label": e["label"],
                     "description": e["description"],
                     "isoform_max": e["isoform_max"], "canonical_max": e["canonical_max"],
                     "delta_max": e["delta_max"],
                     "isoform_mean": e["isoform_mean"], "canonical_mean": e["canonical_mean"],
                     "delta_mean": e["delta_mean"],
                     "isoform_prevalence": e["isoform_prevalence"],
                     "canonical_prevalence": e["canonical_prevalence"],
                     "label_source": e["label_source"]})
    return rows


def assemble_for_preset(preset_name: str):
    """Reproduce the runner's load + assemble for *preset_name*; return (spec, genes)."""
    spec = PRESETS[preset_name]
    ref = UpstreamReference.load(gtf_path=GTF, genome_fasta=GENOME, protein_fasta=PROTEIN)
    if "isoforms" in spec:
        combined = runner.load_combined(ALL_CELL_LINES, ref, build_config())
        final, gene_names = runner.restrict_to_isoforms(
            combined, runner.load_isoform_picks(spec["isoforms"])
        )
    else:
        final = runner.load_single_sample(spec.get("cell_lines", ["HeLa"])[0], ref)
        gene_names = spec["genes"]
    genes = assemble_genes(
        final, gene_names=gene_names, genome_fasta=GENOME, exon_skeletons=ref.exon_skeletons
    )
    return spec, genes


def main(argv: list[str] | None = None) -> int:
    """Assemble the preset, run the SAE comparison, write parquet + long CSV."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--preset", default="cheeseman13", choices=sorted(PRESETS))
    p.add_argument("--model-size", default=DEFAULT_MODEL_SIZE,
                   help="ESM-C SAE size selecting the cache dir (default %(default)s).")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Override the SAE feature cache dir.")
    p.add_argument("--outdir", type=Path, default=None,
                   help="Output dir (default data/output/<run_name>/).")
    args = p.parse_args(argv)

    cache_dir = args.cache_dir or sae_cache_dir(args.model_size)
    spec, genes = assemble_for_preset(args.preset)
    run_name = spec.get("run_name", args.preset)
    outdir = args.outdir or (ROOT / "data" / "output" / run_name)
    outdir.mkdir(parents=True, exist_ok=True)

    module = SAEFeatureModule(build_config(), cache_dir=cache_dir, model_size=args.model_size)

    pair_rows: list[dict] = []
    change_rows: list[dict] = []
    n_ok = n_nocache = 0
    for gene in sorted(genes, key=lambda g: g.gene_name):
        for site in gene.tis_sites:
            if not site.isoform_protein or not site.canonical_protein:
                continue
            ann = module.annotate_site(site)
            ot = site.orf_type.value if site.orf_type else ""
            if ann["status"] != "ok":
                n_nocache += ann["status"] == "no_cache"
                pair_rows.append({
                    "gene": gene.gene_name, "tis_id": site.tis_id, "orf_type": ot,
                    "status": ann["status"],
                    "isoform_protein_hash": protein_hash(site.isoform_protein),
                    "canonical_protein_hash": protein_hash(site.canonical_protein),
                })
                continue
            n_ok += 1
            pair_rows.append({
                "gene": gene.gene_name, "tis_id": site.tis_id, "orf_type": ot,
                "status": ann["status"],
                "n_isoform_total": ann["n_isoform_total"],
                "n_canonical_total": ann["n_canonical_total"],
                "n_isoform_only": ann["n_isoform_only"],
                "n_canonical_only": ann["n_canonical_only"],
                "n_shared": ann["n_shared"],
                "mean_abs_delta_shared": ann["mean_abs_delta_shared"],
                "top_gained_feature_index": ann["top_gained_feature_index"],
                "top_gained_feature_label": ann["top_gained_feature_label"],
                "top_gained_delta_max": ann["top_gained_delta_max"],
                "top_lost_feature_index": ann["top_lost_feature_index"],
                "top_lost_feature_label": ann["top_lost_feature_label"],
                "top_lost_delta_max": ann["top_lost_delta_max"],
                "features_isoform_only": ann["features_isoform_only"],
                "features_canonical_only": ann["features_canonical_only"],
                "shared_feature_deltas": ann["shared_feature_deltas"],
                "isoform_protein_hash": protein_hash(site.isoform_protein),
                "canonical_protein_hash": protein_hash(site.canonical_protein),
            })
            change_rows.extend(_change_rows(gene.gene_name, site.tis_id, ot, ann))

    pair_pq = outdir / "sae_comparison.parquet"
    pd.DataFrame(pair_rows).to_parquet(pair_pq, index=False)

    # Curated, ranked, metric-rich table: top-30 per change_type per isoform
    # (parquet + csv mirror). Full per-feature inventory stays in the nested
    # lists of sae_comparison.parquet.
    changes_pq = outdir / "sae_feature_changes.parquet"
    pd.DataFrame(change_rows, columns=CHANGE_FIELDS).to_parquet(changes_pq, index=False)
    changes_csv = outdir / "sae_feature_changes.csv"
    with open(changes_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CHANGE_FIELDS)
        w.writeheader()
        w.writerows(change_rows)

    print(f"pairs: {len(pair_rows)} ({n_ok} ok, {n_nocache} no_cache) → {pair_pq}")
    print(f"feature-change rows (top-{TOP_N}/type/isoform): {len(change_rows)} → {changes_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
