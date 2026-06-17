"""Add ESM-C per-isoform variant-effect ΔLLR aggregates as columns on plm_vep_delta_llr.csv.

VEP = Variant Effect Prediction: scoring how damaging an amino-acid substitution is.
Here each clinical variant gets  ΔLLR = log P(alt) − log P(wt)  from the cached ESM-C
``aa_logprobs``, then we aggregate per isoform × region (unique/shared) × source
(gnomAD=germline / ClinVar+COSMIC=disease) — the website's "Mean/Min ΔLLR" panel.

This re-assembles a preset's genes and runs ONLY clinical → variant_intersection →
varianteffect (via runner.build_pipeline with a skip set) against ``data/cache/plm_esmc/``,
then MERGES the per-isoform aggregates into the existing per-isoform CSV
``data/output/<run_name>/plm_vep_delta_llr.csv`` as new ``vep_*`` columns, each paired
with its ESM-2 twin (``vep_*_esm2``) read read-only from ``all_paired.parquet``.

Also saves the full per-variant ESM-C table to NEW files
``clinical_variants_esmc.parquet`` (+ ``.csv`` mirror) — one row per clinical variant
with its individual ΔLLR.

Reads everything else read-only — does NOT modify ESM-2 caches or all_paired.parquet.
Writes only NEW files (clinical_variants_esmc.*) plus the analysis CSV
(plm_vep_delta_llr.csv, with the added vep_* columns).
Idempotent: existing ``vep_*`` columns are dropped before re-merging.

Usage:
    python scripts/export/export_variant_delta_llr.py --preset cheeseman13
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

from swissisoform import runner  # noqa: E402, I001
from swissisoform.assembly import assemble_genes  # noqa: E402
from swissisoform.pipeline import UpstreamReference  # noqa: E402
from swissisoform.references import (  # noqa: E402
    ALL_CELL_LINES,
    GENOME,
    GTF,
    PRESETS,
    PROTEIN,
    build_config,
)

ALL_MODULES = {
    "biophysics", "motifs", "massspec", "clinical", "localization", "signalp",
    "targetp", "interproscan", "core_identity", "initiation_context", "conservation",
    "conservation_frame", "variant_intersection", "plm_vep", "varianteffect",
    "structure", "generef",
}
KEEP = {"clinical", "variant_intersection", "varianteffect"}

# ΔLLR aggregate metrics (ESM-model-dependent). AlphaMissense aggregates are
# model-independent, so excluded from the ESM-C-vs-ESM-2 comparison.
AGG_METRICS = ["n_scored_plm"] + [
    f"{stat}_delta_llr_{region}{src}"
    for region in ("unique", "shared")
    for src in ("", "_gnomad", "_disease")
    for stat in ("mean", "min")
]

# Per-variant fields saved to the ESM-C clinical-variant parquet (defensive .get()).
HIT_FIELDS = [
    "variant_id", "source", "chrom", "genomic_pos", "ref", "alt",
    "consequence", "allele_frequency",
    "protein_pos", "aa_ref", "aa_alt",
    "isoform_protein_pos", "isoform_aa_ref", "isoform_aa_alt",
    "in_isoform_unique", "in_isoform_shared",
    "plm_frame", "plm_status", "plm_llr_wt", "plm_llr_alt", "plm_delta_llr",
    "am_pathogenicity", "am_class",
    "effect_lof", "effect_damaging", "effect_tolerated_in_gnomad",
]


def _region(hit: dict) -> str:
    if hit.get("in_isoform_unique") is True:
        return "unique"
    if hit.get("in_isoform_shared") is True:
        return "shared"
    return "other"


def assemble_for_preset(preset_name: str):
    """Reproduce the runner's load + assemble; return (spec, genes, ref)."""
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
    return spec, genes, ref


def main(argv: list[str] | None = None) -> int:
    """Run clinical→intersection→varianteffect on ESM-C; merge aggregates into the CSV."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--preset", default="cheeseman13", choices=sorted(PRESETS))
    p.add_argument("--csv", type=Path, default=None,
                   help="Target CSV (default data/output/<run_name>/plm_vep_delta_llr.csv).")
    p.add_argument("--baseline-parquet", type=Path, default=None,
                   help="ESM-2 all_paired.parquet for the comparison twins (read-only).")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    spec, genes, ref = assemble_for_preset(args.preset)
    run_name = spec.get("run_name", args.preset)
    csv_path = args.csv or (ROOT / "data" / "output" / run_name / "plm_vep_delta_llr.csv")
    base_pq = args.baseline_parquet or (ROOT / "data" / "output" / run_name / "all_paired.parquet")
    if not csv_path.is_file():
        raise SystemExit(f"target CSV not found: {csv_path} (run export_plm_delta_llr.py first)")

    # Run ONLY clinical → variant_intersection → varianteffect (real dispatch,
    # clinical pre-fetch, ConsequenceValidator). VariantEffectModule defaults to
    # the plm_esmc cache. preds={} is safe — every preds-consuming module is skipped.
    cfg = build_config()
    pipeline = runner.build_pipeline(cfg, {}, ref, genes, ALL_MODULES - KEEP)
    pipeline.run(genes)

    # Collect per-isoform aggregates (for the CSV merge) AND per-variant rows
    # (for the ESM-C clinical-variant parquet).
    n_var = n_scored = n_am = 0
    rows: list[dict] = []
    var_rows: list[dict] = []
    for gene in genes:
        for site in gene.tis_sites:
            if not site.isoform_protein:
                continue
            ve = site.isoform_annotations.get("varianteffect") or {}
            hits = ve.get("hits") or []
            n_var += len(hits)
            n_scored += sum(1 for h in hits if h.get("plm_delta_llr") is not None)
            n_am += sum(1 for h in hits if h.get("am_pathogenicity") is not None)
            ot = site.orf_type.value if site.orf_type else ""
            for hit in hits:
                vr = {"gene": gene.gene_name, "tis_id": site.tis_id,
                      "orf_type": ot, "region": _region(hit)}
                vr.update({f: hit.get(f) for f in HIT_FIELDS})
                var_rows.append(vr)
            row = {"tis_id": site.tis_id}
            row.update({f"vep_{m}": ve.get(m) for m in AGG_METRICS})
            rows.append(row)
    esmc = pd.DataFrame(rows)

    # Save the ESM-C per-variant clinical-variant table (NEW parquet + csv mirror).
    outdir = csv_path.parent
    var_df = pd.DataFrame(var_rows)
    var_pq = outdir / "clinical_variants_esmc.parquet"
    var_df.to_parquet(var_pq, index=False)
    var_df.to_csv(outdir / "clinical_variants_esmc.csv", index=False)
    print(f"wrote ESM-C clinical-variant table: {len(var_df)} rows → {var_pq}")

    # ESM-2 twins (read-only) from the existing parquet.
    if base_pq.is_file():
        import pyarrow.parquet as pq
        cols = pq.read_table(base_pq).column_names
        present = {m: f"isoform_varianteffect_{m}" for m in AGG_METRICS
                   if f"isoform_varianteffect_{m}" in cols}
        if present:
            base = pq.read_table(base_pq).select(["tis_id"] + list(present.values())).to_pandas()
            base = base.rename(columns={present[m]: f"vep_{m}_esm2" for m in present})
            esmc = esmc.merge(base, on="tis_id", how="left")
            print(f"appended {len(present)} ESM-2 twins from {base_pq.name}")
        else:
            print(f"⚠ {base_pq.name} has no isoform_varianteffect_* columns — ESM-C only")
    else:
        print(f"⚠ baseline {base_pq} not found — ESM-C only")

    # Merge into the CSV: drop any prior vep_* cols (idempotent), append, interleave twins.
    csv = pd.read_csv(csv_path)
    csv = csv[[c for c in csv.columns if not c.startswith("vep_")]]
    merged = csv.merge(esmc, on="tis_id", how="left")
    base_cols = [c for c in merged.columns if not c.startswith("vep_")]
    vep_base = [f"vep_{m}" for m in AGG_METRICS if f"vep_{m}" in merged.columns]
    vep_ordered: list[str] = []
    for c in vep_base:
        vep_ordered.append(c)
        if f"{c}_esm2" in merged.columns:
            vep_ordered.append(f"{c}_esm2")
    merged = merged[base_cols + vep_ordered]
    merged.to_csv(csv_path, index=False)

    print(f"\nvariants: {n_var} | ESM-C ΔLLR scored: {n_scored} | AlphaMissense: {n_am}")
    print(f"added {len(vep_ordered)} vep_* columns → {csv_path} "
          f"({len(merged)} rows × {len(merged.columns)} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
