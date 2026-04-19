"""Run upstream (filter + canonical imputation) for every cell line in the manifest.

Produces one ``{sample}_TIS_filtered.csv`` per cell line under
``data/output/filtered/``. This is the SwissIsoform v2 equivalent of
the upstream reference's per-sample filter script.
"""

from __future__ import annotations

from pathlib import Path

from swissisoform.config import PipelineConfig
from swissisoform.io.rnaseq import load_sample_manifest
from swissisoform.pipeline import UpstreamReference, run_sample

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "reference"
OUT = ROOT / "data" / "output" / "filtered"

GTF = DATA / "gencode.v49.primary_assembly.annotation.gtf"
GENOME = DATA / "Gencode_v49_GRCh38.primary_assembly.genome.fa"
PROTEIN = DATA / "gencode.v49.pc_translations.fa"
SAMPLE_MANIFEST = DATA / "ribotish_sample_manifest.csv"
REPLICATE_MANIFEST = DATA / "ribotish_replicate_manifest.csv"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading shared reference tables (GTF + FASTAs)…")
    reference = UpstreamReference.load(
        gtf_path=GTF, genome_fasta=GENOME, protein_fasta=PROTEIN
    )

    manifest = load_sample_manifest(SAMPLE_MANIFEST, REPLICATE_MANIFEST)
    cfg = PipelineConfig()

    for _, row in manifest.iterrows():
        sample = row["sample"]
        predict = ROOT / row["predict_file"]
        filtered_out = ROOT / row["filtered_file"]
        dropped_out = ROOT / row["dropped_file"]
        rnaseq_files = [ROOT / f for f in row["rnaseq_count_files"]]

        print(f"\n=== {sample} ===")
        final_df, dropped_df = run_sample(
            predict,
            rnaseq_files,
            GTF,
            sample=sample,
            config=cfg,
            reference=reference,
        )
        filtered_out.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(filtered_out, index=False)
        dropped_out.parent.mkdir(parents=True, exist_ok=True)
        dropped_df.to_csv(dropped_out, index=False)
        print(
            f"  wrote {len(final_df):,} filtered rows → {filtered_out}\n"
            f"  wrote {len(dropped_df):,} dropped rows → {dropped_out}"
        )


if __name__ == "__main__":
    main()
