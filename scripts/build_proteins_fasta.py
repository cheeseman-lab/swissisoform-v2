"""Build a deduped FASTA of all 5-gene-test proteins (canonical + isoforms).

Drives upstream + assembly far enough to get every Gene + TIS protein,
then writes ``data/cache/proteins.fa`` with hash-keyed headers. This is
the canonical input for both ``run_plm_embed.sbatch`` and
``run_fold.sbatch`` GPU precompute jobs.

Usage::

    python scripts/build_proteins_fasta.py

Reads the same paths as ``inspect_e2e.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from swissisoform.assembly import assemble_genes
from swissisoform.config import PipelineConfig
from swissisoform.pipeline import UpstreamReference, run_sample
from swissisoform.plm.embed import protein_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "reference"
GTF = DATA / "gencode.v49.primary_assembly.annotation.gtf"
GENOME = DATA / "Gencode_v49_GRCh38.primary_assembly.genome.fa"
PROTEIN = DATA / "gencode.v49.pc_translations.fa"
HELA = DATA / "HeLa_TIS_predict_all.txt"
HELA_RNASEQ = [
    DATA / "rnaseq_counts/GATCAG_8_htseqcount.txt",
    DATA / "rnaseq_counts/CACGGT_9_htseqcount.txt",
]

TEST_GENES = ["TP53", "EIF4G1", "VEGFA", "CTNND1", "MYC"]


def main() -> None:
    cfg = PipelineConfig()  # noqa: F841 — kept for parity with inspect_e2e
    print("Loading reference (GTF + genome + protein products)…")
    reference = UpstreamReference.load(gtf_path=GTF, genome_fasta=GENOME, protein_fasta=PROTEIN)

    print("Running upstream filter+impute on HeLa (full sample)…")
    final, _ = run_sample(HELA, HELA_RNASEQ, GTF, sample="HeLa", reference=reference)
    print(f"Filter output: {len(final)} rows")

    print(f"Assembling genes (5 test genes only): {TEST_GENES}")
    genes = assemble_genes(
        final[final["Symbol"].isin(TEST_GENES)],
        gene_names=TEST_GENES,
        genome_fasta=GENOME,
        exon_skeletons=reference.exon_skeletons,
    )
    print(f"Assembled {len(genes)} genes, "
          f"{sum(len(g.tis_sites) for g in genes)} total TIS sites.")

    proteins: list[str] = []
    for g in genes:
        if g.canonical_protein:
            proteins.append(g.canonical_protein)
        for s in g.tis_sites:
            if s.isoform_protein:
                proteins.append(s.isoform_protein)
            if s.canonical_protein:
                proteins.append(s.canonical_protein)

    fasta_out = ROOT / "data" / "cache" / "proteins.fa"
    fasta_out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n = 0
    with open(fasta_out, "w") as fh:
        for seq in proteins:
            stripped = seq.rstrip("*").upper()
            if not stripped:
                continue
            h = protein_hash(stripped)
            if h in seen:
                continue
            seen.add(h)
            fh.write(f">{h}\n{stripped}\n")
            n += 1
    print(f"Wrote {n} unique proteins to {fasta_out}")


if __name__ == "__main__":
    main()
