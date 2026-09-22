"""Load a preset and assemble its genes, the way the runner does.

The export scripts each need a `Gene` list for a named preset before they can
render anything, and reproducing the runner's load path by hand is the part that
drifts: two scripts carried this verbatim, so a change to how a preset resolves
its cell lines or its isoform picks had to be made in both or the two exports
would describe different isoforms under the same preset name.
"""

from __future__ import annotations

from typing import Any

from swissisoform import runner
from swissisoform.assembly import assemble_genes
from swissisoform.pipeline import UpstreamReference
from swissisoform.references import (
    ALL_CELL_LINES,
    GENOME,
    GTF,
    PRESETS,
    PROTEIN,
    build_config,
)


def assemble_for_preset(preset_name: str) -> tuple[dict[str, Any], list]:
    """Reproduce the runner's load + assemble for *preset_name*.

    Returns:
        ``(spec, genes)`` — the preset's TOML dict and the assembled genes.
    """
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


__all__ = ["assemble_for_preset"]
