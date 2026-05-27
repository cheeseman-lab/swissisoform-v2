# SwissIsoform v2

Modular pipeline for annotating alternative protein isoforms from translation
initiation sequencing (TI-seq / Ribo-TISH). Consolidates code from three
earlier repos (`swissisoform` v1, `tiap`, `coTISja`) into a single 9-module
architecture with rich domain objects and symmetric canonical / isoform
annotation.

---

## Quick start

```bash
# 1. Create conda env + install in editable mode
eval "$(conda shell.bash hook)"
conda create -n swissisoform-v2 -c conda-forge python=3.11 uv pip -y
conda activate swissisoform-v2
uv pip install -e ".[dev]"

# 2. Download reference data (GENCODE FASTA/GTF + Ribo-TISH inputs)
bash scripts/setup/download_references.sh

# 3. Build reference databases (one-shot, from primary sources)
#    Each target is idempotent and writes a provenance sidecar
#    (_setup.json) next to its artifact.
python scripts/setup/setup_databases.py diamond      # ~5 min
python scripts/setup/setup_databases.py clinvar      # ~2 min
python scripts/setup/setup_databases.py cosmic       # ~30 min (requires .env creds)
python scripts/setup/setup_databases.py deeploc      # ~10 min (conda env + weights)
python scripts/setup/setup_databases.py gnomad       # ~4-6 h (heavy — ~90 M variants)

# 4. Run the 5-gene E2E smoke test (HeLa, writes paired parquet)
python scripts/run.py --preset 5gene

# 5. Run tests (unit + integration; ~30–60 s end to end)
pytest
```

## What the pipeline does

```
  Raw Ribo-TISH predict_all.txt (6 cell lines × 21 columns each)
           │
           │  io.ribotish → load_ribotish_predictions
           ▼
  pipeline.run_sample  (per cell line)
           │  filter (MANE / TSL 1–3 reference + distance dedup)
           │  RNA-seq RPM normalization per gene
           │  impute canonical starts from GTF + protein FASTA
           │  drop uncanonical transcripts (cds_start_NF / retained_intron)
           ▼
  {sample}_TIS_filtered.csv  +  Annotated canonical rows for all Tids
           │
           │  combine.combine_filtered_samples (cross-cell-line union)
           ▼
  assembly.assemble_genes
           │  one Gene per symbol
           │  per-Tid canonical protein lookup (Annotated row in own transcript)
           │  Kozak context (genome-FASTA 13-nt window, strand-aware)
           │  differential region (extension prefix / truncation prefix / uORF-full)
           │  CellLineExpression nested per TIS
           ▼
  Gene objects with TIS sites  (the domain model)
           │
           │  pipeline.AnnotationPipeline
           │    — ProteinModules on canonical (per gene)  AND  isoform (per TIS)
           │    — SiteModules per TIS (orf_type, kozak)
           │    — GeneModules per gene (generef)
           ▼
  Gene.canonical_annotations  +  TIS.isoform_annotations  +  site metadata
           │
           │  io.parquet.paired_tis_dataframe
           ▼
  One row per TIS with canonical_{module}_{field}  +  isoform_{module}_{field}
  (the "differential-ready" table — comparator does the subtraction downstream)
```

## Annotation modules

| Module | Protocol | Interface | Input | Output |
|--------|----------|-----------|-------|--------|
| `biophysics` | ProteinModule | `annotate(protein) → dict` | protein str | scalar (pI, GRAVY, disorder, LLPS score, …) |
| `motifs` | ProteinModule | `annotate(protein) → dict` | protein str | positional hits (regex matches) |
| `localization` | ProteinModule | `annotate(protein) → dict` | protein str, lookup by sha1 hash **or** by `tis_id` | DeepLoc2 prediction + signals + membrane + WoLF PSORT |
| `clinical` | ProteinModule | `annotate(protein, gene_name)` | protein + gene symbol | positional variants from gnomAD + ClinVar + COSMIC, codon-validated consequences |
| `conservation` | ProteinModule | `annotate(protein) → dict` | protein str | DIAMOND-vs-SwissProt hits per residue, score 0–9 |
| `massspec` | ProteinModule | `annotate(protein, canonical, gene_name)` | protein + canonical protein + gene | tryptic peptides + optional PepQuery hit |
| `core_identity` | SiteModule | `annotate_site(tis)` | TIS object | orf_type metadata |
| `initiation_context` | SiteModule | `annotate_site(tis)` | TIS object | Kozak context + Hamming distances to ACCATGG |
| `generef` | Gene-level | `annotate_by_gene(gene_name)` | gene symbol | external reference data (OMIM, etc.) |

**Contract summary** — see `src/swissisoform/modules/base.py` for protocol
definitions. Every module must:
1. Define `MODULE_NAME`, `OUTPUT_COLUMNS`, `SCOPE` as class attrs.
2. Keep `run(tis_sites)` as a backward-compatible wrapper writing to
   `site.isoform_annotations[MODULE_NAME]`.
3. Never drop sites — `len(output) == len(input)`.
4. Use `None` for values it genuinely cannot compute (never silent defaults).

## Reference databases (built by `scripts/setup/setup_databases.py`)

Every artifact sits under `data/reference/<db>/` next to a `_setup.json`
provenance sidecar that records `source_url`, `version`, `fetched_at`, and
any per-build stats.

| Target | Artifact | Source | Notes |
|--------|----------|--------|-------|
| `gencode` | `data/reference/gencode.v49.*` | GENCODE FTP | Delegates to `scripts/setup/download_references.sh`. Includes primary assembly FASTA + annotation GTF + pc_translations FASTA. |
| `diamond` | `data/reference/diamond/swissprot.dmnd` | UniProt reviewed FASTA | Built via `diamond makedb`. Used by `ConservationModule`. |
| `clinvar` | `data/reference/clinvar/variant_summary.parquet` | NCBI `variant_summary.txt.gz` | Filter pushdown via PyArrow in `ClinicalModule`. **Reader uses `ReferenceAlleleVCF` / `AlternateAlleleVCF` / `PositionVCF` columns** — these are genomic +strand oriented. The title-parse fallback (HGVSc `c.XXX>YYY`) returns transcript-direction bases and would mis-validate minus-strand genes. |
| `cosmic` | `data/reference/cosmic/cosmic_variants.parquet/` (**directory** of per-VCF parquets) | Sanger COSMIC v102 GRCh38 | Requires authenticated download. Credentials via `.env` (`COSMIC_EMAIL`, `COSMIC_PASSWORD`) or CLI flags. NonCoding VCF has >100 M rows so parsing streams to a per-VCF parquet; the reader (`pyarrow.dataset`) handles the dir transparently. |
| `gnomad` | `data/reference/gnomad/gnomad_v4.1_exome.parquet` | gnomAD public Google bucket | Downloads per-chrom VCF, VEP-filters to PASS + canonical-transcript SYMBOL, writes per-chrom parquet, then stream-concats via `pyarrow.ParquetWriter`. Not run by default from `all` — pass `--include-gnomad`. |
| `deeploc` | conda env `swissisoform-v2-deeploc` + `./.torch_cache/` | DTU DeepLoc 2.1 tarball | Python-3.8-pinned; isolated env so the main env stays on Python 3.11. ESM-1b weights cached at `./.torch_cache/` because home-dir quota is tight on the shared cluster. |

### COSMIC prerequisites

1. Register at <https://cancer.sanger.ac.uk/cosmic> (free academic license).
2. Copy `.env.example` → `.env` (gitignored) and fill in:

   ```env
   COSMIC_EMAIL=you@example.com
   COSMIC_PASSWORD=...
   ```

3. `python scripts/setup/setup_databases.py cosmic` will auto-load the `.env`.

### gnomAD — memory-safe build

The earlier `pd.concat([read_parquet(p) for p in chroms])` step loaded all
~120 M variants into RAM and OOM-killed the process. The current implementation
uses `pyarrow.parquet.ParquetWriter.write_table()` to stream per-chrom tables
into the final artifact. Peak memory is bounded by the largest per-chrom
parquet (chr1, ~3 GB). Same pattern is used for COSMIC NonCoding VCF.

## Key design choices

**Per-Tid canonical protein.** Each TIS gets the `canonical_protein` from its
own transcript's Annotated row, not the gene-level longest. Ribo-TISH
classifies ORF type (Extended / Truncated / uORF …) relative to each
transcript's CDS, so comparing against the gene-level longest would
mis-classify the shared region. Gene-level longest is still held on
`Gene.canonical_protein` as a representative for gene-level annotations.

**Symmetric canonical / isoform annotation.** Per-protein modules run on
BOTH the canonical protein (once per gene, cached in
`Gene.canonical_annotations`) and each isoform protein (per TIS, stored in
`site.isoform_annotations`). The comparator diffs these two dicts. Never
stores pre-computed "deltas" in the module output.

**Positional hits over scalars when possible.** Everything that CAN be
stored per-coordinate (motif matches, variants, conservation hits) SHOULD
be. Scalars are reserved for inherently whole-protein properties (pI, GRAVY,
localization prediction). The comparator filters positional hits to the
differential-region coordinates at comparison time.

**Authoritative `ConsequenceValidator`.** The HGVSp string from gnomAD /
ClinVar is in *canonical-frame* coordinates, wrong for alternative TIS (5'
extensions, non-canonical starts). `ConsequenceValidator` always runs with
real CDS features — we never fall back to HGVSp parsing for protein
position. ClinVar ref/alt come from the VCF-format columns (genomic +strand)
so minus-strand genes validate correctly. MNVs (multi-base substitutions
where `len(ref) == len(alt) > 1`) are labelled `mnv` and carry protein_pos
but no codon-level aa change.

**CPU-first.** The production run paths — conservation (DIAMOND),
localization (DeepLoc on CPU, "Fast" model), clinical, biophysics — all
work without a GPU. GPUs are scarce on the shared cluster; scale-up
parallelizes across CPUs.

**No per-module disk cache.** Combined-first dedup at assembly time means
each unique (gene, protein) is annotated exactly once per run, so a disk
cache would get zero reuse and add invalidation cost. Databases (DIAMOND,
gnomAD, ClinVar, COSMIC) ARE local, but the per-call cache is in-memory
only.

## Repository layout

```
src/swissisoform/
  assembly.py          # DataFrame → Gene + TIS domain objects
  combine.py           # Cross-cell-line combine (upstream-side dedup)
  config.py            # PipelineConfig (+ per-module configs)
  filtering.py         # 5-step TIS filter (reference transcripts, dedup, significance)
  merging.py           # Canonical vs. alternative pairing helpers
  models.py            # ORFType, Gene, TIS, DifferentialRegion, CellLineExpression
  pipeline.py          # run_sample (upstream) + AnnotationPipeline (wiring)
  clinical/            # VariantFetcher + ConsequenceValidator (codon-level)
  compare/paired.py    # Paired comparison primitives (WIP — roadmap step 4)
  io/                  # ribotish, gtf, parquet, canonical (reference loaders)
  modules/             # The 9 annotation modules + base protocols

scripts/
  run.py                     # Unified E2E driver (--preset 5gene / --all / --genes / --isoforms)
  inspect_e2e.py             # Thin wrapper → run.py --preset 5gene (muscle memory)
  run_e2e_all.py             # Thin wrapper → run.py --all (muscle memory)
  setup/
    download_references.sh   # GENCODE FASTA/GTF (one-shot wget)
    setup_databases.py       # Idempotent DB orchestrator (diamond/clinvar/cosmic/gnomad/deeploc)
    download_zoonomia_*.sh   # Zoonomia PhyloP/PhastCons BigWigs + 241-mammal HAL

tests/                       # pytest — 368 tests (unit + integration)
data/reference/              # gitignored — primary-source reference DBs
data/output/                 # gitignored — pipeline outputs (paired parquets, logs)
docs/                        # design spec, technical spec, wave-1 plan
```

## Development

```bash
# Lint + format (ruff, 100-col Google style)
ruff check src/
ruff format src/

# All tests (unit + integration)
pytest

# With coverage
pytest --cov=swissisoform --cov-report=term-missing

# Single module
pytest tests/test_biophysics.py -v
```

## Roadmap

- [x] Wave 1 foundation (domain model, module protocols, I/O, filtering, assembly)
- [x] Wave 2 modules (biophysics, motifs, localization, clinical, conservation, massspec, core_identity, initiation_context, generef)
- [x] Upstream port (faithful end-to-end port of `filter_ribotish.py`)
- [x] E2E wiring for all expensive modules (clinical cache, conservation batch, DeepLoc precompute)
- [x] Reference-DB builds (diamond + clinvar; cosmic + gnomad memory-safe refactor)
- [ ] Comparator extension — positional subset to diff region + scalar deltas
- [ ] CLI entry point (`python -m swissisoform run …`)
- [ ] Full end-to-end on all 6 cell lines

## References

- GENCODE v49: <https://www.gencodegenes.org/human/release_49.html>
- gnomAD v4.1: <https://gnomad.broadinstitute.org/downloads>
- ClinVar: <https://www.ncbi.nlm.nih.gov/clinvar/>
- COSMIC v102: <https://cancer.sanger.ac.uk/cosmic>
- Ribo-TISH: Zhang et al., _Nat Commun_ 2017
- DeepLoc 2.1: Thumuluri et al., _NAR_ 2022
- DIAMOND v2.1: Buchfink et al., _Nat Methods_ 2015
