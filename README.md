# SwissIsoform v2

Modular pipeline for annotating alternative protein isoforms from translation
initiation sequencing (TI-seq / Ribo-TISH). Consolidates code from three
earlier repos (`swissisoform` v1, `tiap`, `coTISja`) into a single annotation
architecture with rich domain objects and **symmetric canonical / isoform
annotation** — every per-protein module runs on both the canonical protein and
each alternative isoform, and a comparator diffs the two.

---

## Entry point: `scripts/run.py`

**There is one entry point.** `run.py` is the unified end-to-end driver — it
loads references, produces the filtered TIS table, assembles domain objects,
runs every annotation module, compares canonical vs. isoform, scores, and
writes the paired output. Everything else in `scripts/` either wraps it, feeds
it (GPU precompute), or sets up its reference data.

```bash
# 5-gene diagnostic (HeLa only, single-sample path)
python scripts/run.py --preset 5gene

# A subset of genes across all 6 cell lines
python scripts/run.py --genes TP53 EIF4G1 MYC --run-name my_run

# Genes from a file (one HGNC symbol per line, '#' comments ok)
python scripts/run.py --gene-list genes.txt

# A curated set of specific isoforms — a named preset (presets/cheeseman12.toml)
python scripts/run.py --preset cheeseman12

# …or an ad-hoc isoform file (parquet/CSV with Tid, GenomePos, StartCodon)
python scripts/run.py --isoforms picks.csv --run-name my_isoforms

# The full catalog (every gene, all cell lines)
python scripts/run.py --all --run-name full_catalog

# Skip GPU-dependent modules when their caches are empty
python scripts/run.py --genes TP53 --no-gpu          # = --skip-modules plm_vep,structure
python scripts/run.py --genes TP53 --skip-modules clinical,conservation
```

| Mode | Flag | Selects |
|------|------|---------|
| Preset | `--preset <name>` | A named run from `presets/<name>.toml` — currently `5gene` (HeLa diagnostic) and `cheeseman12` (12 reviewer-picked isoforms) |
| Genes | `--genes SYM …` | Named HGNC symbols |
| Gene list | `--gene-list FILE` | One symbol per line |
| Isoforms | `--isoforms FILE` | Ad-hoc TIS picks (parquet/CSV with `Tid`,`GenomePos`,`StartCodon`) |
| Catalog | `--all` | Every gene |

**Presets** are self-contained TOML files in `presets/` (auto-discovered — drop
a new `.toml` to add a named run). Each lists either `genes = [...]` or an inline
`[[isoforms]]` array of `{gene, tid, genome_pos, start_codon}` picks, plus
`run_name` / `min_cell_lines` / optional `cell_lines`. No external pick files —
the isoforms live in the preset.

Other flags: `--cell-lines HeLa,K562,…` (default all 6), `--single-sample`,
`--min-cell-lines N`, `--rebuild-combined` (force-rebuild the cached filtered
DB), `--no-spot-check-limit`. Run `python scripts/run.py --help` for the full
list.

Output lands in `data/output/<run_name>/`: `all_paired.parquet` (one row per
TIS, with `canonical_<module>_<field>` + `isoform_<module>_<field>` columns)
plus one `<gene>_paired.parquet` per gene.

> To run **everything as one Slurm job — including GPU precompute** — use the
> orchestrator `sbatch scripts/slurm/run.sbatch --preset 5gene`: it emits the
> FASTA, spawns the ESM-2 + folding jobs, waits, then runs the full pipeline.
> All args after the script name are forwarded to `run.py`.

---

## How a run works

`run.py main()` is six stages:

```
Stage 1  Load references            UpstreamReference.load(GTF, genome, protein FASTA)
                                     → exon skeletons, CDS, canonical products
Stage 2  Produce the filtered TIS table         ── MODE-DEPENDENT, see below ──
Stage 3  Subset to requested genes / isoforms    (post-load .isin() / restrict_to_isoforms)
Stage 4  Assemble Gene + TIS objects             assemble_genes(...) — canonical pick,
                                                 ORF typing, diff region, Kozak, ORF exons
Stage 5  Build proteins.fa + precompute          write_proteins_fasta(subset) →
                                                 data/cache/proteins.fa, then per-module precompute
Stage 6  Annotate → compare → score              AnnotationPipeline.run → compare_genes →
                                                 EvidenceScoringModule → paired parquet
```

**Two things people get wrong about this pipeline:**

**1. The filtered DB is built/loaded in full, *then* subset — and the path
differs by mode.** There is no "filter upstream to just these genes" pushdown.

| Invocation | Stage 2 behaviour | Full 6-line DB touched? |
|------------|-------------------|--------------------------|
| `--preset 5gene` (single-sample) | `run_sample(HeLa)` in memory, then subset | **No** — HeLa only |
| `--genes` / `--gene-list` / `--isoforms` / `--all` (multi-sample) | `load_combined()` reads the cached `data/output/filtered/all_samples_combined.parquet` (all 6 cell lines, all genes); rebuilds it via `run_sample` per cell line if missing or `--rebuild-combined`; **then** subsets | **Yes** — load-all-then-filter |

So `--preset 5gene` and `--genes …` take *different* data paths. The combined
parquet is the cached full filtered DB; rebuilding it re-runs upstream on every
requested cell line (all genes) before any subsetting.

**2. `proteins.fa` is generated mid-run, for the subset only — and PLM/Structure
are cache-lookup.** Stage 5 writes `data/cache/proteins.fa` containing the
canonical + isoform proteins of *the selected genes*, regenerated every run.
PLM-VEP and Structure precompute are **lookup-only** (`inline=False`): they read
`data/cache/{plm_esm2,structure}/`, which must already be populated by the GPU
jobs (see [GPU precompute](#gpu-precompute)). If those caches are empty the
modules return empty cleanly — use `--no-gpu` to skip them explicitly.

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

# 3. Build reference databases (one-shot, idempotent, provenance sidecars)
python scripts/setup/setup_databases.py diamond      # ~5 min  (dormant homology DB)
python scripts/setup/setup_databases.py clinvar      # ~2 min
python scripts/setup/setup_databases.py cosmic       # ~30 min (requires .env creds)
python scripts/setup/setup_databases.py deeploc      # ~10 min (conda env + weights)
python scripts/setup/setup_databases.py gnomad       # ~4-6 h  (heavy — ~90 M variants)
# Conservation tracks (Zoonomia 241-mammal): see scripts/setup/download_zoonomia_*.sh

# 4. Run the 5-gene E2E smoke test (HeLa, writes paired parquet)
python scripts/run.py --preset 5gene

# 5. Run tests (unit + integration)
pytest
```

---

## Annotation modules

Modules implement one of three protocols (`src/swissisoform/modules/base.py`):
**ProteinModule** (`annotate(protein) → dict`, run on canonical *and* each
isoform), **SiteModule** (`annotate_site(tis) → dict`, needs TIS metadata), or
gene-level. Output is **scalar** (whole-protein aggregate) or **positional**
(`{"hits": [{name, pos, end, …}], "summary": {…}}`); the comparator filters
positional hits to the differential-region coordinates.

| Module | Protocol | Source / method | Mode |
|--------|----------|-----------------|------|
| `biophysics` | Protein | pI, GRAVY, hydrophobicity, disorder/LLPS heuristics | inline (pure Python) |
| `motifs` | Protein | regex motif scan → positional hits | inline |
| `localization` | Protein | DeepLoc 2 prediction + signals + membrane | lookup (`precompute_deeploc`) |
| `clinical` | Protein | gnomAD + ClinVar + COSMIC, codon-validated by `ConsequenceValidator` | inline fetch / cache |
| `signalp` | Protein | SignalP 6 signal-peptide prediction | lookup (`precompute_signalp`) |
| `targetp` | Protein | TargetP 2 N-terminal targeting peptide | lookup (`precompute_targetp`) |
| `interproscan` | Protein | InterProScan 6 domains/families (Nextflow + Singularity) | lookup (`precompute_interproscan`) |
| `massspec` | Protein | tryptic digest + optional PepQuery2 peptide validation | inline + optional lookup |
| `conservation` | Site | Zoonomia 241-mammal **PhyloP + PhastCons BigWig** (TIS codon, Kozak window, unique/shared region means) | BigWig lookup |
| `conservation_frame` | Site | primate + mammalian reading-frame integrity over the Cactus **HAL** (`hal2maf`) | HAL query |
| `variant_intersection` | Site | tags each clinical hit by genomic membership in isoform-unique vs. shared ORF region | inline (uses ORF exons) |
| `plm_vep` | Site | ESM-2 650M masked-marginal LLR per residue, unique-vs-shared enrichment | lookup (GPU cache) |
| `structure` | Site | Boltz-2 / Chai-1 structure metrics (pLDDT, etc.) | lookup (GPU cache) |
| `core_identity` | Site | ORF-type + TIS metadata | inline |
| `initiation_context` | Site | Kozak context, Hamming distance to `ACCATGG` | inline |
| `generef` | Gene-level | external gene reference context | lookup |
| `scoring` | (aggregator) | dual-axis **E1–E7 existence** + **F1–F6 functional** evidence score | inline over all sites |

> The old homology-based conservation (DIAMOND/blastp vs. SwissProt) is retained
> dormant as `conservation_homology.py`, not wired into the pipeline.

**Contract** — every module: defines `MODULE_NAME` / `OUTPUT_COLUMNS` / `SCOPE`;
keeps a backward-compatible `run(tis_sites)` wrapper; never drops sites
(`len(output) == len(input)`); uses `None` for values it genuinely cannot
compute (never silent defaults).

---

## Reference databases (built by `scripts/setup/setup_databases.py`)

Every artifact sits under `data/reference/<db>/` next to a `_setup.json`
provenance sidecar recording `source_url`, `version`, `fetched_at`, and
per-build stats. All of `data/reference/` is gitignored.

| Target | Artifact | Source | Notes |
|--------|----------|--------|-------|
| `gencode` | `data/reference/gencode.v49.*` | GENCODE FTP | Delegates to `scripts/setup/download_references.sh`. Primary-assembly FASTA + annotation GTF + pc_translations FASTA. |
| `clinvar` | `data/reference/clinvar/variant_summary.parquet` | NCBI `variant_summary.txt.gz` | Reader uses the VCF-format `ReferenceAlleleVCF`/`AlternateAlleleVCF`/`PositionVCF` columns (genomic +strand), so minus-strand genes validate correctly. |
| `cosmic` | `data/reference/cosmic/cosmic_variants.parquet/` (directory) | Sanger COSMIC v102 GRCh38 | Authenticated download; creds via `.env` (`COSMIC_EMAIL`,`COSMIC_PASSWORD`). NonCoding VCF (>100 M rows) streams to per-VCF parquet. |
| `gnomad` | `data/reference/gnomad/gnomad_v4.1_exome.parquet` | gnomAD public bucket | Per-chrom VCF → VEP-filter PASS + canonical SYMBOL → per-chrom parquet → streamed concat (memory-safe `ParquetWriter`). |
| `zoonomia` | `data/reference/zoonomia/cactus241way.phyloP.bw`, `hg38.phastCons100way.bw`, `241-mammalian-2020v2.hal` | UCSC (Christmas et al. 2023) | BigWigs ~13 GB (`download_zoonomia_bigwigs.sh`); HAL 200-600 GB, on demand (`download_zoonomia_hal.sh`). |
| `deeploc` | conda env `swissisoform-v2-deeploc` + `./.torch_cache/` | DTU DeepLoc 2.1 | Python-3.8-pinned isolated env; ESM-1b weights cached in-repo (HOME quota is tight). |
| `diamond` | `data/reference/diamond/swissprot.dmnd` | UniProt reviewed FASTA | Dormant — only used by the retired homology conservation module. |

### COSMIC prerequisites

1. Register at <https://cancer.sanger.ac.uk/cosmic> (free academic license).
2. Copy `.env.example` → `.env` (gitignored) and fill `COSMIC_EMAIL` / `COSMIC_PASSWORD`.
3. `python scripts/setup/setup_databases.py cosmic` auto-loads the `.env`.

---

## GPU precompute

PLM-VEP and Structure are **cache-lookup** modules — their caches must be
populated on a GPU node *before* annotation. Two ways:

**One command (recommended)** — the orchestrator does it all as a single Slurm
job (emit FASTA → spawn GPU jobs → wait → full run):

```bash
mkdir -p logs
sbatch scripts/slurm/run.sbatch --preset 5gene
```

**Manual / step-by-step:**

```bash
# 1. Emit data/cache/proteins.fa for the genes you'll run
python scripts/run.py --preset 5gene --emit-fasta

# 2. Populate caches on a GPU node (A6000)
sbatch scripts/slurm/run_plm_embed.sbatch data/cache/proteins.fa   # ESM-2 650M → data/cache/plm_esm2/
sbatch scripts/slurm/run_fold.sbatch data/cache/proteins.fa boltz  # → data/cache/structure/

# 3. Run — plm_vep + structure now find their caches
python scripts/run.py --preset 5gene
```

---

## Key design choices

**Per-Tid canonical protein.** Each TIS gets `canonical_protein` from its own
transcript's Annotated row, not the gene-level longest — Ribo-TISH classifies
ORF type relative to each transcript's CDS. Gene-level longest is kept on
`Gene.canonical_protein` as a representative for gene-level annotations.

**Symmetric canonical / isoform annotation.** Per-protein modules run on both
the canonical (cached in `Gene.canonical_annotations`) and each isoform (in
`site.isoform_annotations`). The comparator diffs the two dicts; it never stores
precomputed deltas in module output.

**Positional hits over scalars when possible.** Everything that *can* be stored
per-coordinate (motif matches, variants, conservation, peptides) *should* be.
Scalars are reserved for inherently whole-protein properties. The comparator
filters positional hits to the differential-region coordinates at compare time.

**Authoritative `ConsequenceValidator`.** HGVSp from gnomAD/ClinVar is in
canonical-frame coordinates — wrong for alternative TIS. The validator always
runs with real CDS features (genomic→coding map, strand-aware); HGVSp is never
trusted for protein position.

**Load-all-then-subset (multi-sample).** Filtering is decoupled from differential
analysis: upstream runs per cell line over the whole sample, the combined parquet
is the cached union, and gene/isoform selection is a downstream filter.

---

## Repository layout

```
src/swissisoform/
  models.py            # ORFType, Gene, TIS, DifferentialRegion, CellLineExpression, TranscriptCoordinates
  config.py            # PipelineConfig (+ per-module configs)
  pipeline.py          # run_sample (upstream filter+impute) + AnnotationPipeline (wiring)
  filtering.py         # 5-step TIS filter; combine.py — cross-cell-line union
  assembly.py          # DataFrame → Gene + TIS objects; coords.py — ORF→genomic exon mapping
  clinical/            # VariantFetcher + ConsequenceValidator (codon-level)
  compare/             # paired canonical-vs-isoform comparison
  conservation_frame/  # MAF/HAL reading-frame integrity subpackage
  plm/ structure/      # ESM-2 embedding + Boltz/Chai folding caches
  io/                  # ribotish, gtf, parquet, rnaseq, canonical loaders
  modules/             # the annotation modules + base protocols

scripts/
  run.py                    # THE entry point (run.py --emit-fasta seeds the GPU caches)
  slurm/                    # run.sbatch (unified orchestrator) + run_small_e2e / run_plm_embed / run_fold
  analysis/                 # export_alt_regions_bed.py — BED12 of differential ORF regions (post-hoc)
  setup/                    # download_references.sh, download_zoonomia_*, setup_databases.py, setup_interproscan.sbatch
  bin/                      # hal2maf, halStats Singularity shims

presets/                     # named runs (auto-discovered *.toml): 5gene, cheeseman12, …
tests/                       # pytest — unit + real-genome 5-gene integration
data/reference/              # gitignored — primary-source reference DBs
data/output/                 # gitignored except the checked-in 5gene_e2e/all_paired.parquet fixture
data/cache/                  # gitignored — proteins.fa + per-module precompute caches
docs/                        # methods (typst) + code-review specs
```

---

## Development

```bash
ruff check src scripts tests       # lint (line length 100, Google docstrings)
ruff format src scripts tests      # format

pytest                             # all tests (unit + integration)
pytest --cov=swissisoform --cov-report=term-missing
pytest tests/test_biophysics.py -v # single module
```

GPU- and network-marked tests are excluded by default
(`-m 'not gpu and not network'`).

---

## Roadmap

- [x] Foundation: domain model, module protocols, I/O, filtering, assembly
- [x] Upstream port (faithful port of coTISja `filter_ribotish.py`: filter + impute)
- [x] All annotation modules wired into the unified `run.py` pipeline
- [x] Reference-DB builds (clinvar, cosmic, gnomad, deeploc, zoonomia)
- [x] ORF→genomic exon mapping; conservation region means; clinical variant intersection
- [x] Comparator (scalar deltas + positional subset) and dual-axis evidence scoring
- [x] Unified `run.py` CLI + `scripts/setup/` reorg
- [ ] Full end-to-end test through scoring + paired parquet (currently partial)
- [ ] Full catalog run across all 6 cell lines

---

## References

- GENCODE v49: <https://www.gencodegenes.org/human/release_49.html>
- gnomAD v4.1: <https://gnomad.broadinstitute.org/downloads>
- ClinVar: <https://www.ncbi.nlm.nih.gov/clinvar/> · COSMIC v102: <https://cancer.sanger.ac.uk/cosmic>
- Zoonomia 241-mammal alignment: Christmas et al., _Science_ 380:eabn3943 (2023)
- Ribo-TISH: Zhang et al., _Nat Commun_ 2017 · DeepLoc 2.1: Thumuluri et al., _NAR_ 2022
- ESM-2: Lin et al., _Science_ 2023 · Boltz-2 / Chai-1 structure prediction
