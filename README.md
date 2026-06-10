# SwissIsoform v2

Modular pipeline for annotating alternative protein isoforms from translation
initiation sequencing (TI-seq / Ribo-TISH). Consolidates code from three
earlier repos (`swissisoform` v1, `tiap`, `coTISja`) into a single annotation
architecture with rich domain objects and **symmetric canonical / isoform
annotation** — every per-protein module runs on both the canonical protein and
each alternative isoform, a comparator diffs the two, and a dual-axis evidence
scorer rolls the result into **E1–E6 (existence)** and **F1–F6 (functional
impact)** criteria.

---

## What it does

1. Reads raw Ribo-TISH `predict_all.txt` calls (one file per cell line).
2. Filters + imputes canonical starts per sample (faithful port of coTISja
   `filter_ribotish.py`).
3. Combines the per-sample tables into one deduplicated cross-cell-line table.
4. Assembles `Gene` / `TIS` domain objects — per-Tid canonical pick, ORF
   typing, differential-region coordinates, Kozak context, ORF→genomic exon
   mapping.
5. Annotates canonical and each isoform protein, **compares** the two
   symmetrically, and **scores** the dual evidence axes — 12 evidence buckets
   (E1–E6 existence, F1–F6 functional).
6. Writes a **paired parquet** (`canonical_<module>_<field>` +
   `isoform_<module>_<field>` per row) consumed by the website viewer,
   collaborator exports, and a per-init-site genome-LM skeleton.

### Data flow

```
Ribo-TISH predict_all.txt  (per cell line)
        │  io/ribotish · io/gtf · io/rnaseq
        ▼
  run_sample()             per-sample filter + canonical imputation   (pipeline.py, filtering.py)
        │
        ▼
  combine_filtered_samples()   cross-cell-line dedup union            (combine.py)
        │
        ▼
  assemble_genes()         DataFrame → Gene + TIS objects             (assembly.py, coords.py, contract.py)
        │                  canonical pick · ORF type · diff region · ORF exons
        ▼
  AnnotationPipeline.run() annotators on canonical + each isoform     (pipeline.py, modules/, clinical/, structure/, plm/, conservation_frame/, evidence/)
        │
        ▼
  compare_genes()          symmetric canonical-vs-isoform diff        (compare/)
        │
        ▼
  EvidenceScoringModule    E1–E6 existence · F1–F6 functional         (modules/scoring.py over evidence/)
        │
        ▼
  all_paired.parquet  ──►  website viewer · exports · init-site skeleton
```

---

## Entry points

Two CLIs exist today; they share the `src/swissisoform` core but are separate
front doors.

### `scripts/run.py` — preset / gene-list / isoform driver

The day-to-day driver. Loads references, produces the filtered TIS table,
assembles domain objects, runs every annotation module, compares, scores, and
writes the paired output.

```bash
# 5-gene diagnostic (HeLa only, single-sample path)
python scripts/run.py --preset 5gene

# A subset of genes across all 6 cell lines
python scripts/run.py --genes TP53 EIF4G1 MYC --run-name my_run

# Genes from a file (one HGNC symbol per line, '#' comments ok)
python scripts/run.py --gene-list genes.txt

# A curated set of specific isoforms — a named preset (presets/cheeseman13.toml)
python scripts/run.py --preset cheeseman13

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
| Preset | `--preset <name>` | A named run from `presets/<name>.toml` — `5gene`, `cheeseman12`, `cheeseman13` |
| Genes | `--genes SYM …` | Named HGNC symbols |
| Gene list | `--gene-list FILE` | One symbol per line |
| Isoforms | `--isoforms FILE` | Ad-hoc TIS picks (parquet/CSV with `Tid`,`GenomePos`,`StartCodon`) |
| Catalog | `--all` | Every gene |

**Presets** are self-contained TOML files in `presets/` (auto-discovered — drop
a new `.toml` to add a named run). Each lists either `genes = [...]` or an inline
`[[isoforms]]` array, plus `run_name` / `min_cell_lines` / optional `cell_lines`.

Output lands in `data/output/<run_name>/`: `all_paired.parquet` (one row per
TIS) plus one `<gene>_paired.parquet` per gene. Run
`python scripts/run.py --help` for the full flag list.

> To run **everything as one Slurm job — including GPU precompute** — use the
> orchestrator `sbatch scripts/slurm/run.sbatch --preset 5gene`: it emits the
> FASTA, spawns the ESM-2 + folding jobs, waits, then runs the full pipeline.
> All args after the script name are forwarded to `run.py`.

### `python -m swissisoform run …` — manifest-based CLI

`src/swissisoform/cli.py` drives the same end-to-end flow from explicit
manifests rather than presets — per-sample upstream → combine → assemble →
precompute → annotate → compare → paired parquet, with `--workers` controlling
per-sample and gene-parallel sections.

```bash
python -m swissisoform run \
    --sample-manifest    data/reference/ribotish_sample_manifest.csv \
    --replicate-manifest data/reference/ribotish_replicate_manifest.csv \
    --gtf                data/reference/gencode.v49.primary_assembly.annotation.gtf \
    --genome             data/reference/Gencode_v49_GRCh38.primary_assembly.genome.fa \
    --protein-fasta      data/reference/gencode.v49.pc_translations.fa \
    --output             data/output/full_run \
    --workers 8
```

---

## Repository layout

All pipeline logic lives in `src/swissisoform/`. **Scripts are thin CLI
drivers** — argparse + a call into `src` — a deliberate convention from the
recent refactor (`scripts/run.py` is the one fat exception, as the orchestrating
driver). Everything substantive is importable from the package.

```
src/swissisoform/
  models.py            # domain objects: Gene, TIS, DifferentialRegion,
                       #   CellLineExpression, TranscriptCoordinates
  contract.py          # CANONICAL-VS-ALTERNATIVE CONTRACT: ORFType,
                       #   orf_type_from_ribotish, diff_region_rule
  config.py            # PipelineConfig (+ per-module configs)
  pipeline.py          # run_sample (upstream filter+impute) + AnnotationPipeline
  combine.py           # cross-cell-line dedup union of filtered samples
  assembly.py          # filtered DataFrame → Gene + TIS objects
  filtering.py         # 5-step TIS filter (coTISja port)
  coords.py            # ORF→genomic exon mapping; interval set algebra
  merging.py           # legacy cross-cell-line merge / canonical pairing helpers
  cli.py / __main__.py # manifest-based entry point (python -m swissisoform run)

  evidence/            # the 12 evidence buckets — one folder per criterion, each
                       #   with a high-level score(site, cfg) function:
                       #     E1 e1_primate_conservation   E2 e2_mammalian_conservation
                       #     E3 e3_phylop_selection       E4 e4_reproducibility
                       #     E5 e5_initiation_efficiency  E6 e6_mass_spec
                       #     F1 f1_structure              F2 f2_localization
                       #     F3 f3_domains                F4 f4_targeting
                       #     F5 f5_germline_constraint    F6 f6_disease_enrichment
                       #   Bucket-specific plumbing lives in the folder: E6 (massspec),
                       #   F2 (localization), F3 (interproscan), F4 (signalp + targetp).
                       #   common.py = CriterionResult + helpers; __init__.py exposes
                       #   EXISTENCE_CRITERIA / FUNCTIONAL_CRITERIA.
  modules/             # SHARED single-file annotators + base protocols (base.py):
                       #   biophysics, motifs, conservation, varianteffect,
                       #   variant_intersection, core_identity, initiation_context,
                       #   generef (+ dormant conservation_homology); plus
                       #   scoring.py — the EvidenceScoringModule orchestrator that
                       #   consumes evidence/
  clinical/            # capability folder (module.py + impl): VariantFetcher
                       #   (gnomAD/ClinVar/COSMIC), ConsequenceValidator, AlphaMissense
  structure/           # capability folder (module.py + impl): Boltz-2 / Chai-1
                       #   folding + structure comparison
  plm/                 # capability folder (module.py + impl): ESM-2 masked-marginal
                       #   embedding cache (embed, cli)
  conservation_frame/  # capability folder (module.py + impl): MAF/HAL reading-frame
                       #   integrity (maf, frame, species, hal, tree)
  compare/             # paired canonical-vs-isoform comparison
  io/                  # ribotish, gtf, parquet, rnaseq, canonical loaders
  site/                # website-data build layer: evidence, llm, skeletons
  export/              # collaborator artifacts: bed, xlsx, structures, folding_colors
  setup/               # reference-DB + generef builders (databases, generef)

scripts/                       # thin CLI drivers over src (+ run.py, the one fat driver)
  run.py                       # the preset/gene/isoform driver (also --emit-fasta)
  export/                      # all export CLIs: export_alt_regions_bed,
                               #   export_structures, export_xlsx, build_folding_colors,
                               #   build_init_site_skeleton (genome-LM skeleton parquet)
  site/                        # build_evidence_records, build_transcript_skeletons,
                               #   run_llm_interpretation (+ prompts/)
  setup/                       # download_references.sh, download_zoonomia_*,
                               #   setup_databases.py, fetch_generef.py, *.sbatch
  slurm/                       # run.sbatch (orchestrator) + run_plm_embed / run_fold /
                               #   run_small_e2e

presets/                       # named runs (auto-discovered *.toml): 5gene,
                               #   cheeseman12, cheeseman13
website/                       # standalone Flask viewer (see below)
tests/                         # pytest (~770 tests); regression/ = snapshot harness
data/reference/                # gitignored — primary-source reference DBs
data/output/                   # gitignored except the checked-in 5gene_e2e fixture
data/cache/                    # gitignored — proteins.fa + per-module precompute caches
docs/                          # methods (typst) + code-review specs
```

---

## Evidence buckets

The pipeline is organised around **12 evidence buckets** — six **existence**
criteria (E1–E6: is this isoform a real biological entity?) and six
**functional-impact** criteria (F1–F6: does it change protein function?). Each
bucket is a self-contained subpackage in `src/swissisoform/evidence/` exposing a
high-level `score(site, cfg) → CriterionResult`; `CriterionResult.value` is
`True` (evidence present), `False` (absent), or `None` (cannot evaluate —
upstream data missing). `EvidenceScoringModule` (`modules/scoring.py`) runs them
over every TIS via `EXISTENCE_CRITERIA` / `FUNCTIONAL_CRITERIA`.

Each bucket reads annotations that the **plumbing** — shared annotators in
`modules/` and the capability folders (`clinical/`, `structure/`, `plm/`,
`conservation_frame/`) — has already attached to the site. For four buckets the
plumbing now lives **inside the bucket folder** (E6, F2, F3, F4).

| Bucket | What it scores | Plumbing |
|--------|----------------|----------|
| **E1** primate conservation | primate AA % identity over the unique region | `conservation_frame/` (MAF/HAL frame integrity) |
| **E2** mammalian conservation | mammalian AA % identity over the unique region | `conservation_frame/` |
| **E3** PhyloP selection | absolute unique-region PhyloP above coding-selection threshold | `modules/conservation.py` (Zoonomia 241-mammal BigWig) |
| **E4** reproducibility | TIS detected in ≥ `min_cell_lines` cell lines | `site.expression` |
| **E5** initiation efficiency | max per-cell-line Ribo-TISH initiation efficiency above threshold | `site.expression` |
| **E6** mass spec | PepQuery2-validated unique peptide matches public MS spectra | `evidence/e6_mass_spec/massspec.py` (tryptic digest + PepQuery2) |
| **F1** structure | diff-region pLDDT (structured gain / loss) | `structure/` (Boltz-2 / Chai-1) + `modules/biophysics.py` |
| **F2** localization | DeepLoc compartment / signals / membrane change | `evidence/f2_localization/localization.py` (DeepLoc 2) |
| **F3** domains | real InterPro domain gained / lost in the diff region | `evidence/f3_domains/interproscan.py` (InterProScan 6) |
| **F4** targeting | SignalP / TargetP category change canonical vs. isoform | `evidence/f4_targeting/signalp.py` + `targetp.py` |
| **F5** germline constraint | ESM-2 constraint enrichment OR gnomAD depletion over the unique region | `modules/varianteffect.py` + `clinical/` (via `plm_vep` / `variant_intersection`) |
| **F6** disease enrichment | ClinVar + COSMIC density enrichment in the unique region | `modules/varianteffect.py` + `clinical/` (via `variant_intersection`) |

### Underlying annotators

Buckets never compute raw biology themselves — they read the annotations these
annotators attach. Each implements one of three protocols
(`src/swissisoform/modules/base.py`): **ProteinModule** (`annotate(protein) →
dict`, run on canonical *and* each isoform), **SiteModule** (`annotate_site(tis)
→ dict`, needs TIS metadata), or gene-level. Output is **scalar** (whole-protein
aggregate) or **positional** (`{"hits": [{name, pos, end, …}], "summary": {…}}`);
the comparator filters positional hits to the differential-region coordinates.

Shared single-file annotators live in `modules/`: `biophysics`, `motifs`,
`conservation`, `varianteffect`, `variant_intersection`, `core_identity`,
`initiation_context`, `generef`. Heavier annotators are capability folders that
own a `module.py` plus their implementation: `clinical/`
(gnomAD/ClinVar/COSMIC + `ConsequenceValidator` + AlphaMissense), `structure/`
(Boltz-2 / Chai-1, GPU cache), `plm/` (ESM-2 650M masked-marginal LLR, GPU
cache), `conservation_frame/` (primate/mammalian frame integrity over the
Cactus HAL). The bucket-owned annotators (`massspec`, `localization`,
`interproscan`, `signalp`, `targetp`) live under their `evidence/` folder.

> The old homology-based conservation (DIAMOND/blastp vs. SwissProt) is retained
> dormant as `modules/conservation_homology.py`, not wired into the pipeline.

**Contract** — every annotator: defines `MODULE_NAME` / `OUTPUT_COLUMNS` /
`SCOPE`; keeps a backward-compatible `run(tis_sites)` wrapper; never drops sites
(`len(output) == len(input)`); uses `None` for values it genuinely cannot
compute (never silent defaults).

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
python scripts/setup/setup_databases.py clinvar      # ~2 min
python scripts/setup/setup_databases.py cosmic       # ~30 min (requires .env creds)
python scripts/setup/setup_databases.py deeploc      # ~10 min (conda env + weights)
python scripts/setup/setup_databases.py gnomad       # ~4-6 h  (heavy — ~90 M variants)
# Conservation tracks (Zoonomia 241-mammal): scripts/setup/download_zoonomia_*.sh

# 4. Run the 5-gene E2E smoke test (HeLa, writes paired parquet)
python scripts/run.py --preset 5gene

# 5. Run tests
pytest
```

---

## GPU precompute

`plm_vep` and `structure` are **cache-lookup** modules — their caches must be
populated on a GPU node *before* annotation. The orchestrator does it as a
single Slurm job (emit FASTA → spawn GPU jobs → wait → full run):

```bash
mkdir -p logs
sbatch scripts/slurm/run.sbatch --preset 5gene
```

Or step by step:

```bash
python scripts/run.py --preset 5gene --emit-fasta                 # data/cache/proteins.fa
sbatch scripts/slurm/run_plm_embed.sbatch data/cache/proteins.fa  # ESM-2 650M → data/cache/plm_esm2/
sbatch scripts/slurm/run_fold.sbatch      data/cache/proteins.fa boltz  # → data/cache/structure/
python scripts/run.py --preset 5gene                              # caches now found
```

If the caches are empty the modules return empty cleanly — use `--no-gpu` to
skip them explicitly.

---

## Key design choices

**The canonical-vs-alternative contract lives in `contract.py`.** `ORFType`,
`orf_type_from_ribotish` (16 Ribo-TISH compound strings → 8 enum values), and
`diff_region_rule` (which coordinate space the isoform-unique differential
region occupies per ORF type) are the single source of truth everything
downstream depends on.

**Per-Tid canonical protein.** Each TIS gets `canonical_protein` from its own
transcript's Annotated row, not the gene-level longest — Ribo-TISH classifies
ORF type relative to each transcript's CDS. Gene-level longest stays on
`Gene.canonical_protein` as a representative for gene-level annotations.

**Symmetric canonical / isoform annotation.** Per-protein modules run on both
the canonical (cached in `Gene.canonical_annotations`) and each isoform (in
`site.isoform_annotations`). The comparator diffs the two dicts; it never stores
precomputed deltas in module output.

**Positional hits over scalars when possible.** Everything that *can* be stored
per-coordinate (motif matches, variants, conservation, peptides) *should* be;
scalars are reserved for inherently whole-protein properties. The comparator
filters positional hits to the differential-region coordinates at compare time.

**Authoritative `ConsequenceValidator`.** HGVSp from gnomAD/ClinVar is in
canonical-frame coordinates — wrong for alternative TIS. The validator always
runs with real CDS features (genomic→coding map, strand-aware); HGVSp is never
trusted for protein position.

---

## Website viewer

`website/` is a standalone Flask app over the paired-evidence parquet — a gene
grid plus per-gene pages showing each alternative TIS with its dual evidence
axis (E1–E6 / F1–F6), key metrics, pathogenic variants in the isoform-unique
region, an embedded Mol\* structure viewer, and an LLM-written interpretation
when available. It imports only `swissisoform.site.evidence` (numpy + pandas),
not the full analysis package, so the image stays small.

Live: <https://swissisoform-viewer-production.up.railway.app>

```bash
# Local dev
cd website
eval "$(conda shell.bash hook)" && conda activate swissisoform-v2
pip install -e .
flask --app swissisoform_site.app run --port 5050

# Deploy to Railway (Docker; data baked into the image)
./prepare_deploy.sh   # dereference website/data/ symlinks + stage site.evidence
railway up            # builds via Dockerfile, healthcheck at /healthz
```

`prepare_deploy.sh` dereferences the `website/data/` symlinks (which point at
the latest `data/output/cheeseman_13gene/` run) and stages
`swissisoform.site.evidence` into the build context. See `website/README.md`
for routes and data layout.

---

## Development

```bash
ruff check  src scripts tests      # lint (line length 100, Google docstrings)
ruff format src scripts tests      # format

pytest                             # ~770 tests; gpu/network markers excluded by default
pytest --cov=swissisoform --cov-report=term-missing
pytest tests/test_biophysics.py -v # single module
```

GPU- and network-marked tests are excluded by default
(`-m 'not gpu and not network'`).
`tests/regression/snapshot_paired.py` is the 13-gene regression harness —
`capture` writes a baseline fingerprint of `all_paired.parquet`, `compare`
asserts a later run is unchanged.

---

## References

- GENCODE v49: <https://www.gencodegenes.org/human/release_49.html>
- gnomAD v4.1: <https://gnomad.broadinstitute.org/downloads>
- ClinVar: <https://www.ncbi.nlm.nih.gov/clinvar/> · COSMIC v102: <https://cancer.sanger.ac.uk/cosmic>
- Zoonomia 241-mammal alignment: Christmas et al., _Science_ 380:eabn3943 (2023)
- Ribo-TISH: Zhang et al., _Nat Commun_ 2017 · DeepLoc 2.1: Thumuluri et al., _NAR_ 2022
- ESM-2: Lin et al., _Science_ 2023 · Boltz-2 / Chai-1 structure prediction
