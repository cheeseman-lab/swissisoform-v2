# CLAUDE.md — SwissIsoform v2

## Project

**SwissIsoform v2** — Modular pipeline for annotating alternative protein isoforms from translation initiation sequencing (TI-seq). Consolidates code from three repos (`swissisoform`, `tiap`, `coTISja`) into a unified 9-module architecture with rich domain objects and symmetric canonical/isoform annotation.

## What's Built

Current state only. When something changes, update this section in place rather than appending a dated entry.

### Upstream: per-sample TIS filtering (`run_sample`)

```
load predict_all.txt + GTF
  → recategorize TisType
  → merge per-gene HTSeq RNA-seq counts + total mapped reads
  → NormTISCounts = TISCounts / TotalRNASeqCounts × 1e6   (true RPM)
  → filter_tis (smaffa thresholds)
  → impute_missing_canonical_starts (GTF CDS + start_codon + pc_translations.fa)
  → drop uncanonical transcripts (cds_start_NF, retained_intron on coding genes)
  → (final_df, dropped_df) per sample
```

`scripts/run.py` is a thin front-end that builds a `RunSpec` and calls `runner.run()` (`runner.py`: `prepare` / `annotate` / `run`). Sample inputs come from `data/reference/ribotish_sample_manifest.csv` + `ribotish_replicate_manifest.csv` (6 cell lines → predict file + RNA-seq replicates) and `rnaseq_counts/*_htseqcount.txt`.

### Foundation layers

| Layer | Files | What it does |
|---|---|---|
| Filtering | `filtering.py` | 5-step TIS filter (ported from coTISja) |
| Merging | `merging.py` | Cross-cell-line combine, canonical-vs-alternative pairing |
| Assembly | `assembly.py` | DataFrame → `Gene` objects: canonical selection, ORF-type mapping, `DifferentialRegion` via sequence comparison, Kozak context from the genome FASTA |
| ORF coordinates | `models.py` (`TranscriptCoordinates`), `io/gtf.py` (`load_exon_skeletons`), `coords.py` | Layer 1: one exon skeleton (5'UTR + CDS + 3'UTR) per transcript. Layer 2: `orf_exons_from_skeleton` walks it from each ORF start for `aa_len × 3` nt, skipping introns, and populates `TIS.orf_exons`, `TIS.canonical_orf_exons` and `Gene.canonical_orf_exons`. `interval_difference` / `interval_intersection` / `interval_length` derive the unique and shared regions |
| Tag layer | `tags/`, `setup/tags.py`, `modules/tags.py` | Frozen tag vocabulary + cutoffs → tri-state `isoform_tags_*` columns (see Design Decisions) |

### Annotation modules

| Module | Location | Type | What it does |
|---|---|---|---|
| Biophysics | `modules/biophysics.py` | ProteinModule | Scalar properties (pI, GRAVY, …), inline |
| Motifs | `modules/motifs.py` | ProteinModule | Positional regex hits |
| Localization | `evidence/l1_localization/localization.py` | ProteinModule | DeepLoc lookup (precomputed) |
| Clinical | `clinical/module.py` | ProteinModule | gnomAD / ClinVar / COSMIC variants from local DBs, codon-level consequence validation |
| Mass spec | `evidence/d3_mass_spec/massspec.py` | ProteinModule | Tryptic digest + PepQuery search of isoform-unique peptides |
| Conservation | `modules/conservation.py` | SiteModule | Zoonomia 241-mammal PhyloP + PhastCons BigWig lookups at the TIS codon (3 nt) and Kozak window (13 nt, strand-aware), plus unique/shared region means and enrichment |
| Conservation frame | `conservation_frame/` | SiteModule | Primate (25) + mammalian (20) reading-frame integrity over the unique region: MAF parse, per-species start-codon / frameshift / premature-stop / identity, deepest intact species from the HAL species tree (`halStats --tree`). Statuses `not_run` / `no_skeleton` / `no_unique_region` / `no_alignment` / `ok` |
| Variant intersection | `modules/variant_intersection.py` | SiteModule | Tags each clinical hit as isoform-unique or shared by genomic membership; emits aggregate + pathogenic-in-unique counts |
| Variant effect | `modules/varianteffect.py` | SiteModule | Per-variant ESM-C masked-marginal ΔLLR (frame-aware) + AlphaMissense, aggregated over the unique region |
| PLM VEP | `plm/module.py` | SiteModule | ESM-C per-residue constraint from cached `logP(wt)`, unique vs shared region enrichment |
| SAE features | `plm/sae.py`, `plm/sae_module.py`, `plm/atlas.py` | SiteModule | Top-K SAE on the ESM-C residual stream (default 6B layer 60), features differentially active in unique vs shared region, with ESM-Atlas term labels (aligned only for 6B) |
| Structure | `structure/module.py` | SiteModule | Cached fold lookups (ESMFold2 default; Boltz-2 / Chai-1 supported) for canonical + isoform: diff-region pLDDT, TM-score, shared-region RMSD, contacts |
| Secondary structure | `structure/sse.py` | (in Structure) | P-SEA helix/strand elements with per-type length floors (`MIN_LENGTH = {helix: 5, strand: 3}`), each tagged `unique` / `shared` / `spans` |
| Core identity, initiation context | `modules/core_identity.py`, `modules/initiation_context.py` | SiteModule | TIS metadata |
| GeneRef | `modules/generef.py` | Gene-level | External reference context, not diffed |
| Evidence scoring | `modules/scoring.py` + `evidence/<criterion>/` | — | `EvidenceScoringModule`: 16 criteria on two axes, existence (C+D) and functional (L+M+P+S). Each returns True / False / None |

`modules/conservation_homology.py` (DIAMOND/blastp homology) is dormant: kept, not wired in.

GPU precomputes run out-of-band and are read as static inputs: `run_plm_embed.sbatch` (ESM-C embed, then the SAE encode in the same job; skip the SAE with `--skip-modules sae`) and `run_fold.sbatch` (structures).

### Not built

- **Full end-to-end on all 6 cell lines.**
- **Cross-validation against published Ribo-seq** (`crossval.py`): dropped, because the available human datasets are cross-species, gene-list-only, or lack GRCh38 coordinates. The replacement is feeding published human Ribo-seq in as additional upstream *inputs*, which is its own unscoped workstream.

## Design Decisions & Gotchas

Rules the code alone won't teach. Build history lives in `git log`, not here.

**Upstream (TIS filtering)**
- Upstream (`run_sample`) is a faithful port of coTISja's `filter_ribotish.py` and runs **per cell line, independently**: one `{sample}_TIS_filtered.parquet` each, and **no cross-sample merging at this layer**. Cross-cell-line comparison belongs downstream (assembly → annotation → comparator → `merging.py`).
- Ribo-TISH is the swappable piece. The contract is coTISja's filter + imputation output schema, so any TIS caller that produces the same filtered-DataFrame schema feeds the rest of the pipeline unchanged.

**Coordinates and canonicals**
- Genomic coordinates are **0-based half-open, plus-strand** throughout (`TranscriptCoordinates`, `coords.py`). mRNA-order concerns live in the walker (`orf_exons_from_skeleton`), not in the data.
- Each TIS's `canonical_protein` comes from **its own transcript's** Annotated row (`_build_canonical_by_tid` in `assembly.py`), not the gene-level longest: Ribo-TISH classifies ORF type relative to each transcript's CDS. `Gene.canonical_protein` stays the gene-level longest.
- Clinical fetchers always return `protein_pos=None`. HGVSp is canonical-frame, which is wrong for alternative TIS, so `ConsequenceValidator` is authoritative.

**Not evaluable ≠ absent**
- A value that couldn't be computed is `None`, never `False` or `0`. Keep statuses like `not_run` distinct from `no_hits`. Turning "could not evaluate" into "evidence absent" is the failure the tag layer (issue #30) exists to remove.

**Scoring naming (CDLMPS)**
- Criteria are named C/D/L/M/P/S all the way through `ScoringConfig` fields (`c1_pident_min`, `p1_plddt_threshold`, …). **The E/F axis is not criterion numbering and stays**: `"axis": "E"/"F"`, `EXISTENCE_CRITERIA` (C+D) / `FUNCTIONAL_CRITERIA` (L+M+P+S), and the `isoform_scoring_{existence,functional}_*` columns are the two-score framing the site renders.

**Tag layer** (additive, beside `EvidenceScoringModule`, which it never touches)
- Output: `isoform_tags_states` (struct of bool; **null = not-evaluable**), `isoform_tags_citations`, `isoform_tags_registry_version`, `isoform_tags_registry_sha256` (content hash of the tag definitions; `merge.py` refuses a campaign mixing two builds of one version).
- The registry is **provisioned reference data** in `data/reference/tags/<version>/`, built by `python scripts/setup/build_tag_registry.py --version <v> --cutoffs config`. `v3` is the default and **provisional**: it and distributions v3 were frozen from the Aug-12 `full_catalog`, which predates the M1 sign flip and gate (2a48f89), so re-freeze both as v4 after the next genome-wide run. `distributions.DEFAULT_VERSION` is the one default every reader and builder takes. `v1` is a historical artifact: rebuilding `--version v1` against today's table would emit reviewed contents under the v1 name.
- All sixteen criteria are `derived` tags that call the criterion's own scorer, so they equal the criterion **by construction**. Derived tags fire at the scorer's `ScoringConfig`; a registry `cutoff_overrides` entry is logged, not applied, so a tag and its criterion can never disagree (v3's swept S3 cutoff had given 7 of 50 cheeseman50 rows two S3 answers). Threshold forms dropped gates on 5 of 13 criteria.
- `--cutoffs config` reproduces today's scoring exactly, so build it first: a calibration finding must never be confusable with a wiring bug. Under `--cutoffs distribution`, D1/D3 land between integers and need cutoffs chosen on the integers.
- Firing-time guards apply to v3 without a rebuild: a `*_ratio` with a `*_shared` sibling is NA where shared ≤ 0 (a negative denominator inverts it); unique-region constraint metrics are valid only off `NO_CANONICAL_BASELINE_ORFS`, M1's own gate, and `Tag.label_for(orf_type)` names the unique region "extension" / "lost region" / "ORF" by ORF type and reads "gained" as "lost" on truncations; a `*_changed` bool with one side present is a change, not-evaluable only when the predictor did not run. Registries may carry a per-ORF-type `cutoff_by_stratum`; no build fills it yet.
- Tags run **last, over the finished frame** (`runner.annotate` → `_attach_tags`). A **missing registry warns and emits nothing** rather than failing the run.

**Validation**
- The canonical validation set is `cheeseman13` (`presets/cheeseman13.toml`; `python scripts/run.py --preset cheeseman13`). Integration is gated by the snapshot harness `tests/regression/snapshot_paired.py` (`capture` / `compare` on `all_paired.parquet`), not by pytest, which covers fast unit tests only.

## Documentation

| Document | Path | Purpose |
|----------|------|---------|
| Methods | `docs/methods/methods.typ` (+ `methods.bib`) | Publication-ready paper methods section (typst source; renders to `methods.pdf`) |
| Reviews | `docs/reviews/` | Code-review / gap-analysis documents from external reviewers |
| Architecture (current state) | `docs/architecture/` | Code-grounded descriptions of subsystems **as they exist** — no planned changes |
| Plans | `docs/plans/` | Feature/integration plans, each written **against** an architecture doc |

## Working Convention: Scope Before Plan

**Before designing or integrating any new feature, first write down what the
relevant part of the codebase *currently does* — then, and only then, plan the
change.** This separation is mandatory and keeps "what is" from getting tangled
with "what we want."

Two-step workflow:

1. **Scope (current state) → `docs/architecture/`.** Document the existing
   subsystem the feature touches. Rules:
   - Describe **only what exists today**. No proposed changes, no "we should,"
     no aspirational behavior. Planned work goes in step 2, never here.
   - **Ground every claim in code** with `file:line` references so the doc is
     reproducible and checkable against source.
   - Include **how to reproduce/regenerate** any artifact described (the exact
     command), and the **columns/schema** of every file the subsystem produces.
   - Date the doc and note the commit/baseline it was verified against.
   - Prefer updating an existing architecture doc over forking a new one.

2. **Plan (proposed change) → `docs/plans/`.** Only after the scope doc exists.
   The plan references the architecture doc, states the goal, the chosen
   insertion point(s), trade-offs, and the concrete edits. Keep current-state
   facts in the architecture doc; the plan links to them rather than restating.

Rationale: scoping first surfaces the real seams and constraints (where data is
available, what invariants hold) before committing to a design, and produces a
durable, reproducible reference that outlives any single change.

Worked example: `docs/architecture/upstream_filtering_and_dedup.md` scopes the
filter → merge → dedup pipeline (current state) ahead of the splice-aware
filtering feature, whose plan will live in `docs/plans/`.

## Source Repos (Read-Only Reference)

| Repo | Path | What It Contributes |
|------|------|---------------------|
| swissisoform v1 | `../swissisoform/` | BED parsing, translation, mutations, genome handling |
| TIAP | `../tiap/` | Modular annotation pipeline (14 modules), pipeline.py pattern |
| coTISja | `/lab/barcheese01/smaffa/coTISja/` | Ribo-TISH filtering, Kozak, expression normalization |

## Input Data

Raw Ribo-TISH predict files in `data/reference/` (gitignored, 6 cell lines):
- `HeLa_TIS_predict_all.txt`, `K562_TIS_predict_all.txt`, `U2OS_TIS_predict_all.txt`
- `RPE1_Async_TIS_predict_all.txt`, `RPE1_Que_TIS_predict_all.txt`, `RPE1_Sen_TIS_predict_all.txt`

Reference genome (download with `bash scripts/setup/download_references.sh`):
- `GRCh38.primary_assembly.genome.fa`, `gencode.v49.pc_translations.fa`, GTF

## Source-transcript resolution — alignment tool vs. tracked filtering

Pinning each TIS to one high-confidence source mRNA (long-read IsoQuant
expression + a sequence window-purity test) is **Elizabeth's workstream**. The
boundary sits between *read alignment* (out of repo) and *the disambiguation
science* (tracked filtering):

```
sourceseq/ (gitignored alignment tool)         │ boundary │  swissisoform-v2 (tracked)
reads → mapping (minimap2 / IsoQuant)          │ aligned  │  unified cascade:
→ transcript_counts.tsv ───────────────────────┼─► data/ ─┼─► long-read filter → window-purity
                                               │ quant    │  → abundance label → source per TIS
```

- **Out-of-repo (gitignored `sourceseq/`):** *only* read alignment +
  quantification — `mapping/` (minimap2 / IsoQuant / SRA + envs) and `setup/`
  (downloads). It writes the long-read quantification to `data/reference/`
  (`longread/isoquant_{cell}/OUT/…tsv`). That is the *processed data in* — like
  the Ribo-TISH predicts and HTSeq counts. See `sourceseq/README.md`.
- **Tracked (this repo):** the disambiguation **is our filtering**, so it lives
  in `src/swissisoform/sourceresolve/` (`mrna` / `purity` / `expression` /
  `resolve` / `collapse` / `diagnostics`) and runs as a **per-sample step inside
  `run_sample`** — it depends on that sample's own long-read RNA-seq, so it is
  intrinsically per cell line (HeLa only today). `resolve_sources` groups the
  filtered TIS by init_site and runs a **single linear cascade** per site:

  1. **long-read filter** — keep candidate transcripts present in IsoQuant
     (count ≥ `isoquant_min_count`); none survive ⇒ `window_status="no_support"`.
  2. **window-purity** (`purity_decision`) on the survivors, over independent
     `window_upstream` / `window_downstream` bounds (both default 100 nt).
  3. **abundance label** — *pure/single* → most-abundant survivor; *divergent*
     → top survivor must hold ≥ `divergence_dominance_frac` (default 0.5) of the
     divergent total, else `unresolved` (`None` ⇒ most-abundant-wins).

  It **tags** every TIS with `resolved` / `window_status` / `source_transcript`
  / `source_evidence` / `tie_initiation_efficiency`. Labels: `window_status` ∈
  `single|pure|divergent|no_support`; `source_evidence` ∈
  `window_pure|divergent_pass|no_support|unresolved` — the **only** `unresolved`
  sites are divergent ones that fail the threshold; long-read drop-outs are
  `no_support`. Tag-only here (full rows kept for audit). Short-read salmon was
  removed: long-read only, for both presence and abundance. Gated by
  `PipelineConfig.source_resolution` (built by `references.build_config`) + the
  sample's optional `isoquant_table` manifest column.

  **Collapse to one mRNA per TIS** — the verdict is consumed by
  `collapse_to_source` (`sourceresolve/collapse.py`) at the assembly boundary
  (`runner.prepare`, before `assemble_genes`): keeps all Annotated rows + each
  resolved site's source-transcript row, dropping non-resolved
  (`no_support`/`unresolved`) alt rows, so only resolved TIS — one mRNA each —
  advance. **Gated to rows a long-read sample actually scored:** a TIS called
  only in samples without long-read data (e.g. K562/U2OS/RPE1 when only HeLa has
  IsoQuant) has `NaN` in every `{sample}_resolved` column, was never evaluated,
  and passes through unchanged — so a single-long-read-sample phase keeps the
  full cross-sample TIS set alive for downstream `min_cell_lines` scoring. No-op
  when the verdict columns are absent.

  **CLI** (`scripts/run.py`, effective when the combined catalog is (re)built):
  `--skip-source-resolution` (disable cascade+collapse), `--divergence-threshold`
  (default 0.5), `--window-upstream` / `--window-downstream` (default 100). The
  divergent threshold is chosen empirically from
  `figures/mRNA_source_divergence/export_source_divergence_distribution.py` (per-site
  read distribution: CSV + quantiles + a 100%-stacked-bar plot, one bar per
  divergent TIS; CSV + PNG written alongside the script in
  `figures/mRNA_source_divergence/`).

## Development

```bash
eval "$(conda shell.bash hook)" && conda activate swissisoform-v2
uv pip install -e ".[dev]"
```

## Architecture

- **Input:** Raw Ribo-TISH `predict_all.txt` TSV (21 columns per cell line, includes `AASeq`)
- **Pipeline:** Read → Filter → Merge → Annotate (canonical + isoform) → Compare → Serialize
- **Domain objects** in `models.py`: `TranslationInitiationSite`, `Gene`, `DifferentialRegion`, `VariantAnnotation`
- **Module protocols** in `modules/base.py`: `ProteinModule` (`annotate(protein) -> dict`) and `SiteModule` (`annotate_site(site) -> dict`)
- **Wiring layer** in `pipeline.py`: `AnnotationPipeline` orchestrates ProteinModules on canonical (per-gene) + isoform (per-TIS), SiteModules per TIS, GeneModules per gene
- **Comparison** in `compare/comparator.py`: `Comparator` computes scalar deltas, categorical changes, and the positional subset of hits in the differential region (`hits_in_diff_region`); `compare/paired.py` holds the static `PairedComparison` helpers
- **Serialization** in `io/parquet.py`: round-trip TIS ↔ DataFrame ↔ Parquet

### Annotation → Comparison Design

Per-protein modules run **symmetrically on both canonical and isoform proteins**. A final comparator layer diffs the results.

```
Path 1: Annotate → Compare (per-protein modules)
  canonical_protein → [biophysics, motifs, localization, clinical, conservation] → canonical annotations
  isoform_protein   → [biophysics, motifs, localization, clinical, conservation] → isoform annotations
                                                                                       │
                                                                        comparator ◄───┘
                                                                        ├─ scalar deltas (Δ_pI, location_changed)
                                                                        └─ positional subset (hits in diff region)

Path 2: Gene-level context (no comparison, not diffed)
  gene_name → generef → attached as reference context
```

### Annotation Types

Modules produce two kinds of output:

- **Scalar** — whole-protein aggregate, no position (pI, GRAVY, localization prediction). Compared via delta.
- **Positional** — per-coordinate hits with `pos`/`end` fields (motif matches, variants, conservation per-residue). Compared via subset to differential region coordinates.

**Rule:** everything that CAN be stored per-coordinate SHOULD be. Scalars are only for inherently whole-protein properties. Counts and densities derived from positional hits are computed by the comparator from the filtered hit list, not stored in the module output.

### Differential Region Coordinates

- **Extensions:** `isoform[0 : delta_aa]`
- **Truncations:** `canonical[0 : abs(delta_aa)]` (the lost region)
- **uORFs/altORFs:** entire isoform (no shared region)

## Execution Contract — fresh reruns

**The CPU pipeline recomputes from scratch on every run. No step may rely on cached results of a prior run.** Identical inputs → identical outputs, computed fresh; no hidden accumulated state. Speed comes from parallelism and per-unit efficiency, never from skipping work via a results cache (the InterProScan non-reproducibility — 337→107 hits on rebuild — is the cautionary example).

The only persisted artifacts allowed are:

1. **Provisioned reference data** — genome, GTF, `pc_translations`, the clinical parquets (ClinVar / gnomAD / COSMIC), and the local PepQuery spectra library (`python -m swissisoform.setup.databases pepquery-spectra` mirrors the public PepQueryDB S3 library, ~196 GiB). These are *inputs* downloaded once via the setup phase; identical regardless of what runs against them.
2. **GPU precomputes** — ESM/PLM embeddings and Boltz structures, keyed by `protein_hash`. The *sole compute exception*, because they are prohibitively expensive inline; produced by the GPU sbatch scripts and treated as static inputs to the CPU run.

Everything else — PepQuery search, all annotation, scoring, comparison — runs fresh each run.

**Carve-out: intra-campaign shard resume.** The genome-wide harness
(`scripts/slurm/full_run/`) is the one sanctioned exception. Its annotate array
skips a shard whose `all_paired.parquet` already exists, which *is* a CPU result
cache — so the boundary is drawn explicitly:

- **The exception is scoped to one campaign.** A campaign is the unit of
  freshness, not a single job: `$SWISSISO_CAMPAIGN` namespaces the inputs
  (`data/output/$CAMPAIGN/`) and every shard output
  (`data/output/${CAMPAIGN}_shard_<k>/`). Within one campaign, resuming a
  117-shard multi-day array is resume, not caching — the alternative is losing
  days of completed work to one preempted task.
- **A fresh campaign gets a fresh name.** Any code, config, or gene-set change
  means a new `SWISSISO_CAMPAIGN` (it defaults to `full_catalog_<UTC date>`);
  never resubmit an old campaign's array after changing what the pipeline
  computes. The marker is keyed only to shard *contents*
  (`shard_meta.json`'s `shard_list_sha1`, checked by `merge.py`) — **not** to
  code version, config, or GPU-cache completeness, so nothing detects a
  stale-code or evidence-holed shard for you.
- **The marker means finished, not complete.** `runner.run` writes the per-gene
  slices and the scoring sidecar first, then renames `all_paired.parquet` into
  place atomically, and the array revalidates the trailing `PAR1` magic before
  skipping — so the marker cannot survive a job killed mid-write. But it attests
  that the *CPU annotation* finished, not that the *GPU precompute it read* was
  complete: under the `afterany` chain a shard whose embed/fold chunk died still
  annotates successfully, recording `status="no_cache"` for structure/PLM/SAE. It
  is then legitimately "done" and will be skipped forever. Those holes surface as
  `frac_no_cache_*` in `merge_report.tsv`, and clearing them means deleting the
  marker by hand — see the REFILL block in `00_prepare.sbatch`'s header.

**PepQuery implication:** the only contract-legal prep is **pre-downloading the spectra library** (reference data) — the `pepquery-spectra` setup target mirrors the public PepQueryDB S3 library locally so runs can search it via local `-ms` instead of re-pulling (and deleting) spectra from S3 every search. The search input is already scoped to the differential region: `collect_unique_peptides` submits only isoform-unique peptides (the isoform tryptic digest minus the canonical digest — a sequence set-difference, not a `diff_region` coordinate intersection), so canonical/shared peptides are never searched. There is no caching shortcut: on top of that scoping, a real PepQuery *speedup* still requires **sharding the search**, a fundamental pipeline architecture change (per-protein Snakemake DAG + a fresh, peptide-sharded PepQuery stage over the local library), tracked as its own project — not a quick win. *Local `-ms` wiring (done 2026-07-10):* `precompute_pepquery` now auto-detects the staged mirror (`data/reference/pepquery/spectra/<dataset>/`, via `_pepquery_local_library_dirs`) and searches each dataset from disk with one `-ms <folder>` invocation (PepQuery reads the mass-binned `*.mgf.gz` index directly — verified: no re-index, no S3), aggregating per-dataset outputs exactly like the `-b` layout; it falls back to `-b` (on-demand download) only when a dataset isn't mirrored. This removes the per-shard re-download across the 117-shard annotate array once `setup_databases.py pepquery-spectra` has run. *Known deviation still to address:* `precompute_pepquery`'s on-disk result cache (`data/cache/pepquery/*.json`) is a CPU result cache that violates this contract.

## Module Contract

All modules must:
1. Define `MODULE_NAME`, `OUTPUT_COLUMNS`, `SCOPE` as class attributes
2. Keep `run(tis_sites)` as a backward-compatible wrapper that writes to `site.isoform_annotations[MODULE_NAME]`
3. Never drop sites (`len(output) == len(input)`)
4. Use `None` for values that can't be computed

**ProteinModules** additionally:
- Implement `annotate(protein: str) -> dict[str, Any]` as a pure function
- The wiring layer calls this on canonical (→ `gene.canonical_annotations[MODULE_NAME]`) and each isoform (→ `tis.isoform_annotations[MODULE_NAME]`)
- Output format:
  - **Scalar**: plain values (float, str, bool)
  - **Positional**: `{"hits": [{name, pos, end, ...}, ...], "summary": {...}}`

**SiteModules** additionally:
- Implement `annotate_site(site: TranslationInitiationSite) -> dict[str, Any]`
- Use when the module needs TIS metadata (orf_type, kozak_context) beyond just the protein sequence
- Only ever run on TIS sites (never on canonical proteins)

**Rule:** everything that CAN be stored per-coordinate SHOULD be. Scalars are only for inherently whole-protein properties. Counts and densities derived from positional hits are computed by the comparator from the filtered hit list, not stored in the module output.

## Tests

```bash
# Unit tests (gpu/network markers excluded by default; integration is the cheeseman13 snapshot harness)
pytest

# Single module
pytest tests/test_biophysics.py -v
```

## Code Style

- Linter/formatter: `ruff` (line length 100, Google docstrings)
- Type annotations required
- Tests: `pytest` with synthetic fixtures in `conftest.py`
- Module names are single words (no underscores) to avoid Parquet column prefix ambiguity

### No generated `*.md` reports

**Export and build scripts write data, never a prose `report.md` / `summary.md`
companion.** Print the summary to stdout if a human wants it at run time; the
artifact on disk is the CSV/TSV/parquet.

A generated markdown file is a second copy of numbers that already live in the
data file, and it goes stale the moment the data file is edited by hand — which
is routine here, since `tag_candidates.csv` carries a human `decision` column.
Two artifacts then give two answers to the same question and nothing says which
is current. Removed on this basis: `tag_review.md` (`tags/candidates.py`) and
`feature_catalog_summary.md` (`export_feature_catalog.py`).

This does not cover **hand-written** docs (`docs/architecture/`, `docs/plans/`,
this file) or machine-readable provenance sidecars (`_setup.json`,
`merge_report.tsv`) — those are the right way to record the same facts.

### Reuse before you add

Before writing a helper, grep for one that already exists and import it. A
second copy of `_sha256` or a path resolver is not free: the copies drift, and a
fix lands in one of them. Shared spellings and magic strings belong in one
module that both sides import — `metrics.LEN_SUFFIX` is the worked example,
after the profiler and the runtime disagreed about `__len` and silently shipped
a tag that could never fire.
