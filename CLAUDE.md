# CLAUDE.md — SwissIsoform v2

## Project

**SwissIsoform v2** — Modular pipeline for annotating alternative protein isoforms from translation initiation sequencing (TI-seq). Consolidates code from three repos (`swissisoform`, `tiap`, `coTISja`) into a unified 9-module architecture with rich domain objects and symmetric canonical/isoform annotation.

## Status

**All modules ported + hardened against shortcuts after code review.** 347 tests (317 fast + 30 slow), all passing. Pipeline orchestration, assembly layer (with Kozak FASTA extraction, required-column validation, cross-transcript fallback warnings) done.

**Upstream now a faithful coTISja port** — see "Upstream port (2026-04-18)" below. Assembly still on the old upstream signature; migrating it is the next step.
Next up: simplify assembly to consume the new upstream output; then wire expensive modules into E2E (massspec, clinical cache, conservation cache); then comparator, CLI.

### Upstream port (2026-04-18)

Rewrote `run_upstream` as `run_sample` — a faithful end-to-end port of `coTISja/src/scripts/filter_ribotish.py`, one sample (cell line) at a time:

```
load predict_all.txt + GTF
  → recategorize TisType
  → merge per-gene HTSeq RNA-seq counts + total mapped reads
  → NormTISCounts = TISCounts / TotalRNASeqCounts × 1e6   (true RPM)
  → filter_tis (smaffa thresholds, exempt_annotated=False)
  → impute_missing_canonical_starts (GTF CDS + start_codon + pc_translations.fa)
  → drop uncanonical transcripts (cds_start_NF, retained_intron on coding genes)
  → (final_df, dropped_df) per sample
```

**Multi-cell-line design decision:** upstream runs per cell line, independently. The pipeline writes one `{sample}_TIS_filtered.parquet` per sample. **No cross-sample merging at this layer.** Cross-cell-line comparison is a *downstream* concern (assembly → annotation → comparator → `merging.py`). This matches smaffa's design and decouples filtering from differential analysis.

**Audit (validated then retired 2026-05-26):** `tests/test_smaffa_audit.py` compared our HeLa output row-for-row against `data/reference/smaffa_filtered_audit/` (ours ⊆ smaffa; difference = the uncanonical drop). Final run: 8 passed. The test + the 38M reference were then removed — re-derivable from smaffa source (`/lab/barcheese01/smaffa/coTISja/`) if ever needed.

**Imputation behaviour on HeLa:**

| | count |
|---|---|
| Filter output (native) | 33,783 rows — exact match to smaffa |
| Imputed canonicals | +5,933 synthetic Annotated rows (e.g. EIF4G1 goes from 1 Annotated to 13) |
| Uncanonical drop | −1,337 rows on 223 transcripts (cds_start_NF + non-coding-on-coding-gene) |
| Final | ~38,400 rows, every Tid has a canonical |

**Files copied into `data/reference/` (end-to-end self-contained):**

| Item | Source | Purpose |
|---|---|---|
| `ribotish_sample_manifest.csv` + `ribotish_replicate_manifest.csv` | smaffa (rewritten with relative paths) | 6-cell-line sample → predict file + RNA-seq replicates |
| `rnaseq_counts/*_htseqcount.txt` (12) | `/lab/barcheese01/aTIS_data/counts/` | TIS-condition HTSeq counts for RPM normalization |
| `gencode.v49.pc_translations.fa` | smaffa | Imputation — AASeq/AALen per transcript |
| `Gencode_v49_GRCh38.primary_assembly.genome.fa` | smaffa (replaced the chr3-only dev FASTA) | Imputation — start-codon trinucleotides |

**Scripts:**
- `scripts/run.py` — thin front-end that builds a `RunSpec` and calls `runner.run()` (`src/swissisoform/runner.py`: `prepare`/`annotate`/`run`). Drives the pipeline end-to-end and produces `data/output/filtered/{sample}_TIS_filtered.parquet` per cell line.

**Explicit: Ribo-TISH itself is the long-term swappable piece.** This upstream port locks us in to coTISja's filter + imputation contract, not to Ribo-TISH. Any future TIS caller can produce the same filtered-DataFrame schema and feed the rest of the pipeline unchanged.

### Correctness hardening (2026-04-17 code review round)

Shortcuts removed and made honest:
- **Pipeline context dispatch** — `AnnotationPipeline` now uses introspection to pass `gene_name` and `canonical_protein` to modules whose `annotate()` signature accepts them (clinical, massspec). Pure-function modules (biophysics, motifs) unchanged.
- **HGVSP shortcut** — `VariantFetcher` no longer sets `protein_pos` from HGVSp (which is canonical-frame, wrong for alternative TIS). Fetchers always return `protein_pos=None`; `ConsequenceValidator` is authoritative. Canonical-frame hint preserved in `metadata.hgvsp_canonical_hint`.
- **MassSpec unique_to_isoform** — returns `None` (unknown) when canonical protein not provided, not `False`.
- **Clinical silent-empty** — emits WARN when `gene_name` is empty (almost always a wiring bug).
- **Conservation status semantics** — distinguishes "not_run" (score=None) from "no_hits" (score=0). Adds `status` field to output.
- **ClinVar ref/alt** — parsed from the HGVSc notation in the title instead of hardcoded `""`.
- **Assembly cross-transcript diff_region** — logs WARN when falling back to whole-isoform; adds tail verification to prefix-match heuristic.
- **Assembly required columns** — raises on missing columns instead of silently returning None values.
- **Kozak context** — populated from genome FASTA (strand-aware 13-nt window, ATG at indices 9-11) when `genome_fasta` is provided to `assemble_genes`.
- **GeneRefModule** — added `annotate_gene(gene)` so it can be wired into `AnnotationPipeline` as a gene_module.
- **InitiationContext** — removed four always-None output columns; schema now matches computed output.
- **E2E tests** — replaced "field exists" assertions with value assertions (pI in plausible range, Kozak ATG at correct indices, motif hits found for TP53, etc.).
- **Per-transcript canonical for TIS** — `assemble_genes` now picks each TIS's `canonical_protein` from its own transcript's Annotated row (`_build_canonical_by_tid`). Ribo-TISH classifies ORF type relative to each transcript's CDS, so comparing against the gene-level longest was wrong. When a Tid has no Annotated row in the Ribo-TISH output, falls back to gene-level canonical with a one-shot INFO log per (gene, Tid). `Gene.canonical_protein` stays the gene-level longest (representative for gene-level annotations).
- **Upstream pipeline wired into E2E** — new `run_upstream(ribotish_path, gtf_path, config, sample=None) -> (filtered_df, annotated_df)` in `pipeline.py`. Runs `load_ribotish_predictions(gtf_path=...)` → `recategorize_tis_type` → `normalize_tis_counts` → `filter_tis`. E2E now tests filtered input (~100 TIS over 5 genes, not 789).
- **Exempt Annotated from filter significance** — `filter_tis(exempt_annotated=True)` (new default) keeps Annotated rows from reference transcripts regardless of re-detection p-values. Annotated comes from GTF coordinates, so significance-filtering it would strip canonical reference sequences. `FilterConfig.exempt_annotated` plumbs the flag.
- **Canonical source decoupled from alt-TIS filtering** — `assemble_genes(annotated_df=...)` accepts a separate canonical-source DataFrame. `run_upstream`'s `annotated_df` carries ALL `TisType == "Annotated"` rows (regardless of filter outcome or transcript support), so downstream per-Tid canonical lookup survives even when filtering drops an Annotated row from the alt stream.
- **E2E gene set swap** — `PPP1R15A` replaced with `MYC` in `TEST_GENES`. PPP1R15A has no TSL-1/2/3 transcripts in HeLa so the filter drops it entirely; MYC has CUG-initiated MYC1 extension biology and survives filter with Ann/Ext/Trunc representation.

### What's Built — Foundation

| Layer | Files | What It Does |
|-------|-------|-------------|
| Domain model | `models.py`, `config.py` | `TIS`, `Gene`, `DifferentialRegion`, `ORFType`, `PipelineConfig` |
| Module protocols | `modules/base.py` | `ProteinModule`, `SiteModule` protocols with validation |
| I/O | `io/ribotish.py`, `io/gtf.py`, `io/parquet.py` | Ribo-TISH TSV reader (with AASeq), GTF transcript + CDS loaders, Parquet round-trip |
| Filtering | `filtering.py` | 5-step filtering (ported from coTISja) |
| Merging | `merging.py` | Cross-cell-line combine, canonical-vs-alternative pairing |
| Assembly | `assembly.py` | DataFrame → Gene objects: canonical selection, ORF type mapping, DifferentialRegion via sequence comparison |
| Pipeline | `pipeline.py` | `AnnotationPipeline` wiring — runs ProteinModules on canonical + isoform, SiteModules per TIS, GeneModules per gene |
| Comparator | `compare/paired.py` | Shared canonical-vs-isoform delta logic (needs positional subset extension) |

### What's Built — Modules

All 9 modules implemented. Data model supports symmetric canonical/isoform annotation with differential region coordinates.

| Module | Protocol | Interface | Mode |
|--------|----------|-----------|------|
| `biophysics.py` | ProteinModule | `annotate(protein) -> dict` (scalar) | Inline (pure Python) |
| `motifs.py` | ProteinModule | `annotate(protein) -> dict` (positional hits) | Inline (regex) |
| `localization.py` | ProteinModule | `annotate_by_key(key) -> dict` (lookup) | Lookup (DeepLoc results) |
| `clinical.py` | ProteinModule | `annotate(protein, gene_name) -> dict` (positional variants) | Inline fetch (gnomAD/ClinVar/COSMIC) + codon-level consequence validation, or cache |
| `conservation.py` | SiteModule | `annotate_site(site) -> dict` (nucleotide-level scores) | Zoonomia 241-mammal PhyloP + PhastCons BigWig lookups (point-based: TIS codon + Kozak). Region means stubbed until protein→genomic CDS mapper lands. |
| `massspec.py` | ProteinModule | `annotate(protein, canonical, gene_name) -> dict` (positional peptides) | Inline tryptic digest + optional PepQuery lookup |
| `core_identity.py` | SiteModule | `annotate_site(site) -> dict` | TIS metadata |
| `initiation_context.py` | SiteModule | `annotate_site(site) -> dict` | TIS metadata |
| `generef.py` | Gene-level | `annotate_by_gene(gene_name) -> dict` (lookup) | External reference data |

### Clinical Subpackage

| File | Purpose |
|------|---------|
| `clinical/fetch.py` | `VariantFetcher`: gnomAD (GraphQL POST), ClinVar (NCBI E-utilities), COSMIC (local parquet via PyArrow) |
| `clinical/validate.py` | `ConsequenceValidator`: genomic→coding position map from GTF CDS, codon-level SNV/indel analysis, strand-aware |

### Roadmap

| Step | What | Status |
|------|------|--------|
| 1. Wiring layer | `pipeline.py` — AnnotationPipeline | **Done** |
| 2. Harder modules | clinical, conservation, massspec | **Done** |
| 3. Assembly + real test | `assembly.py` + 5-gene E2E on HeLa Ribo-TISH data | **Done** |
| 3b. **Expensive modules E2E** | massspec, clinical (+ ConsequenceValidator), conservation (batch), localization (DeepLoc precompute) wired; all 3 clinical DBs (gnomAD + ClinVar + COSMIC) built from source and queried locally | **Done (2026-04-19)** |
| 4. Comparator extension | Positional subset to diff region + scalar deltas | **Done** |
| 4b. Conservation rewrite | Zoonomia PhyloP/PhastCons BigWig SiteModule (point-based) | **Done (2026-04-21)** |
| 4c. ORF exon infrastructure | `TranscriptCoordinates` skeleton + Layer-2 walker; conservation region means (unique/shared/enrichment) now live. Unblocks clinical genomic intersection, Scope-A positional subsetting for all genomic modules, Evo 2 DNA extraction. | **Done (2026-04-21)** |
| 4d. Conservation Path 1/2 | Primate + mammalian reading-frame integrity module (MAF parse, frame analysis, `hal2maf` wrapper, species lists, Cactus species-tree depth). Pure logic fully tested; active on download of Zoonomia HAL. | **Done (2026-04-21)** |
| 4e. Clinical genomic intersection | `VariantIntersectionModule` (SiteModule): tags each clinical hit with genomic membership in isoform-unique / shared ORF region, emits aggregate + pathogenic-in-unique counts. Uses `orf_exons` + `canonical_orf_exons`. | **Done (2026-04-21)** |
| 4f. Evidence scoring framework | `EvidenceScoringModule`: dual-axis E1–E7 (existence) + F1–F6 (functional impact). Each criterion returns True / False / None so unbuilt-module criteria (structure/functional/VEP/proteomics) gracefully report unavailability. Wires up to `conservation_frame`, `conservation`, `massspec`, `variant_intersection`, `comparison["localization"]` today. | **Done (2026-04-21)** |
| 5. CLI | `scripts/run.py` thin front-end (RunSpec → `runner.run`) | **Done** |
| 7. Functional / Structure / VEP stubs | Precompute+lookup modules (InterProScan, Chai-1, AlphaMissense) returning None until data exists | Pending |
| 4g. PLM VEP (ESM-2 LLR) | `PLMVEPModule` (SiteModule) + `swissisoform.plm.embed` cache (`<hash>.npz`, on-disk). Masked-marginal LLR per residue, unique vs shared region enrichment using `diff_region` coords (canonical-space for truncations, isoform-space otherwise). Precompute via `scripts/slurm/run_plm_embed.sbatch` (ESM-2 650M, A6000). InterPLM SAE feature path stubbed (cache stashes layer-18 embeddings) — feature module is a follow-up. | **Done (2026-04-28)** |
| 4h. ESM-C SAE features (Part 3 of ESM migration) | Top-K sparse-autoencoder interpretability on the ESM-C 600M layer-27 residual stream (`biohub/ESMC-600M-sae-k64-codebook16384`). `embed.py` caches `embedding_sae` (layer 27); `plm/sae.py` encodes it to sparse `(L,64)` features in `data/cache/sae_esmc/`; `plm/sae_module.py` `SAEFeatureModule` (SiteModule) ranks features differentially active in the isoform-unique vs shared region (`diff_region`-based, like PLM VEP); `plm/atlas.py` caches the ESM-Atlas term dictionary (`data/reference/sae_atlas/`, **PLACEHOLDER — describes the 6B-layer60 dictionary, not the 600M, so feature #N labels are provisional**). `scripts/export/sae_top_terms.py` writes a whole-protein top-5-terms CSV. | **Productionized (2026-06-23): the SAE encode now rides the embed GPU job — `run_plm_embed.sbatch` runs `python -m swissisoform.plm.sae` after the embed, so a fresh `run.sbatch` lands `sae_esmc/` automatically (skip with `--skip-modules sae` → `SWISSISO_SKIP_SAE=1`). `SAEFeatureModule` is wired into the annotate stage (`runner.py`), and `sae_module`/`export_sae_comparison.py` emit the 4 counts + top-30 `sae_feature_changes.parquet`; `all_paired.parquet` carries the `isoform_sae_*` columns. `scripts/export/sae_top_terms.py` remains a standalone manual diagnostic (whole-protein top-5 terms); the redundant `_sae_step1.sbatch` harness and `_verify_sae_layer.py` were retired. Still descriptive — no E/F scoring criterion; LLM evidence record/website panel surfacing is the remaining follow-up.** |
| 8. Full end-to-end | All modules on real data, all 6 cell lines | Pending |

### Canonical validation set — cheeseman13

The canonical validation set is `cheeseman13`: 13 reviewer-picked isoforms
defined in `presets/cheeseman13.toml`. Run end-to-end via
`python scripts/run.py --preset cheeseman13` (also the default with no mode
flag). Integration is gated by the snapshot regression harness
`tests/regression/snapshot_paired.py` (`capture` writes a baseline fingerprint
of `all_paired.parquet`; `compare` asserts a later run is unchanged), not a
pytest E2E — `tests/test_endtoend.py` has been deleted, and pytest now covers
fast unit tests only.

### Conservation rewrite (2026-04-21)

Swapped the homology-based `ConservationModule` (DIAMOND/blastp/MMseqs2 against
SwissProt) for a BigWig-lookup SiteModule backed by the Zoonomia 241-mammal
Cactus alignment (Christmas et al. 2023).  Rationale in
`docs/reviews/conservation_module_spec.md`.

- New `modules/conservation.py` — SiteModule that opens PhyloP + PhastCons
  BigWigs once per worker and reads scores at the TIS start codon (3 nt) and
  Kozak window (13 nt: −9..+4 mRNA, strand-aware).  Distinguishes `not_run`
  (no BigWig/config) from genuine missing values.
- Old homology module preserved as `modules/conservation_homology.py`
  (`ConservationHomologyModule`) — dormant, not wired into the pipeline, kept
  so protein-similarity evidence can be reintroduced as a separate module.
- `ConservationConfig` now carries `phylop_bigwig` + `phastcons_bigwig` as
  first-class fields; `diamond_db` / `tblastn_db` kept as dormant.
- CLI: `--diamond-db` replaced by `--phylop-bigwig` / `--phastcons-bigwig`.
  Homology precompute path removed (BigWig random access is cheap).
- `scripts/setup/download_zoonomia_bigwigs.sh` — idempotent fetch of the two
  tracks (~13 GB total) from UCSC into `data/reference/zoonomia/`.
- Path 1/2 from the spec (primate + mammalian MAF frame-intactness) —
  scaffolded 2026-04-21 (see below); active once the HAL download lands.

### Conservation Path 1/2 (2026-04-21)

Primate + mammalian reading-frame integrity per
`docs/reviews/conservation_path12_spec.md`. Pure logic tested end-to-end;
HAL-dependent path emits `status="not_run"` until the download lands.

- `src/swissisoform/conservation_frame/` subpackage:
  - `maf.py` — MAF parser (`parse_maf`, `concat_species_rows`).
  - `frame.py` — per-species `analyze_species` (start-codon conservation,
    frameshift detection, premature-stop scan, AA pident) plus
    `aggregate_species_results`.
  - `species.py` — curated `PRIMATE_SPECIES` (22) and `MAMMALIAN_SPECIES`
    (23) lists in UCSC assembly names; refinable via `halStats --genomes`.
  - `hal.py` — `hal2maf` subprocess wrapper with graceful `None` on
    missing binary / HAL / non-zero exit.
- `conservation_frame/module.py` — `ConservationFrameModule` (SiteModule)
  consumes `TIS.orf_exons` and `TIS.canonical_orf_exons`, queries the
  unique region via `hal2maf`, reports primate/mammalian aggregates.
  Distinguishes `not_run` / `no_skeleton` / `no_unique_region` /
  `no_alignment` / `ok`.
- `ConservationConfig` gains `hal_path`, `hal_ref_genome`,
  `hal2maf_binary`, `primate_species`, `mammalian_species`.
- `scripts/setup/download_zoonomia_hal.sh` — resumable curl with provenance
  sidecar; 200-600 GB, run on demand.
- Phylogenetic depth: `conservation_frame/tree.py` parses the HAL's own
  species tree (via `halStats --tree`, with `hal_tree_newick` config
  override for tests / offline runs) into a depth map. The module emits
  `primate_deepest_species` / `primate_max_depth` and mammalian twins —
  the deepest-MRCA species whose frame is still intact. Named clade
  labels are deliberately not emitted; depth + species is enough and
  doesn't drift with Zoonomia releases.
- Tests: `test_conservation_frame.py` (MAF parse, frame analysis: identity,
  substitution, premature stop, start-codon loss, frameshift vs. in-frame
  deletion, all-gap target), `test_conservation_frame_module.py`
  (not_run paths, no_skeleton / no_unique_region, revcomp MAF helper,
  deepest-intact selection with a synthetic tree), and
  `test_conservation_tree.py` (Newick parser, MRCA depth).

### ORF exon infrastructure (2026-04-21)

Landed the protein→genomic mapper that Conservation's region metrics
(and a growing queue of other modules) were blocked on. Two layers:

**Layer 1 — transcript skeleton, shared per transcript_id:**
`TranscriptCoordinates` dataclass (`models.py`): full exon structure
(5'UTR + CDS + 3'UTR), `cds_start`, `cds_end`, chrom, strand. Built once
at GTF loading time by `load_exon_skeletons` (`io/gtf.py`), held on
`UpstreamReference.exon_skeletons`. Coordinates are 0-based half-open
plus-strand throughout; mRNA-order concerns live in the walker, not the
data.

**Layer 2 — per-ORF genomic intervals:**
`orf_exons_from_skeleton(coords, orf_start_genomic, aa_len)`
(`coords.py`): walks the skeleton from each ORF's genomic start forward
through `aa_len * 3` nucleotides, skipping introns. Strand-aware.
`assemble_genes(..., exon_skeletons=...)` populates:

- `Gene.canonical_orf_exons` (gene-level longest Annotated)
- `TIS.orf_exons` (per-TIS isoform ORF)
- `TIS.canonical_orf_exons` (per-Tid canonical, matching `TIS.canonical_protein`)

Also shipped `interval_difference` / `interval_intersection` /
`interval_length` in `coords.py` — genomic set algebra used to derive
unique vs. shared regions.

**Conservation region metrics now live.** `modules/conservation.py`
computes `phylop_unique_region_mean`, `phylop_shared_region_mean`,
`phylop_enrichment`, and the phastcons twins by:
1. `unique = site.orf_exons \ site.canonical_orf_exons`
2. `shared = site.orf_exons ∩ site.canonical_orf_exons`
3. Length-weighted mean over each interval set, enrichment = unique/shared.
Stub status `region_map_not_implemented` retired — now returns
`region_status="ok"` or `"no_skeleton"`.

**What this unblocks (per handoff):**
- Clinical isoform-level variant intersection (gnomAD/ClinVar coords vs `orf_exons`)
- Motifs / clinical / massspec Scope-A positional subsetting (genomic path)
- Evo 2 / AlphaGenome DNA sequence extraction
- Conservation Path 1/2 MAF extraction over unique regions (pending HAL download)

### Deferred (unclear value or needs redesign)

| Module | Source | Complexity | Reason |
|--------|--------|------------|--------|
| `scoring.py` | TIAP | Medium | Needs full redesign around comparison outputs, not a port |
| `crossval.py` | TIAP | Medium | Dropped 2026-04-19 after attempted port: the available human datasets (Ingolia, QTI, Fedorova, Kagan) are either cross-species, gene-list-only, or without GRCh38 coordinates — the tiered-matching framework can't exercise Tier 1/2 against them. The correct replacement is to treat published human Ribo-seq studies (Chen/Weissman, Chothani, etc.) as additional *inputs* to our upstream pipeline, giving coordinate-level confirmation by construction. That is its own workstream (raw read alignment + matched RNA-seq + Ribo-TISH re-run) and has not been scoped. |

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
  `figures/source_divergence/export_source_divergence_distribution.py` (per-site
  read distribution: CSV + quantiles + a 100%-stacked-bar plot, one bar per
  divergent TIS; CSV + PNG written alongside the script in
  `figures/source_divergence/`).

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
- **Paired comparison** in `compare/paired.py`: shared canonical-vs-isoform delta logic (needs positional subset extension)
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

**PepQuery implication:** the only contract-legal prep is **pre-downloading the spectra library** (reference data) — the `pepquery-spectra` setup target mirrors the public PepQueryDB S3 library locally so runs can search it via local `-ms` instead of re-pulling (and deleting) spectra from S3 every search. The search input is already scoped to the differential region: `collect_unique_peptides` submits only isoform-unique peptides (the isoform tryptic digest minus the canonical digest — a sequence set-difference, not a `diff_region` coordinate intersection), so canonical/shared peptides are never searched. There is no caching shortcut: on top of that scoping, a real PepQuery *speedup* still requires **sharding the search**, a fundamental pipeline architecture change (per-protein Snakemake DAG + a fresh, peptide-sharded PepQuery stage over the local library), tracked as its own project — not a quick win. *Known deviations to address:* runtime still uses `-b` (S3) until wired to local `-ms`; and `precompute_pepquery`'s on-disk result cache (`data/cache/pepquery/*.json`) is a CPU result cache that violates this contract.

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
# All tests (unit + real-genome integration)
pytest

# Single module
pytest tests/test_biophysics.py -v
```

## Code Style

- Linter/formatter: `ruff` (line length 100, Google docstrings)
- Type annotations required
- Tests: `pytest` with synthetic fixtures in `conftest.py`
- Module names are single words (no underscores) to avoid Parquet column prefix ambiguity
