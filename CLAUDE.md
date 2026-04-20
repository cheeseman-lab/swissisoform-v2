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

**Multi-cell-line design decision:** upstream runs per cell line, independently. The pipeline writes one `{sample}_TIS_filtered.csv` per sample. **No cross-sample merging at this layer.** Cross-cell-line comparison is a *downstream* concern (assembly → annotation → comparator → `merging.py`). This matches smaffa's design and decouples filtering from differential analysis.

**Audit:** `tests/test_smaffa_audit.py` compares our HeLa output row-for-row against `data/reference/smaffa_filtered_audit/HeLa_TIS_filtered.csv`. All 5 test genes pass (ours ⊆ smaffa; difference is exactly the uncanonical drop).

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
| `smaffa_filtered_audit/*_TIS_filtered.csv` (6) | smaffa | Regression reference — we must match this |
| `gencode.v49.pc_translations.fa` | smaffa | Imputation — AASeq/AALen per transcript |
| `Gencode_v49_GRCh38.primary_assembly.genome.fa` | smaffa (replaced the chr3-only dev FASTA) | Imputation — start-codon trinucleotides |

**Scripts:**
- `scripts/run_upstream_all.py` — drives the manifest end-to-end, produces `data/output/filtered/{sample}_TIS_filtered.csv` for all 6 cell lines.

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
| `conservation.py` | ProteinModule | `annotate(protein) -> dict` (positional alignment hits) | Inline subprocess (DIAMOND/blastp/MMseqs2) or cache |
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
| 4. Comparator extension | Positional subset to diff region + scalar deltas | **Next** |
| 5. CLI | `__main__.py` entry point | Pending |
| 6. Full end-to-end | All modules on real data, all 6 cell lines | Pending |

### End-to-End Test Genes (5-gene diagnostic set)

| Gene | Strand | ORF types | Exons | Intron-crossing | Key axis |
|------|--------|-----------|-------|-----------------|----------|
| TP53 | - | Ext, Trunc, uORF | 11 | Yes (1 intron) | Minus strand + multi-ORF |
| EIF4G1 | + | uORF, Trunc | 33 | Yes (26 introns) | Many exons + extreme length range |
| VEGFA | + | Trunc, Novel | 8 | No | Known biology + within-exon truncation |
| CTNND1 | + | Trunc, Internal, uORF | 21 | Yes (2 introns) | Cross-transcript ORF mismatch |
| PPP1R15A | + | uORF, Ext | 3 | No | Small gene + uORF edge case |

### Deferred (unclear value or needs redesign)

| Module | Source | Complexity | Reason |
|--------|--------|------------|--------|
| `scoring.py` | TIAP | Medium | Needs full redesign around comparison outputs, not a port |
| `crossval.py` | TIAP | Medium | Gene-level dataset matching; unclear if spec adds meaningful evidence |

## Documentation

| Document | Path | Purpose |
|----------|------|---------|
| **Design spec** | `docs/superpowers/specs/2026-04-14-swissisoform-v2-orchestration-design.md` | Domain model, module contracts, wave execution strategy |
| **Wave 1 plan** | `docs/superpowers/plans/2026-04-14-wave1-foundation.md` | Completed implementation plan |
| Scientific overview | `docs/tis_projects_overview.md` | Biology, 2-paper structure, evidence scoring framework |
| Technical spec | `docs/tis_technical_spec.md` | Module specs 0-12, code provenance, implementation phases |
| Gap analysis | `docs/swissisoform_review.md` | Missing modules, scoring expansion, tool recommendations |

## Source Repos (Read-Only Reference)

| Repo | Path | What It Contributes |
|------|------|---------------------|
| swissisoform v1 | `/lab/barcheese01/mdiberna/swissisoform/` | BED parsing, translation, mutations, genome handling |
| TIAP | `/lab/barcheese01/mdiberna/tiap/` | Modular annotation pipeline (14 modules), pipeline.py pattern |
| coTISja | `/lab/barcheese01/smaffa/coTISja/` | Ribo-TISH filtering, Kozak, expression normalization |

## Input Data

Raw Ribo-TISH predict files in `data/reference/` (gitignored, 6 cell lines):
- `HeLa_TIS_predict_all.txt`, `K562_TIS_predict_all.txt`, `U2OS_TIS_predict_all.txt`
- `RPE1_Async_TIS_predict_all.txt`, `RPE1_Que_TIS_predict_all.txt`, `RPE1_Sen_TIS_predict_all.txt`

Reference genome (download with `bash scripts/download_references.sh`):
- `GRCh38.primary_assembly.genome.fa`, `gencode.v49.pc_translations.fa`, GTF

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
