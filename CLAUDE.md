# CLAUDE.md — SwissIsoform v2

## Project

**SwissIsoform v2** — Modular pipeline for annotating alternative protein isoforms from translation initiation sequencing (TI-seq). Consolidates code from three repos (`swissisoform`, `tiap`, `coTISja`) into a unified 9-module architecture with rich domain objects and symmetric canonical/isoform annotation.

## Status

**All modules ported.** 312 tests, all passing. Pipeline orchestration done.
Next up: real test layer (small data slice), comparator extension, CLI, end-to-end test.

### What's Built — Foundation

| Layer | Files | What It Does |
|-------|-------|-------------|
| Domain model | `models.py`, `config.py` | `TIS`, `Gene`, `DifferentialRegion`, `ORFType`, `PipelineConfig` |
| Module protocols | `modules/base.py` | `ProteinModule`, `SiteModule` protocols with validation |
| I/O | `io/ribotish.py`, `io/gtf.py`, `io/parquet.py` | Ribo-TISH TSV reader (with AASeq), GTF transcript + CDS loaders, Parquet round-trip |
| Filtering | `filtering.py` | 5-step filtering (ported from coTISja) |
| Merging | `merging.py` | Cross-cell-line combine, canonical-vs-alternative pairing |
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
| 3. **Real test layer** | Small representative set from actual Ribo-TISH data | Next |
| 4. Comparator extension | Positional subset to diff region + scalar deltas | Pending |
| 5. CLI | `__main__.py` entry point | Pending |
| 6. End-to-end test | Full pipeline on real data | Pending |

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
# Fast tests only (Tier 1 — synthetic, < 5s)
pytest

# Single module
pytest tests/test_biophysics.py -v

# Integration (Tier 2 — real genome data)
pytest -m slow
```

## Code Style

- Linter/formatter: `ruff` (line length 100, Google docstrings)
- Type annotations required
- Tests: `pytest` with synthetic fixtures in `conftest.py`
- Module names are single words (no underscores) to avoid Parquet column prefix ambiguity
