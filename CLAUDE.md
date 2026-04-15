# CLAUDE.md — SwissIsoform v2

## Project

**SwissIsoform v2** — Modular pipeline for annotating alternative protein isoforms from translation initiation sequencing (TI-seq). Consolidates code from three repos (`swissisoform`, `tiap`, `coTISja`) into a unified 13-module architecture with rich domain objects.

## Status

**Wave 1 (Foundation) complete.** 160 tests, all passing. Ready for Wave 2 (8 parallel module ports).

### What's Built (Wave 1)

| Layer | Files | What It Does |
|-------|-------|-------------|
| Domain model | `models.py`, `config.py`, `modules/base.py` | `TranslationInitiationSite`, `Gene`, `ORFType` enum with Ribo-TISH mapping, `PipelineConfig`, `ModuleProtocol` |
| I/O | `io/ribotish.py`, `io/gtf.py`, `io/parquet.py` | Read raw Ribo-TISH predict_all.txt TSV, parse GTF annotations, serialize to/from Parquet |
| Filtering | `filtering.py` | 5-step filtering ported from coTISja: transcript selection, count/significance thresholds, distance dedup |
| Merging | `merging.py` | Cross-cell-line combine, canonical-vs-alternative pairing with fold change |
| Comparison | `compare/paired.py` | Shared delta logic (categorical, set, scalar, structure comparisons) |
| Module 1 | `modules/core_identity.py` | ORF classification, protein lengths, in-frame check, truncation warning |
| Module 2 | `modules/initiation_context.py` | Kozak Hamming distance (full/major/partial), GC content |

### What's Done (Wave 2)

All module protocols implemented. Symmetric canonical/isoform architecture in place.

| Module | Protocol | Interface | Status |
|--------|----------|-----------|--------|
| `biophysics.py` | ProteinModule | `annotate(protein) -> dict` (scalar) | Done |
| `motifs.py` | ProteinModule | `annotate(protein) -> dict` (positional hits) | Done |
| `localization.py` | ProteinModule | `annotate_by_key(key) -> dict` (lookup) | Done |
| `core_identity.py` | SiteModule | `annotate_site(site) -> dict` | Done |
| `initiation_context.py` | SiteModule | `annotate_site(site) -> dict` | Done |
| `generef.py` | Gene-level | unchanged (outside comparison pipeline) | Done |

### Roadmap

| Step | What | Status |
|------|------|--------|
| 1. **Wiring layer** | `pipeline.py` — Gene-level orchestration, runs ProteinModules on canonical + isoform | Next |
| 2. **Harder modules** | `massspec.py` (PepQuery2), `conservation.py` (BLAST/PhyloP), `clinical.py` (gnomAD/ClinVar) | Pending |
| 3. **Real test layer** | Small representative set from actual Ribo-TISH data | Pending |
| 4. **Comparator extension** | Positional subset to diff region + scalar deltas | Pending |
| 5. **CLI** | `__main__.py` entry point | Pending |
| 6. **End-to-end test** | Full pipeline on real data | Pending |

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
- **Domain objects** in `models.py`: `TranslationInitiationSite`, `Gene`, `VariantAnnotation`
- **Module protocol** in `modules/base.py`: `MODULE_NAME`, `OUTPUT_COLUMNS`, `SCOPE`, `.run(tis_sites) -> tis_sites`
- **Paired comparison** in `compare/paired.py`: shared canonical-vs-isoform delta logic
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

Every module must:
1. Define `MODULE_NAME`, `OUTPUT_COLUMNS`, `SCOPE` as class attributes
2. Implement `run(tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]`
3. Write ONLY to `site.annotations[MODULE_NAME]` — never mutate other fields
4. Never drop sites (`len(output) == len(input)`)
5. Use `None` for values that can't be computed
6. Positional annotations: store as list of dicts with `pos` (and optionally `end`) keys
7. Scalar annotations: store as plain values (float, str, bool)

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
