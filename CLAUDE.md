# CLAUDE.md — SwissIsoform v2

## Project

**SwissIsoform v2** — Modular pipeline for annotating alternative protein isoforms from translation initiation sequencing (TI-seq). Consolidates code from three repos (`swissisoform`, `tiap`, `coTISja`) into a unified 13-module architecture with rich domain objects.

## Documentation

| Document | Path | Purpose |
|----------|------|---------|
| **Design spec** | `docs/superpowers/specs/2026-04-14-swissisoform-v2-orchestration-design.md` | Domain model, module contracts, wave execution strategy |
| Scientific overview | `docs/tis_projects_overview.md` | Biology, 2-paper structure, evidence scoring framework |
| Technical spec | `docs/tis_technical_spec.md` | Module specs 0-12, code provenance, implementation phases |
| Gap analysis | `docs/swissisoform_review.md` | Missing modules, scoring expansion, tool recommendations |
| V1 execution plan | `docs/execution_plan_v1.md` | Original sequential Claude Code execution plan (reference) |

## Source Repos (Read-Only Reference)

| Repo | Path | What It Contributes |
|------|------|---------------------|
| swissisoform v1 | `/lab/barcheese01/mdiberna/swissisoform/` | BED parsing, translation, mutations, genome handling |
| TIAP | `/lab/barcheese01/mdiberna/tiap/` | Modular annotation pipeline (14 modules), pipeline.py pattern |
| coTISja | `/lab/barcheese01/smaffa/coTISja/` | Ribo-TISH filtering, Kozak, expression normalization |

## Development

```bash
eval "$(conda shell.bash hook)" && conda activate swissisoform-v2
uv pip install -e ".[dev]"
```

## Architecture

- **13 modules** in `src/swissisoform/modules/`, each with `.run(tis_sites) -> tis_sites`
- **Domain objects** in `src/swissisoform/models.py` (`TranslationInitiationSite`, `Gene`, `VariantAnnotation`)
- **Module protocol** in `src/swissisoform/modules/base.py` — every module defines `MODULE_NAME`, `OUTPUT_COLUMNS`, `SCOPE`
- **Paired comparison** in `src/swissisoform/compare/paired.py` — shared canonical-vs-isoform delta logic

## Module Contract

Every module must:
1. Define `MODULE_NAME`, `OUTPUT_COLUMNS`, `SCOPE` as class attributes
2. Implement `run(tis_sites: list[TranslationInitiationSite]) -> list[TranslationInitiationSite]`
3. Write ONLY to `site.annotations[MODULE_NAME]` — never mutate other fields
4. Never drop sites (`len(output) == len(input)`)
5. Use `None` for values that can't be computed

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
