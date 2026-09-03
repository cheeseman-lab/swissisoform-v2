# Tag candidate review

Swept against distributions `v3`. 820 proposals → **89 candidates** + 264 percentile chips. Counts in the funnel are *proposals* (a metric is proposed in both directions); the chip table is per metric.

Fill in `decision` (keep / drop / reword / merge-into) and rewrite
`proposed_label` in `tag_candidates.csv`. Accepted rows become the registry.

## Funnel

| stage | n |
|---|---|
| proposed | 820 |
| dropped — unimodal — percentile cutoff only | 525 |
| dropped — sparse (fill < 0.8) | 105 |
| dropped — no CDLMPS category | 39 |
| dropped — redundant (Jaccard >= 0.8) | 20 |
| dropped — duplicate of another column | 16 |
| dropped — complementary direction of the same metric | 16 |
| dropped — constant | 8 |
| demoted to chip | 527 proposals → 264 metrics |
| **kept for review** | **89** |

## Per category

| category | candidates |
|---|---|
| C | 9 |
| D | 6 |
| L | 12 |
| M | 11 |
| P | 9 |
| S | 42 |

## Blocked

- **m1_pathogenic_variant_enrichment_isoform_plm_vep_constraint_enrichment** (isoform_plm_vep_constraint_enrichment) — stale metric: PLMVEPModule emits constraint_delta (unique minus shared mean logP, cutoff 0.0) but full_catalog carries only isoform_plm_vep_constraint_enrichment, whose values are strictly positive and span orders of magnitude — a ratio, not a difference. The genome-wide run predates the rename, so the frozen distribution describes a different quantity than the scorer computes and no cutoff may be derived from it. M1 rests on its gnomAD branch until the run is regenerated.

## Deliberately not swept

- **Categorical string columns.** The `cmp_*_changed` booleans already
  encode 'the compartment / targeting call differs'; a raw compartment
  string would need pairing logic that reproduces them.
- **Canonical-pane metrics.** They describe the gene, not the isoform's
  change. Note this excludes only the `canonical_` pane — the differential
  metrics live on `isoform_`, so a stricter pane filter would starve C and P.
