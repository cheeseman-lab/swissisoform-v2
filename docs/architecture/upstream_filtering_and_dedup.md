# Upstream Filtering, Merging & Deduplication — Current State

> **Scope:** This document describes the pipeline **as it currently exists on `main`**
> (commit baseline `7c7b865`). It documents *what is*, not *what is planned*. No
> proposed changes appear here — those belong in `docs/plans/`.
> **Last verified:** 2026-06-16 against the source files cited inline (`file:line`).

This is the path that turns raw Ribo-TISH `predict_all.txt` files (one per cell
line) into the deduplicated tables that feed the annotation pipeline:
`all_samples_combined.parquet` and `unique_tis_deduped.parquet`.

---

## 1. How to reproduce

The whole filter → merge → dedup path is driven by `scripts/run.py` (a thin
front-end over `src/swissisoform/runner.py`). The combined + deduped tables are a
**cached side effect** of any run; they are (re)built when missing or when forced:

```bash
eval "$(conda shell.bash hook)" && conda activate swissisoform-v2

# Rebuild the combined + deduped tables from the 6 per-sample predict files:
python scripts/run.py --all --rebuild-combined --run-name full_catalog

# Any run reuses the cache if present (only rebuilds if absent or --rebuild-combined):
python scripts/run.py --preset cheeseman13
```

Cache location (`runner.py:81-84`):
- `data/output/filtered/all_samples_combined.parquet`
- `data/output/filtered/unique_tis_deduped.parquet`
- `data/output/filtered/{sample}_TIS_filtered.csv` (one per cell line)

The init-site table is **not** part of a normal run — it is a separate export
(`scripts/export/build_init_site_skeleton.py`, see §5.4).

---

## 2. Pipeline overview

```
 6 × predict_all.txt    RNA-seq HTSeq counts    GENCODE GTF + genome.fa + pc_translations.fa
        │                       │                          │
        └──────────── run_sample()  (per cell line, independent)  ── pipeline.py:43
                          ├─ load_ribotish_predictions  ── io/ribotish.py:40
                          ├─ recategorize_tis_type      ── io/ribotish.py:91
                          ├─ sum_replicate_counts → RPM ── io/rnaseq.py:36 + filtering.py:43
                          ├─ filter_tis  (5-step)       ── filtering.py:131   ★ THE FILTER
                          ├─ impute_missing_canonical_starts ── io/canonical.py:160
                          └─ drop uncanonical transcripts ── pipeline.py:147
                                  │
                          {sample}_TIS_filtered.csv  (×6)
                                  │
          combine_filtered_samples()  (UNION across cell lines) ── combine.py:62   ★ THE MERGE
                                  │
                  all_samples_combined.parquet   ──►  assemble_genes → annotation pipeline
                                  │
          dedupe_unique_proteins()  (collapse by protein_hash) ── combine.py:169   ★ THE DEDUP
                                  │
                  unique_tis_deduped.parquet    ──►  structure folding / PLM embedding set
```

Orchestration lives in `runner.load_combined` (`runner.py:119-175`): it loops the
sample manifest, calls `run_sample` per cell line, writes each
`{sample}_TIS_filtered.csv`, then `combine_filtered_samples`, then
`_write_unique_tis` (`runner.py:109`).

---

## 3. Stage-by-stage detail

### 3.1 Inputs

| Input | Path | Notes |
|---|---|---|
| Ribo-TISH predictions | `data/reference/{sample}_TIS_predict_all.txt` | 21-col TSV per cell line (cols in §5.1) |
| RNA-seq counts | `data/reference/rnaseq_counts/*_htseqcount.txt` | HTSeq `Gid\tcount`; 2 replicates/sample |
| Sample manifest | `data/reference/ribotish_sample_manifest.csv` | `sample, predict_file, filtered_file, dropped_file` |
| Replicate manifest | `data/reference/ribotish_replicate_manifest.csv` | `sample, replicate, condition, rnaseq_count_file` |
| GTF | `data/reference/gencode.v49.primary_assembly.annotation.gtf` | MANE/TSL, CDS, start_codon, exons |
| Genome FASTA | `data/reference/Gencode_v49_GRCh38.primary_assembly.genome.fa` | start-codon trinucleotides (imputation) |
| Protein FASTA | `data/reference/gencode.v49.pc_translations.fa` | canonical AASeq/AALen (imputation) |

Cell lines (`runner.py`, `ALL_CELL_LINES`): `HeLa, K562, U2OS, RPE1_Async, RPE1_Que, RPE1_Sen`.

### 3.2 Load + annotate — `load_ribotish_predictions` (`io/ribotish.py:40`)

1. Read the 21-col TSV.
2. **Strip the trailing stop `*` from `AASeq`** and recompute `AALen = len(AASeq)`
   so native rows agree with imputed canonicals (`io/ribotish.py:69-72`).
3. Add `Chromosome`, `Strand`, `Locus` from `GenomePos` (`_annotate_tis_locus`,
   `io/ribotish.py:136`). `Locus` = range start on `+`, range end on `−`.
4. Left-join GTF transcript annotations on `(Gid, Tid)` → adds `MANE_Select`,
   `transcript_support_level`, `gene_type`, `transcript_type`
   (`_merge_gtf_annotations`, `io/ribotish.py:162`).

### 3.3 Recategorize TIS type — `recategorize_tis_type` (`io/ribotish.py:91`)

Compresses Ribo-TISH's compound `TisType` into `RecatTISType` ∈
{`Annotated`, `Truncated`, `Extended`, `uORF`, `Other`} via substring match
(`_simplify_tis_type`, `io/ribotish.py:123`): `"Annotated"→Annotated`,
`"Truncated"→Truncated`, `"5'UTR"→uORF`, `"Extended"→Extended`, else `Other`.

### 3.4 RNA-seq normalization (RPM)

- `sum_replicate_counts` (`io/rnaseq.py:36`): per-gene HTSeq counts summed across
  replicates, QC rows (`__no_feature`, …) dropped → `GeneRNASeqCounts`.
- `TotalRNASeqCounts` = the grand total of mapped reads (sum of the gene-count
  Series; `pipeline.py:96-100`).
- `normalize_tis_counts` (`filtering.py:43`):
  **`NormTISCounts = TISCounts / TotalRNASeqCounts × 1e6`** (true RPM against
  total mapped reads).

### 3.5 The filter — `filter_tis` (`filtering.py:131`)

Runs **per cell line**, called from `run_sample` with **`exempt_annotated=False`**
(`pipeline.py:122` — imputation replaces the role exemption played, so the
`FilterConfig` default of `True` is overridden here). Each row that fails gets a
pipe-separated `DropReason`; survivors have `DropReason is null`.

| Step | Code | Rule | Drop tag |
|---|---|---|---|
| 1. Reference transcripts | `identify_reference_transcripts` `filtering.py:70` | `MANE_Select == True` **OR** `transcript_support_level ∈ {1,2,3}`. (Called with permissive p/q = 1.0, min counts = 0, so it is purely the MANE/TSL gate.) | — |
| 2. Non-reference | `filtering.py:208` | TIS whose `Tid` is not a reference transcript | `NotReferenceTranscript` |
| 3. Low count | `filtering.py:223` | `NormTISCounts < min_normalized_counts` (0.1) | `LowReadcounts` |
| 4. Not significant | `filtering.py:229` | NOT (`TISPvalue ≤ 0.01` AND `RiboPvalue ≤ 0.01` AND `FisherQvalue ≤ 0.05`) | `NotSignificant` |
| 5. Distance dedup | `filtering.py:249` | Within each transcript, sort by count desc, **prefer Annotated**, keep the chosen TIS, drop any other TIS within `Start … Start+30 nt` downstream | `UpstreamTIS` |

Defaults come from `FilterConfig` (`config.py:13`): `transcript_support_levels=["1","2","3"]`,
`min_normalized_counts=0.1`, `tis_enrichment_max_p=0.01`, `frame_test_max_p=0.01`,
`combined_test_max_q=0.05`, `tis_distance_buffer=30`, `exempt_annotated=True` (default; **overridden to `False`** in `run_sample`).

`run_sample` returns `(filtered_df, dropped_df)`; only `filtered_df` flows downstream.

### 3.6 Canonical imputation — `impute_missing_canonical_starts` (`io/canonical.py:160`)

For every transcript present in the filtered set that has **no `Annotated` row**,
synthesize one so every transcript has a canonical to compare against:
- `GenomePos`/`Start` from GTF CDS coords + 5'UTR length (no UTR ⇒ `Start=0`),
- `StartCodon` from the genomic start-codon trinucleotide,
- `AASeq`/`AALen` from the GENCODE protein product.
- Static fields set (`_IMPUTED_STATIC`, `io/canonical.py:151`):
  `TisType="Annotated"`, `RecatTISType="Annotated"`, `TISGroup=0`, `TISCounts=0`,
  `NormTISCounts=0`, plus **`Imputed=True`**.
- Candidates missing `GenomePos`/`StartCodon`/`AALen` are dropped.

Native (non-imputed) filtered rows carry `Imputed=False` (`pipeline.py:125`).

### 3.7 Uncanonical drop (`pipeline.py:147-167`)

After imputation, any transcript *still* without an `Annotated` row (GENCODE
`cds_start_NF` / incomplete CDS — canonical start unknown by definition) is
dropped entirely. The result is `final_df` = filtered + imputed − uncanonical,
written to `{sample}_TIS_filtered.csv`.

### 3.8 Cross-sample merge — `combine_filtered_samples` (`combine.py:62`)

A **UNION** across the 6 per-sample frames — a TIS surviving in **any one** cell
line enters the combined table. There is **no cross-condition AND / reproducibility
gate at this layer.**

- **Dedup key** (`DEDUP_KEY`, `combine.py:32`): `(Symbol, Tid, GenomePos, StartCodon)`.
- **Shared fields** copied once from the first sample (`SHARED_FIELDS`, `combine.py:36`):
  `Symbol, Gid, Tid, GenomePos, StartCodon, TisType, RecatTISType, AASeq, AALen, Start`.
  `_verify_shared_fields` (`combine.py:149`) **raises** if `AASeq/AALen/TisType/RecatTISType/Start/Gid`
  disagree across samples for one key (genome-level invariant guard).
- **Per-sample metrics** pivoted to wide `{sample}_{metric}` columns
  (`PER_SAMPLE_METRICS`, `combine.py:50`): `TISCounts, NormTISCounts, TISPvalue,
  RiboPvalue, FisherQvalue, Imputed, GeneRNASeqCounts, TotalRNASeqCounts`.
- **Presence**: `present_{sample}` = that sample's `TISCounts` is non-null;
  `samples` = list of calling samples; `n_samples` = count (`combine.py:129-138`).

> **`n_samples` is recorded but not enforced.** It only feeds the soft E4 evidence
> criterion in scoring (`modules/scoring.py:172`), never a hard drop. `min_cell_lines`
> is a scoring threshold, not a filter.

### 3.9 Protein dedup — `dedupe_unique_proteins` (`combine.py:169`)

Collapses the combined per-TIS table to **one row per unique protein**, keyed by
`protein_hash` = sha1 of the stop-stripped, upper-cased `AASeq`
(`plm.embed.protein_hash` — the same key the structure/PLM on-disk caches use).
This is the genome-wide "what to fold / embed" set, so each distinct sequence is
folded/embedded once.

- **Representative** TIS per protein = sort by `(protein_hash, n_samples↓, total TISCounts↓, Tid↑)`, keep first (`combine.py:209-213`).
- Provenance preserved: every contributing transcript / start site / gene
  (`n_transcripts`, `n_init_sites`, `n_genes`, `all_genes`, `all_transcripts`, `orf_types_all`).

---

## 4. Key semantics & caveats (current behavior)

1. **Per-condition filter, then union.** Significance/count thresholds apply
   *within* each cell line; the merge is a union. No TIS is required to appear in
   ≥k cell lines.
2. **`exempt_annotated=False` in production.** `run_sample` overrides the
   `FilterConfig` default; Annotated rows are *not* auto-exempted from
   count/significance drops — instead imputation re-adds canonicals.
3. **`min_cell_lines` is not a filter** — only a soft evidence score (E4).
4. **Dropped rows are not persisted.** `run_sample` returns `dropped_df`, but
   `runner.load_combined` writes only `final_df` to `{sample}_TIS_filtered.csv`
   (`runner.py:163`). The manifest's `dropped_file` column is currently unused by
   the run path.
5. **Two different "unique" collapses exist.** `dedupe_unique_proteins`
   (by `protein_hash`) feeds the pipeline; `dedupe_unique_init_sites`
   (by genomic `init_site`) is a separate DNA-model export (§5.4) — **not** a
   pipeline input.

---

## 5. Generated files and their columns

### 5.1 Input reference — `{sample}_TIS_predict_all.txt` (21 cols, TSV)

`Gid, Tid, Symbol, GeneType, GenomePos, StartCodon, Start, Stop, TisType,
TISGroup, TISCounts, TISPvalue, RiboPvalue, RiboPStatus, FisherPvalue, TISQvalue,
FrameQvalue, FisherQvalue, AALen, Seq, AASeq`.

### 5.2 Per-sample — `{sample}_TIS_filtered.csv` (×6)

Schema derived from code (`run_sample` → `final_df`; regenerated each rebuild, not
currently on disk). Columns = the predict_all columns **plus** the columns added
upstream: `Chromosome, Strand, Locus` (locus annotation); `MANE_Select,
transcript_support_level, gene_type, transcript_type` (GTF merge); `RecatTISType`;
`GeneRNASeqCounts, TotalRNASeqCounts, NormTISCounts` (RNA-seq + RPM); `Sample`;
`GenomeStart, DropReason` (filter — `DropReason` is null for all kept rows);
`Imputed` (False for native, True for imputed canonicals).

### 5.3 `all_samples_combined.parquet` — 66 cols, ~105,524 rows  ← **main pipeline input**

One row per unique `(Symbol, Tid, GenomePos, StartCodon)`.

| Group | Columns | Type |
|---|---|---|
| Identity (shared) | `Symbol, Gid, Tid, GenomePos, StartCodon` | object |
| ORF / sequence (shared) | `TisType, RecatTISType, AASeq` (obj); `AALen, Start` (float) | mixed |
| Per-sample metrics (×6 cell lines, `{sample}_…`) | `_TISCounts, _NormTISCounts, _TISPvalue, _RiboPvalue, _FisherQvalue, _GeneRNASeqCounts, _TotalRNASeqCounts` (float); `_Imputed` (object bool) | mixed |
| Presence | `present_{sample}` (bool, ×6); `samples` (list); `n_samples` (int) | mixed |

`{sample}` ∈ {HeLa, K562, RPE1_Async, RPE1_Que, RPE1_Sen, U2OS}.
Per-row meaning: `*_TISCounts` raw Ribo-TISH count; `*_NormTISCounts` RPM;
`*_TISPvalue`/`*_RiboPvalue`/`*_FisherQvalue` the three filter stats;
`*_GeneRNASeqCounts`/`*_TotalRNASeqCounts` the RNA-seq denominators;
`*_Imputed` whether that sample's row is a synthetic canonical.

### 5.4 `unique_tis_deduped.parquet` — 25 cols, ~74,924 rows  ← **fold/embed set**

One row per `protein_hash`.

| Column | Type | Meaning |
|---|---|---|
| `protein_hash` | object | sha1 of stop-stripped AASeq (cache key) |
| `gene`, `representative_transcript` | object | representative TIS's gene / Tid |
| `init_site` | object | `chrom:gstart:strand:codon` of representative |
| `start_codon`, `genome_pos`, `orf_type` | object | representative codon / ORF span / RecatTISType |
| `is_canonical` | bool | representative `TisType` starts with "Annotated" |
| `length_aa` | int | protein length |
| `n_cell_lines`, `max_cell_lines` | int | reproducibility (rep / max across contributors) |
| `n_source_rows`, `n_transcripts`, `n_init_sites`, `n_genes` | int | dedup provenance counts |
| `all_genes`, `all_transcripts`, `orf_types_all` | object | comma-joined provenance |
| `sequence` | object | stop-stripped AA sequence |
| `max_norm_{sample}` (×6) | float | peak RPM per cell line |

### 5.5 `init_site_skeleton.parquet` — 47 cols, ~48,018 rows  ← **separate DNA-model export, NOT a pipeline input**

Built by `scripts/export/build_init_site_skeleton.py` →
`dedupe_unique_init_sites` (`combine.py:260`). One row per genomic `init_site`
(`chrom:gstart:strand:codon`).

| Group | Columns |
|---|---|
| Site anchor | `init_site, chrom, gstart, strand, start_codon, genome_pos` |
| Representative identity | `gene, representative_transcript, protein_hash, orf_type, is_canonical, is_imputed` |
| Lengths | `length_aa, length_aa_min, length_aa_max` |
| Reproducibility | `n_cell_lines, max_cell_lines, n_source_rows` |
| Multi-mapping | `n_transcripts, n_proteins, n_genes, n_genome_pos, all_genes, all_transcripts, all_protein_hashes, orf_types_all` |
| Best significance | `min_tis_pvalue, min_ribo_pvalue, min_fisher_qvalue` |
| Per-condition (×6) | `max_norm_{sample}` (peak RPM), `present_{sample}` (bool), `gene_rnaseq_{sample}` (gene RNA-seq counts) |

---

## 6. Source-file index

| Concern | File:line |
|---|---|
| Per-sample orchestration | `src/swissisoform/pipeline.py:43` (`run_sample`), `:179` (`UpstreamReference`) |
| Read + locus + GTF merge | `src/swissisoform/io/ribotish.py:40` |
| TIS-type recategorization | `src/swissisoform/io/ribotish.py:91` |
| RNA-seq counts | `src/swissisoform/io/rnaseq.py:36` (`sum_replicate_counts`), `:48` (`load_sample_manifest`) |
| RPM + 5-step filter | `src/swissisoform/filtering.py:43` (norm), `:70` (reference tx), `:131` (`filter_tis`) |
| Canonical imputation | `src/swissisoform/io/canonical.py:160`, `:151` (`_IMPUTED_STATIC`) |
| Cross-sample merge | `src/swissisoform/combine.py:62` (`combine_filtered_samples`) |
| Protein dedup | `src/swissisoform/combine.py:169` (`dedupe_unique_proteins`) |
| Init-site dedup (export) | `src/swissisoform/combine.py:260` (`dedupe_unique_init_sites`) |
| Filter config / defaults | `src/swissisoform/config.py:13` (`FilterConfig`) |
| Run orchestration + caching | `src/swissisoform/runner.py:119` (`load_combined`), `:109` (`_write_unique_tis`) |
| CLI front-end | `scripts/run.py`; init-site export `scripts/export/build_init_site_skeleton.py` |
