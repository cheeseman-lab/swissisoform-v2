# Planned refactor

*Written 2026-11-16 against swissisoform-v2 `feat/landing-table` @ e8d684e and
ribotish `main` @ 6269657.*

This doc proposes changes, so by the CLAUDE.md convention it would normally live
in `docs/plans/`. It sits at the repo root while it is under review.

1. **Now:** take ribotish's `starts.parquet` as the filtered input, and remove
   swissisoform's own Ribo-TISH filtering.
2. **Next:** modular, parallel annotation run by Nextflow. PepQuery becomes
   its own stage running during the GPU phase (2A), and each annotation
   category becomes a Nextflow module (2B).
3. **Outputs:** split annotation into three outputs: `proteins.parquet` (any
   protein), `annotations.parquet` (MCP) and `all_paired.parquet` (website).
4. **Later:** general code cleanup (section 4, short list only).

---

## 1. Take `starts.parquet` as the input

### 1.1 The new boundary

All filtering of Ribo-TISH output now lives in the ribotish repo
(`/lab/barcheese01/ating/ribotish`, `pipeline/integrate/starts/`). Its
`swissisoform` selection writes the finished list of start sites.
swissisoform stops doing any filtering and starts from that file.

```
ribotish repo                                         │  swissisoform-v2
──────────────────────────────────────────────────────┼─────────────────────────────
call (Ribo-TISH) → annotate_starts.py                 │
  → select_starts.py --selection swissisoform         │
  → <run>/starts/swissisoform/starts.parquet ─────────┼─► connector → assemble
       (+ references.parquet)                         │     → annotate → score → site
```

**What swissisoform does today and where it moves.** Each step of swissisoform's
current filtering has a ribotish equivalent:

| swissisoform today | ribotish now (`pipeline/integrate/starts/tisstarts/`) |
|---|---|
| `io/ribotish.py` reads `predict_all.txt` | `ribotish.py`, `load.py` read the Ribo-TISH `candidates` tables |
| `io/rnaseq.py` reads HTSeq counts and computes `NormTISCounts` | `rnaseq.py` (denominator), `annotate.py` |
| `filtering.py` 5-step filter | `filters.py`: counts, significance, tis_type, start_codon, initiation_efficiency, dedup30 |
| `io/canonical.py` imputes canonical starts | `canonical.py`, plus the `imputed` step |
| `pipeline.run_sample` drops uncanonical transcripts | the `start_incomplete` step |
| `sourceresolve/` long-read cascade (HeLa only) | `longread.py` filter + transcript choice, for every line with IsoQuant |
| `collapse_to_source` (one mRNA per TIS) | `transcripts.py`: one transcript per start, consensus across lines |
| `combine.py` cross-line union | `starts.parquet` holds all lines, one row per start per line |
| none | `cage.py`: CAGE support, used only in the transcript choice |

The `swissisoform` selection (`selections/swissisoform.yaml`) applies these
settings:
- Count floor `NormTISCounts ≥ 0.1`, `TISPvalue ≤ 0.01`, `RiboPvalue ≤ 0.01`, `FisherQvalue ≤ 0.05`.
- A 30 nt downstream dedup.
- Imputed canonical starts kept.
- Start-incomplete transcripts dropped.
- The long-read absence filter switched on.
- CAGE used only in the transcript choice.
- One transcript per start across lines.
- All conditions included.

The thresholds are the ones swissisoform's `FilterConfig` holds today. The long-read step and the CAGE step are new, and they now cover all lines, not just HeLa.

### 1.2 What `starts.parquet` looks like

Source: `pipeline/integrate/starts/README.md` and `tisstarts/selection.py:190-318`.

- **Long format:** one row per start per cell line (pooled mode), with a `line` column. Today swissisoform's combined parquet is **wide**, one row per `(Symbol, Tid, GenomePos, StartCodon)` with `present_{sample}` and `{sample}_{metric}` columns.
- **Columns** are the annotated-table columns (verified on `results/v50/clean/annotated_all_starts.parquet`):
  - Identity and sequence: `Condition`, `Mode`, `RowId`, `Gid`, `Tid`, `Symbol`, `GenomePos`, `StartCodon`, `Start`, `Stop`, `AALen`, `AASeq`, `Chromosome`, `Strand`.
  - Type: `TisType`, `RecatTISType`.
  - Statistics and counts: `TISCounts`, `TISPvalue`, `RiboPvalue`, `FisherQvalue`, `GeneRNASeqCounts`, `TotalRNASeqCounts`, `NormTISCounts`, `Imputed`.
  - Site keys: `InitSite`, `SiteKey`.
  - Long-read and CAGE columns.
  - Added by the selection: `line`, `Role`, and the long-read/CAGE evidence columns.
- **Canonical references are in a separate file.** When a picked transcript's Annotated start is not itself a kept start, it goes to `references.parquet`, not `starts.parquet`. swissisoform needs an Annotated row per transcript (`assembly._build_canonical_by_tid`) to build each TIS's canonical protein. So the connector must load **both files**.
- **8 lines, not 6.** The lines are HeLa, K562, U2OS, RPE1_Async, RPE1_Que, RPE1_Sen, **iPSC, CM**. `config.CELL_LINES` (`config.py:15`) lists six.
- **GENCODE v50, not v49.** ribotish's `pipeline/reference/reference.env` sets `RELEASE=50`. swissisoform's `references.py:28-30` points at v49. Transcript IDs, CDS coordinates and protein sequences have to come from the same release as the starts.

### 1.3 The change

**Add the connector**, `src/swissisoform/io/starts.py`: the single bridge from ribotish's `starts.parquet` into annotation. Nothing downstream reads `starts.parquet` directly; everything goes through `load_starts`.

- `load_starts(run_dir, lines=None) -> pd.DataFrame` reads `starts.parquet` + `references.parquet`.
- It checks the required columns and the GENCODE release recorded in ribotish's `provenance.json`.
- Pivoting to the wide frame keeps `assembly.py` and everything downstream of it unchanged, apart from the initiation-efficiency change below. That keeps this refactor to "replace the input", and lets a mismatch be traced to the input rather than to annotation. Teaching `assemble_genes` to read the long frame directly is possible later.

**What the connector does before annotation.** No sequence work is needed. `starts.parquet` already carries every column `assemble_genes` requires (`Symbol`, `Gid`, `Tid`, `TisType`, `GenomePos`, `StartCodon`, `Start`, `AALen`, `AASeq`) plus the per-line statistics. ribotish has already done the sequence processing swissisoform does today:
- `AASeq` has its stop stripped, so `AALen == len(AASeq)`.
- Annotated rows start with Met (`tisstarts/load.py`).
- Imputed canonicals take their sequence from `pc_translations.fa` (`tisstarts/canonical.py`).
- `TisType` keeps the raw Ribo-TISH strings that `contract.orf_type_from_ribotish` maps.

Assembly computes the rest itself from the start position and the reference: ORF type, the per-transcript canonical protein, the differential region, Kozak context, ORF exons, and Met on non-ATG isoforms. The connector only has to do two things:

1. **Pivot long → wide.** Produce one row per `(Tid, GenomePos, StartCodon)`:
   - shared fields once, checked to agree across lines (as `combine._verify_shared_fields` does today), failing loudly if they don't;
   - `present_{line}`;
   - `{line}_{TISCounts,NormTISCounts,TISPvalue,RiboPvalue,FisherQvalue,InitiationEfficiency}`.
2. **Add each transcript's canonical start as a full Annotated row** (`Tid`, `GenomePos`, `StartCodon`, `Start`, `AALen`, `AASeq`, `TisType = "Annotated"`). The row comes from `starts.parquet` when the canonical is itself a kept start, else from `references.parquet`. Without it, `assemble_genes` raises "no Annotated row" for that transcript (`assembly.py:792-801`).

Separately, swissisoform's reference files (GTF, genome, protein FASTA) must be GENCODE v50, because assembly looks transcripts and positions up in them (see 1.5 step 1).

**Initiation efficiency comes from ribotish only.** ribotish computes `InitiationEfficiency = TISCounts / GeneRNASeqCounts`, left empty when the gene has no RNA-seq reads (`tisstarts/annotate.py:100`). swissisoform recomputes the same ratio in `assembly._expression_from_row` (`assembly.py:513-518`). Remove swissisoform's copy:
- The connector carries ribotish's value through as `{line}_InitiationEfficiency`.
- `_expression_from_row` sets `CellLineExpression.initiation_efficiency` from that column.
- `{line}_GeneRNASeqCounts` is no longer read.

D2 and the canonical-vs-isoform initiation-efficiency comparison then use ribotish's definition directly, and the two repos cannot drift apart.
- **Where the input lives.** The ribotish run dir becomes a config value: a `STARTS_DIR` path in `references.py`, overridable with `--starts-dir`. It is never copied into `data/reference/`, so the version swissisoform reads is always the one ribotish recorded.

**Change these files:**

| File | Change |
|---|---|
| `runner.py` | `prepare()` calls `load_starts` instead of `load_combined` / `load_single_sample`. Delete `load_single_sample`, `load_combined`, `_write_unique_tis`, `_manifest_single_path`, the collapse + population block in `prepare` (`runner.py:690-733`), and the input constants `HELA_*`, `SAMPLE_MANIFEST`, `REPLICATE_MANIFEST`, `COMBINED_PARQUET`, `UNIQUE_TIS_PARQUET` (`runner.py:91-103`). Build the config once. `population.json` becomes a record of which `starts.parquet` was read (path, selection name, ribotish commit, row count) |
| `assembly.py` | `_expression_from_row` reads `{sample}_InitiationEfficiency` instead of computing `TISCounts / GeneRNASeqCounts`. Remove the single-sample branch of `_row_to_tis` (`assembly.py:609-634`), which only `--single-sample` uses and which also computes the ratio. Update the `assemble_genes` docstring, which still says it expects `run_sample` output |
| `RunSpec` | Remove `single_sample`, `rebuild_combined`, `skip_source_resolution`, `divergence_threshold`, `window_upstream`, `window_downstream`, `drop_unsupported_tis`. Add `starts_dir` |
| `scripts/run.py` | Remove the matching flags (`--single-sample`, `--rebuild-combined`, `--skip-source-resolution`, `--divergence-threshold`, `--window-*`, `--drop-unsupported-tis`). `--genes`, `--gene-list`, `--isoforms` and `--preset` subset the starts frame. `restrict_to_isoforms` already works on any frame with `Tid`/`GenomePos`/`StartCodon` |
| `references.py`, `config.py` | GENCODE paths to v50. `CELL_LINES` becomes whatever lines `starts.parquet` holds, not a constant. Remove `FilterConfig`, `SourceResolutionConfig`, and `build_config`'s source-resolution arguments |
| `export/presets.py` | Use the same `load_starts` + restriction as `run.py`. This also removes its separate single-sample HeLa path |
| `scripts/slurm/full_run/00_prepare.sbatch`, `split.py` | Read genes from `starts.parquet`, not `all_samples_combined.parquet`. Drop `--drop-unsupported-tis` from `00_prepare.sbatch` and `annotate_array.sbatch` |
| `setup/generef.py`, `scripts/export/export_alt_regions_bed.py`, `scripts/export/build_init_site_skeleton.py`, `figures/clustering_dims/principled_sampling/write_preset.py` | Read their gene / start list from `load_starts`, not `COMBINED_PARQUET` |
| `scripts/run_external_starts.py`, `io/external_tis.py` | See open question 3. Until that is decided it keeps working, since it still needs canonical imputation and `build_transcript_mrna` (see "Keep") |
| `presets/*.toml` | `[[isoforms]]` picks are keyed by `Tid` + `GenomePos` + `StartCodon`. Re-check that every pick still matches under v50 and the new transcript choice, and list any that drop out |

**Delete** (about 2.4k lines, plus their tests):

| Code | Lines |
|---|---:|
| `filtering.py` | 273 |
| `merging.py` (only its test imports it) | 114 |
| `sourceresolve/` (all 7 files) | 967 |
| `io/rnaseq.py` | 70 |
| `combine.py` except `dedupe_unique_init_sites` and the `init_site` helpers | ~300 of 443 |
| `io/ribotish.py` except `parse_genome_pos` | ~160 of 187 |
| `io/canonical.py` imputation (move `read_fasta` to `io/fasta.py` first) | ~250 of 306 |
| `pipeline.run_sample`, `normalize_canonical_initiator` | ~190 of 439 |
| `runner.py` loading/collapse code listed above | ~200 |
| `figures/mRNA_source_divergence/` | — |

Tests to delete: `test_filtering`, `test_merging`, `test_combine_filtered_samples`, `test_sourceresolve_*` (5 files), `test_ribotish`, `test_canonical_startsite`, `test_canonical_expression`. Edit `test_runner` and `test_population_sidecar`. Add `tests/test_starts_connector.py` on a small fixture cut from a real `starts.parquet`.

Inputs that leave swissisoform: `data/reference/*_TIS_predict_all.txt`, `ribotish_{sample,replicate}_manifest.csv`, `rnaseq_counts/`, `longread/`, and `data/output/filtered/`.

**Keep**, because other code uses these pieces:
- `combine.dedupe_unique_init_sites` and the init-site helpers, used by `build_init_site_skeleton.py`.
- `io/ribotish.parse_genome_pos`, used by `io/external_tis.py`.
- `io/canonical.read_fasta`, used by `plm/cli.py`, `plm/sae.py` and `structure/cli.py`.
- The imputation function and `sourceresolve/mrna.py`, **only while `external_tis` needs them** (see open question 3).
- `UpstreamReference` in `pipeline.py`, which loads the GTF, genome and exon skeletons that assembly still needs.

**Docs to update in the same change:** CLAUDE.md sections "Upstream", "Source-transcript resolution" and the "Execution Contract" mention of the combined catalog; README "Data flow".

### 1.4 What will differ, and how to check it

The start-site population changes by design:
- Long-read filtering across lines.
- A different transcript choice.
- Two more lines.
- v50.

So the `cheeseman13` snapshot **will not** match, and the normal byte-identical gate does not apply here. Check it in this order:

1. **Connector only.** Run the `swissisoform` selection restricted to the six original lines and with the long-read filter off. Pivot it with `load_starts`. Compare with today's `all_samples_combined.parquet` on the shared start keys. Differences should come only from v49 → v50 and the transcript choice, and each class of difference should be explained.
2. **Full input.** Run `cheeseman13` and `cheeseman50` from the real `starts.parquet`. Report which picked isoforms are still present, which changed transcript, and which are gone, and get sign-off.
3. **Then re-capture** the `cheeseman13` snapshot as the new baseline. From this point the byte-identical gate applies again to later refactors.

D1 (`min_cell_lines`, default 3) is counted over the lines present, so moving from 6 to 8 lines changes what D1 measures. Decide this before step 2 (open question 2).

### 1.5 Order of work

1. Move swissisoform's references to GENCODE v50: download the files and point `references.py` at them. On its own this changes no code path.
2. Add `io/starts.py` and its test. Wire `runner.prepare` and `run.py` to it behind `--starts-dir`, with the old path still present.
3. Validate (1.4 steps 1–2).
4. Switch the default input, then delete the filtering code, flags, config, inputs and tests listed above.
5. Update the campaign harness and the other `COMBINED_PARQUET` readers.
6. Update the docs, then re-capture the snapshot.

### 1.6 Open questions

1. **`references.parquet`.** Confirm that `starts.parquet` + `references.parquet` together give an Annotated row for every picked transcript, including the imputed canonicals. Otherwise canonical lookup falls back to the gene-level longest.
2. **iPSC and CM.** Include them in swissisoform (they are in the `swissisoform` selection), or select only the six in-house lines via `datasets.conditions`? This decides what D1 counts.
3. **External start sites** (`run_external_starts.py`). Should placing a non-Ribo-TISH start and imputing its canonical move into ribotish as well, so swissisoform has exactly one input path? Or does `external_tis` stay here, keeping `io/canonical` imputation and `sourceresolve/mrna.py` alive?
4. **Replicates.** `replicate_starts.parquet` is written but has no use here today. Does anything (D1 reproducibility) need it?
5. **Which ribotish run.** The `results/v50/` run on disk has only the older `clean/` layout and no `starts/swissisoform/starts.parquet` yet. Which run dir should swissisoform point at?

---

## 2. Modular, parallel annotation with Nextflow

**Goal.** Each annotation category is its own module, with its own inputs, outputs and resources. That gives three things:
- **Swap one tool, rerun only its part.** For example, replacing ESMFold2 reruns fold → structure → join and nothing else.
- **Nextflow runs and documents the whole pipeline:** it draws the dependency graph, schedules the Slurm jobs, and reports per-task timing.
- **Independent work runs concurrently,** including PepQuery, which is today's bottleneck.

Today annotation is one sequential Python process per job. The only parallelism is the campaign's 117 gene shards (`scripts/slurm/full_run/`), each a Slurm job running the whole pipeline in order.

### 2.0 Where the time goes today

Stages of one annotate shard (`swissiso_full_annot_11123510_110`, 38 genes, 78 TIS, 8 CPUs):

| Step | Time |
|---|---:|
| Load references, assemble | 2 min |
| DeepLoc (L1) | 10 min |
| **PepQuery (D3)**, two datasets back to back | **509 min** (238 + 271) |
| SignalP + TargetP (L2) | 3 min |
| InterProScan (S1) | 13 min |
| `build_pipeline` (per-gene clinical fetch + module setup) | 11 min |
| Annotation loop (`AnnotationPipeline.run`) | 99 min |
| Compare, score, tags, write | < 1 min |

Campaign stages (`00_prepare.sbatch`):
- **A:** prepare and split.
- **B:** GPU embed + fold arrays.
- **C:** annotate array, 117 shards, 12 at a time, `afterany` on **all** of B.
- **D:** merge into the campaign `all_paired.parquet`.

Timings:
- The July campaign spent **3.6 d in B and 3.4 d in C (≈ 7 d)**.
- In the October campaign the median shard took 7.1 h and the slowest 15 h.
- Summed over the 117 shards, PepQuery took **39,079 min (≈ 650 h)**, about 80% of Phase C.

**Dependencies that matter.** Only four modules read GPU output, all from cached arrays, on CPU:

| Module | Reads | Feeds |
|---|---|---|
| `plm_vep` | ESM-C cache: `llr` (per-residue logP of the wild-type residue) | M1 |
| `varianteffect` (ΔLLR part) | ESM-C cache: `aa_logprobs` | variant effect |
| `sae` | SAE cache (encoded from ESM-C `embedding_sae`) | S3 |
| `structure` | fold cache | P1, P2, P3 |

- **The ESM-C and fold dependencies are separate.** SAE and the VEP modules depend on the ESM-C embedding, not on folding.
- **Only `massspec` (D3) reads PepQuery's output.**
- **Everything else depends only on the assembled sites:**
  - DeepLoc, SignalP, TargetP and InterProScan, which all run on CPU today;
  - the clinical fetch;
  - every other module.

### 2A. PepQuery speed-ups

**Why it is slow.** Almost all of PepQuery's time is Step 5, the `-hc` competitive filter: every spectrum that matched one of our peptides is searched against the whole human proteome with unrestricted modifications. Shard 110: 205 peptides → 2,568 PSMs on one dataset; Step 5 ended at 238 of 238 minutes. Cost follows matched spectra, not peptide count:
- Across shards, time barely tracks peptide count (r = 0.32). 31 peptides took 69 min; 532 took 397 min.
- There is no load-once/query-many mode, and the reference FM-index is already built once (`data/reference/pepquery/db/*.fmi`).

So none of these changes cut the total compute. They cut **wall time**, by running the work concurrently and off the critical path.

**Speed-ups included:**

1. **PepQuery as its own campaign stage, running during the GPU phase.**
   - The Input step writes the campaign's isoform-unique peptides once (`collect_unique_peptides` over all assembled genes, deduplicated).
   - A `PEPQUERY` process runs per peptide chunk, on partition `24`. Per-user CPU limits are per partition (`24`: 300 CPUs, `20`: 120), so it doesn't compete with the other CPU modules on `20`.
   - It starts as soon as Input finishes, alongside fold and embed.
   - `massspec` reads its results through the existing `*/psm_rank.txt` glob in `_parse_pepquery_output`, instead of calling `precompute_pepquery`.
2. **Both datasets searched at once.** Each chunk × dataset (`Deep_29…PXD010154`, `GTEx_32…PXD016999`) is a separate task, replacing the back-to-back `for` loop at `massspec.py:775`. Per chunk, PepQuery takes as long as the longer dataset instead of the sum (shard 110: 509 → ~271 min).
3. **Explicit cores.** Each task passes `-cpu ${task.cpus}` instead of PepQuery's "all available" default, so tasks sharing a node don't oversubscribe it.
4. **Balanced chunks.** Size chunks by expected matched spectra, not just peptide count, since Step 5 cost follows PSMs. Per-shard PepQuery time ranges from 69 to 716 min today, and the slowest shard ends each wave. Choose chunk counts so the stage finishes before the GPU phase.
5. **Memory set per task, with retry.** `memory { 64.GB * task.attempt }` with `errorStrategy 'retry'`. `cheeseman50` peaked at 67 GB against a 64 GB request.
6. **Single runs** (`cheeseman13`, `cheeseman50`) use the same processes. With nothing else waiting on them, the two dataset searches and the rest of annotation run concurrently. Only `massspec` waits for PepQuery.
7. **Replaces the result cache.** The per-shard `data/cache/pepquery/*.json` cache violates the fresh-rerun contract. The stage is computed fresh for each campaign and handed to annotation, the same pattern as the GPU caches and the "fresh, peptide-sharded PepQuery stage" CLAUDE.md already names.

**Before building it:**
- Benchmark one search with PepQuery's **full** log kept. `massspec.py` keeps only the last 1,000 characters today; the full log gives per-step times and confirms Step 5 dominates.
- Time one search at `-cpu 4`, `8` and `16`, to measure how PepQuery scales with cores.
- Measure peak memory.

**Not part of this change:** whether `-hc` with `-varMod 2,5 -maxVar 3` is needed. Step 5 is the cost, so it matters for speed, but it changes D3's sensitivity and is a scientific decision.

### 2B. Annotation modules as a Nextflow workflow

**Nextflow** (25.10.4, installed at `/usr/local/bin/nextflow`; InterProScan already runs through it):
- Pipelines are written in its DSL2, a small language built on Groovy.
- Each step is a `process` that declares its inputs, outputs and resources. Its `script:` is a shell command, here a call to our Python CLI.
- Processes are connected by `channel`s inside a `workflow`, and Nextflow builds the dependency graph from those connections.
- The science stays in the `swissisoform` package; the `.nf` files only declare the steps, their resources, and what feeds what.

**Why Nextflow over Snakemake** (review item 8):
- **Processes, not file-name rules.** A Nextflow process declares its inputs and outputs and is wired to others by channels; a Snakemake rule is matched by output file names. A new consumer of an existing output, e.g. another module reading the ESM-C embeddings as `SAE` does, connects to the `EMBED` channel instead of hard-coding upstream file names and wildcards.
- **Data handling beyond files.** Nextflow has built-in operators for reading and splitting file contents and metadata, including FASTA (`splitFasta`), so chunking proteins for a tool is one operator. In Snakemake every step is a file on disk, which forces temporary files or ad hoc workarounds to get stream-like behaviour.

**Python changes first** (needed with or without Nextflow, and testable against the `cheeseman13` snapshot):
1. **A sites artifact.** Connector → assembly writes the assembled sites to `sites.parquet`, plus `proteins.fa` and the peptide list. `io/parquet.tis_to_dataframe` / `dataframe_to_tis` (`parquet.py:246, 294`) already do most of this.
2. **One CLI per module:** `swissiso annotate-module <name> --sites sites.parquet --out <name>.parquet`, keyed by `tis_id`, plus per-gene rows for canonical annotations.
3. **A join step.** It rebuilds `Gene`/TIS objects from the sites and the per-module outputs, then runs compare → score → tags → writes `all_paired.parquet`. Compare, score and tags need every module of a site, so they belong here.
4. **Task size:** a shard of genes, or a chunk of proteins for protein-level tools. Not single sites: every task costs a Slurm submission.

**Modules as subworkflows, by category:**

| Subworkflow | Processes (GPU marked) | Feeds |
|---|---|---|
| **Input** | connector → assembly → `sites.parquet`, `proteins.fa`, peptides; `core_identity`, `initiation_context`, `generef` (context) | everything |
| **Structure** | `FOLD` (GPU: ESMFold2, or Boltz/Chai) → `STRUCTURE` (confidence, TM-score, RMSD, SSE) | P1, P2, P3 |
| **Protein LM** | `EMBED` (GPU, ESM-C) → `SAE_ENCODE` (GPU) → `PLM_VEP`, `SAE`, `VEP_LLR` | M1 (constraint), S3, variant effect |
| **Detection (D)** | expression from the input (D1, D2); `PEPQUERY` (chunk × dataset, 2A) → `MASSSPEC` | D1, D2, D3 |
| **Localization (L)** | `DEEPLOC`, `SIGNALP`, `TARGETP` | L1, L2 |
| **Sequence / domains (S)** | `INTERPROSCAN`, `BIOPHYSICS`, `MOTIFS` | S1, S2 |
| **Conservation (C)** | `CONSERVATION` (PhyloP/PhastCons), `CONSERVATION_FRAME` (HAL) | C1, C2, C3 |
| **Mutation (M)** | `CLINICAL_FETCH` → `VARIANT_INTERSECTION`; `VARIANTEFFECT` (also needs Protein LM's `aa_logprobs`) | M1, M2 |
| **Finish** | `JOIN` per shard (compare → score → tags) → `MERGE` | all 16 criteria |

Cross-category dependencies are expressed as graph edges:
- M1 needs `PLM_VEP` and `VARIANT_INTERSECTION`.
- `VARIANTEFFECT` needs the clinical hits and the ESM-C `aa_logprobs`.
- The comparator's Scope-A step reruns `biophysics` on the differential region inside `JOIN`.

**Graph:**

```
starts.parquet ─► INPUT (connector + assembly) ─► sites.parquet, proteins.fa, peptides
        ├─► FOLD (GPU, by chunk) ───────────► STRUCTURE ──────────────┐
        ├─► EMBED → SAE_ENCODE (GPU) ────────► PLM_VEP, SAE, VEP_LLR ─┤
        ├─► PEPQUERY (CPU 24, chunk × dataset) ─► MASSSPEC ───────────┤
        ├─► DEEPLOC, SIGNALP, TARGETP ─────────► L modules ───────────┤
        ├─► INTERPROSCAN, BIOPHYSICS, MOTIFS ──► S modules ───────────┼─► JOIN (per shard):
        ├─► CONSERVATION, CONSERVATION_FRAME ──► C modules ───────────┤    compare → score → tags
        └─► CLINICAL_FETCH ─► VARIANT_INTERSECTION ─► M modules ──────┘    ─► MERGE ─► all_paired.parquet
```

`nextflow run main.nf -preview -with-dag dag.html` (or `.mmd` for Mermaid) renders this graph without running anything. Each run also writes `-with-report`, `-with-timeline` and `-with-trace`.

**How each optimization maps onto Nextflow:**

| Optimization | Before (today / hand-built) | In Nextflow |
|---|---|---|
| PepQuery during the GPU phase (2A.1) | new Slurm array between A and C | `PEPQUERY` depends only on Input, so it starts with `FOLD`/`EMBED` |
| Both PepQuery datasets at once (2A.2) | `Popen` both, split `-cpu` | chunk × dataset tasks run concurrently by construction |
| Annotate waits only for what it needs | `afterany` on all of B | `MASSSPEC` waits for `PEPQUERY`, `STRUCTURE` for `FOLD`, the Protein LM modules for `EMBED`; nothing else waits |
| CPU modules run during the GPU phase | a C1/C2 two-pass annotate split with a persisted intermediate | **not needed**: CPU modules depend only on Input, and `JOIN` is the finishing pass |
| DeepLoc / SignalP / TargetP / InterProScan concurrently | concurrent subprocesses | separate processes |
| Annotation fanned across genes | Python worker pool | gene shards are channel items (`.collate(100)`), so each module × shard is a task |
| More GPU partitions for fold | resubmit by hand | `queue 'nvidia-A6000-20,nvidia-L40S-24,nvidia-RTX6000-24'` (Slurm takes the first with room) + `maxForks` under the per-user limits. The A6000 limit is 2 running jobs, so fold queued about 37 h behind embed in July. Check GPU memory for 1,024-aa proteins |
| Atomic outputs | temp file + `os.replace` | tasks write in their own `work/` folder; outputs are published only on success |
| Chunking / sharding | `split.py` + `awk` on its manifest | `.splitFasta(by: 1500)`, `.collate(100)` |
| Merge | `merge.py` + `merge.sbatch` | `MERGE` over `.collect()` of the `JOIN` outputs; it writes the scoring/population sidecars `merge.py` misses today |
| Per-module timing | not logged | `-with-trace` / `-with-report` |
| Shard resume (contract carve-out) | `all_paired.parquet` as the done-marker | `-resume` + versioned module outputs (below) |

**Replaced, not built:** `00_prepare.sbatch`'s chain, `split.py`, `annotate_array.sbatch`, `merge.sbatch`, and the hand-built two-pass annotate split.

**Rerunning one module: needs a contract decision.** Reusing unchanged module outputs is a CPU result cache. CLAUDE.md's Execution Contract allows that today only for GPU precomputes and intra-campaign shard resume. Nextflow's `-resume` alone is not safe:
- Its task hash covers the process script and input files, **not the `swissisoform` code the script imports**, so a module code change could silently reuse a stale result.
- The cache lives in `work/`, which is large and is lost if deleted.

Proposed rule:
- Module outputs are written to `data/output/<campaign>/modules/<module>/`.
- Each is keyed by **(sites-artifact hash, module code version, module parameters)**. The code version is the git SHA or a per-module version constant.
- That version is passed into each process as an explicit input, so Nextflow's hash changes with the code.
- An output is reused only when all three keys match. The Execution Contract in CLAUDE.md is amended to say so.

**Running it:** the Nextflow driver runs as a small long-lived CPU Slurm job, not on the head node. Each process names its conda env; we already have `swissisoform-v2`, `-fold`, `-plm`, `-deeploc` and `-signalp`.

**Costs:**
- DSL2 files to maintain alongside the Python.
- `work/` disk usage, especially fold outputs, cleaned per campaign.
- The per-module CLI/join refactor, which is most of the effort.

### 2C. Fix regardless

GPU cache writes are not atomic. `np.savez_compressed` and the fold writes go straight to the final path (`plm/embed.py:193`, `plm/sae.py:102`, `structure/fold.py:92-101`), and `manifest.tsv` is appended without a lock (`plm/embed.py:196-210`). Write to a temp file and `os.replace`. This matters for anything that writes these caches outside Nextflow.

### 2D. Order of work

1. §1: the connector.
2. PepQuery benchmark: full log, `-cpu` scaling, memory.
3. Sites artifact + per-module CLIs + `JOIN`. Gate: the `cheeseman13` snapshot is unchanged when the modules run sequentially through the CLIs.
4. Decide the module-output reuse rule; amend the CLAUDE.md contract.
5. `main.nf`: Input, the category subworkflows, `PEPQUERY` (2A), `JOIN`, `MERGE`, with Slurm profiles. Gate: `cheeseman13` via Nextflow matches the snapshot.
6. Split the outputs (§3): `proteins.parquet`, `annotations.parquet` (MCP), `all_paired.parquet`. Gate: `all_paired.parquet` on `cheeseman13` is unchanged.
7. Run `cheeseman50`, then a campaign. Read `-with-trace` for per-module times; then add fold partitions and tune chunk sizes.

### 2E. Expected result

| | GPU phase | PepQuery | After the GPU phase | Campaign |
|---|---|---|---|---:|
| Today | 3.6 d | inside each shard | 3.4 d (annotate) | ≈ 7 d |
| Nextflow + 2A | 3.6 d | concurrent with GPU | Protein LM / structure modules + `JOIN` (hours) | ≈ 4 d |
| + more GPU partitions for fold | < 3.6 d | concurrent with GPU | same | lower still |

A single run (`cheeseman50`, about 10.3 h today) gets about 4.5 h from concurrent dataset searches plus running the rest of annotation alongside PepQuery. These estimates assume PepQuery's stage fits inside the GPU phase, which the 2A benchmark will confirm.

---

## 3. Three outputs

Annotation produces three outputs, each built from the one before it. Today there is only `all_paired.parquet`, which mixes module annotations with everything derived from them.

In `cheeseman_13gene`'s 478 columns, the derived layers are:
- `cmp_*` (139): comparator deltas, categorical changes, hits in the differential region.
- `diff_*` (22): Scope-A, biophysics rerun on the differential region.
- `isoform_scoring_*`: scoring.
- `isoform_tags_*`: tags.

`io/parquet.paired_tis_dataframe` appends them in that order (`parquet.py:565-568`), so they separate cleanly.

```
proteins.fa ─► protein-level modules ─► 1. proteins.parquet          (any protein)
                                              │
sites.parquet + site-level modules + Scope-A ─┴─► 2. annotations.parquet  (MCP output)
                                                        │
                                   COMPARE → SCORE → TAGS
                                                        ▼
                                                 3. all_paired.parquet    (website, unchanged)
```

### 3.1 Output 1: `proteins.parquet`, protein annotation of any sequence

**Purpose.** Annotate **any** protein sequence, not only alternative isoforms.
- Input is a FASTA.
- Output has one row per unique protein, keyed by `protein_hash` (`hashing.py`), plus the sequence and length.
- Entry point: `swissiso annotate-proteins --fasta x.fa --out proteins.parquet`, and the Nextflow subworkflow it maps to.

**Contents:** the modules whose output depends on the sequence alone, i.e. those with `annotate(protein)` and no other context.

| Module | What it reports | Compute |
|---|---|---|
| `biophysics` (`modules/biophysics.py:357`) | pI, GRAVY, instability, disorder, … | CPU |
| `motifs` (`modules/motifs.py:89`) | positional motif hits | CPU |
| `localization` (DeepLoc, `localization.py:256`) | localization class + probabilities | CPU |
| `signalp` (`signalp.py:235`), `targetp` (`targetp.py:228`) | signal / targeting peptides | CPU |
| `interproscan` (`interproscan.py:494`) | domain hits | CPU |
| ESM-C (from the embed cache) | per-residue `llr`, with whole-protein summaries | GPU |
| SAE (from the SAE cache) | per-protein feature activations | GPU |
| fold (from the fold cache) | per-residue pLDDT, pTM, the structure path | GPU |

**Not included,** because each needs more than the sequence:
- `clinical` (needs the gene and its variants);
- `massspec` (uniqueness needs a canonical to subtract);
- every site module: they need a start site, genomic coordinates or a differential region (conservation, conservation_frame, variant_intersection, varianteffect, plm_vep's unique/shared constraint, structure comparison, core_identity, initiation_context).

In the isoform pipeline this table is computed **once per unique protein**. It replaces running the protein modules on the canonical protein once per gene and then again on each isoform, and the canonical and isoform columns of output 2 read from it.

### 3.2 Output 2: `annotations.parquet`, the MCP output

**Purpose.** Expose every annotation, for both the canonical protein and each isoform, before any comparison, so an MCP server (or anyone else) can query the underlying values rather than the website's derived ones.

**Contents.** One row per TIS, keyed by `tis_id`:
- identity and site columns: gene, transcript, ORF type, start, differential-region coordinates, ORF exons;
- `expr_*` (per-line expression);
- `canonical_<module>_*` and `isoform_<module>_*` for every module: protein-level from output 1, site-level from the site modules;
- `generef_*`;
- **Scope-A:** the differential-region biophysics (see the review note).

**No `cmp_*`, no scoring, no tags.**

**Requirements:**
- **Column dictionary** shipped beside it (`annotations.columns.tsv` or a JSON schema). For each column: module, side (canonical / isoform / differential region), type, meaning and units. Generated from each module's `OUTPUT_COLUMNS`, not written by hand.
- **Missing values documented.** `None` means "could not evaluate", never "absent". Each module's `status` (`not_run`, `no_cache`, `no_hits`, `no_alignment`, …) is kept in the table.
- **Provenance sidecar** (`annotations.manifest.json`): input `starts.parquet` (path, selection, ribotish commit), GENCODE release, swissisoform code version, per-module versions and parameters.
- **Positional hits** (variants, motifs, domains, peptides) stay as nested lists in the row, and are also exposed as long tables (one row per hit, keyed by `tis_id`, like `variants_long.parquet`), which are easier to query.
- **Canonical duplication:** the gene's canonical annotations repeat on each of its TIS rows, as in `all_paired.parquet` today. Alternatively, an MCP client can join to output 1 by the canonical's `protein_hash`.

> **For review: Scope-A is the only comparator output included.**
> Scope-A (`_scope_a_enrichment`, `comparator.py:247`) does two things:
> 1. It re-runs `biophysics` on `diff_region.sequence` to get the **unique-region** values, paired with the **shared-region** values.
> 2. It computes `ratio = unique / shared` and `enriched = ratio > 1` (`PairedComparison.compare_scalar_regions`, `paired.py:49`).
>
> The plan:
> - Move step 1 into annotation, so the `biophysics` module reports unique-region and shared-region values itself.
> - Put those values in output 2.
> - Leave the ratio and `enriched` in the comparator, so they appear only in output 3.
>
> Every other comparator output stays out of output 2: scalar deltas, categorical changes, hits in the differential region.
>
> **To decide:**
> - Include Scope-A's raw unique/shared values only (as proposed), or also the ratio/`enriched` columns?
> - The ratio is not sign-safe for GRAVY, disorder and instability (`metrics.SIGNED_REGION_PROPERTIES`), which argues for leaving it out of an output meant for others to interpret.

### 3.3 Output 3: `all_paired.parquet`, unchanged

The same file the website, exports and tags read today, with the same columns. It is built as output 2 → COMPARE → SCORE → TAGS:
- COMPARE adds `cmp_*`, plus Scope-A's ratio and `enriched` on the `diff_*` columns.
- SCORE adds `isoform_scoring_*`.
- TAGS adds `isoform_tags_*`.

Gate: the snapshot of `all_paired.parquet` on `cheeseman13` is identical before and after this split.

### 3.4 What this changes in the code and in the Nextflow graph

- **`biophysics`** gains a differential-region mode, so Scope-A step 1 runs as annotation. The comparator keeps only the ratio.
- **The comparator** reads output 2. Either it rebuilds `Gene`/TIS objects from the table (`io/parquet.dataframe_to_tis`, `parquet.py:294`, plus per-gene canonical annotations), or it is rewritten to work on the table as tags already do. Compare + score take under a second per shard, so running them once over the merged campaign table is cheap; check memory (the merge job uses 256 GB today).
- **Nextflow (§2B):** `JOIN` splits in two:
  - `MERGE_ANNOTATIONS` joins the module outputs per TIS and concatenates shards → output 2 (+ dictionary, manifest, long hit tables);
  - `COMPARE_SCORE_TAGS` → output 3.

  Protein-level processes write output 1 directly.
- **Rerunning only part:** changing a comparison, a scoring threshold, a tag cutoff or a website-only feature reruns only `COMPARE_SCORE_TAGS` from output 2. The annotation modules don't rerun.

---

## 4. Later: code cleanup (not part of this change)

These came out of the codebase survey. Each is its own change, and each is gated on the `cheeseman13` snapshot staying identical.

- **Bugs to fix first:**
  - `--skip-modules sae` is rejected because `sae` is missing from `references.ALL_SITE_MODULES`.
  - `full_run/split.py` deletes `chunks/` and `shards/` before its stale-output check, and `merge.py` writes no `scoring_config.json` or `population.json`. Both scripts are replaced by Nextflow in §2B, so fix them only if campaigns run on the bash harness before then.
- **Paths:**
  - One resolver (`paths.py`) replacing about 25 `Path(__file__).parents[N]` and the cwd-relative `Path("data/…")` in `split.py`/`merge.py`.
  - Split `data/output/` into `runs/`, `campaigns/`, `research/`.
- **One CLI:**
  - Add `[project.scripts]` with a `swissiso` command (`run`, `gpu`, `campaign`, `site`, `export`, `setup`, `tags`).
  - Move logic out of `scripts/` (`build_orf_index.py`, `export_static_html.py`) into the package. `split.py`/`merge.py` are replaced by §2B.
  - The sbatch files that remain outside Nextflow (single GPU jobs, site build) become resource headers plus one CLI call, with env selection in one shared script.
- **One definition per concept:**
  - The 16 criterion IDs, ORF-type sets and column names in `contract.py`. They are currently restated in `tags/`, `site/evidence.py`, `modules/scoring.py` and `website/data.py`.
  - `protein_hash` from `hashing.py` everywhere.
  - Shared helpers for missing values, FASTA and `conda run`.
- **Package layout:**
  - Group annotators by the §2B categories (mutation, conservation, detection, structure, protein LM, localization, sequence/domains) instead of scattering them across `modules/`, `evidence/` and standalone subpackages, so the package mirrors the Nextflow subworkflows.
  - `evidence/` keeps only scorers.
  - Remove the `io → sourceresolve → combine → plm` cycle. It mostly disappears with section 1.
- **Site:**
  - Split `site/llm.py` (3.4k lines) and `site/evidence.py` (2.6k lines) along their concerns.
  - Move the LLM prompts from `scripts/site/prompts/` into package data.
  - Install the package into the website image instead of `prepare_deploy.sh` copying a hand-kept file list. The copy currently in the image is already stale.
- **Research code:** keep `judge/` and `figures/` apart from the pipeline, with no `sys.path` hacks.
- **Tests:** mirror `src/`, add a `realdata` marker, no tests that load scripts by path.
- **Hygiene:**
  - Delete `modules/conservation_homology.py` (dormant), the unused `PipelineConfig`/`ScoringConfig` fields, `scripts/_backfill_cols.py` and `scripts/_run_cheeseman13.sh`.
  - Decide whether `docs/architecture/` and `docs/plans/` should be tracked; `.gitignore` excludes them today.
