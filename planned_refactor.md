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
3. **Outputs:** every module writes standardized positional layers; from them come
   three outputs: the layers (MCP), `prot_annot.parquet` (any protein) and
   `all_paired.parquet` (website) (section 3).
4. **MCP:** a read-only `mcp/` package over the layers, the catalog and `all_paired.parquet`, with a small generic tool set (section 4).
5. **Cleanup:** moving the LLM code into its own package (required), then a short list of later cleanup (section 5).
6. **Modal:** run the GPU stage (and PepQuery, if the cluster can't fit it in a day) on Modal, to reach a full run in about a day (section 6).

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
   - The `MASSSPEC` process runs per peptide chunk × dataset, on partition `24`. Per-user CPU limits are per partition (`24`: 300 CPUs, `20`: 120), so it doesn't compete with the other CPU modules on `20`.
   - It starts as soon as Input finishes, alongside `FOLD` and `ESMC`.
   - Each task searches its chunk and parses its own `psm_rank.txt` (`_parse_pepquery_output`) into the peptide Feature layer, keeping the dataset. Parsing takes seconds against hours of search, so it is not a separate process.
   - Which peptides to search is an Input decision, not module logic: the isoform-unique peptides by default, every tryptic peptide for a plain FASTA catalog.
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
- **Processes, not file-name rules.** A Nextflow process declares its inputs and outputs and is wired to others by channels; a Snakemake rule is matched by output file names. A new consumer of an existing output, e.g. another module reading the ESM-C embeddings as `SAE` does, connects to the `ESMC` output instead of hard-coding upstream file names and wildcards.
- **Data handling beyond files.** Nextflow has built-in operators for reading and splitting file contents and metadata, including FASTA (`splitFasta`), so chunking proteins for a tool is one operator. In Snakemake every step is a file on disk, which forces temporary files or ad hoc workarounds to get stream-like behaviour.

**Python changes first** (needed with or without Nextflow, and testable against the `cheeseman13` snapshot):
1. **The catalog** (§3.1): connector → assembly writes `orfs.parquet`, `proteins.parquet`, `proteins.fa` and the peptide list.
2. **One CLI per module:** `swissiso module run <name> --catalog <dir> --out <dir>/layers/<name>/`, writing positional layers (§3.3).
3. **A compare step.** It reads the catalog and the layers, then runs compare → score → tags → writes `all_paired.parquet`. Compare, score and tags need every module of a site, so they belong here.
4. **Task size:** a shard of genes, or a chunk of proteins for protein-level tools. Not single sites: every task costs a Slurm submission.

**One process per module.** Each of the 16 modules in §3.4 is one Nextflow process with the module's name, running `swissiso module run <name>`; its declared outputs are exactly that module's layers (§3.3). Chunks of proteins or genes are its tasks. Four more processes are pipeline steps, not modules: `INPUT`, `COMPARE`, `SCORE_TAGS`, `MERGE`.

| Subworkflow | Process = module | Runs on | Needs | Layers |
|---|---|---|---|---|
| **Input** | `INPUT` (not a module) | CPU | `starts.parquet` | catalog, `proteins.fa`, peptide list |
| **Structure** | `FOLD` | GPU (ESMFold2, or Boltz/Chai) | `INPUT` | Track pLDDT, Pair PAE, Scalar pTM |
| | `SSE` | CPU | `FOLD` | Feature helix/strand |
| **Protein LM** | `ESMC` | GPU | `INPUT` | Track log P (wt + 20 aa) |
| | `SAE` | GPU | `ESMC` | Track (sparse) |
| **Detection** | `MASSSPEC` | CPU `24`, chunk × dataset (2A) | `INPUT` | Feature peptides |
| **Localization** | `LOCALIZATION` (DeepLoc) | CPU | `INPUT` | Scalar |
| | `SIGNALP`, `TARGETP` | CPU | `INPUT` | Feature |
| **Sequence** | `DOMAINS` (InterProScan) | CPU | `INPUT` | Feature |
| | `BIOPHYSICS` | CPU | `INPUT` | Track, Feature, Scalar |
| | `MOTIFS` | CPU | `INPUT` | Feature |
| **Conservation** | `CONSERVATION` | CPU | `INPUT` | Track PhyloP, PhastCons over the catalog's exons |
| | `CONSERVATIONFRAME` | CPU | `INPUT` | Feature, Scalar per species |
| **Mutation** | `VARIANTS` (ClinVar, gnomAD, COSMIC) | CPU | `INPUT` | Variant |
| | `ALPHAMISSENSE` | CPU | `INPUT` | Variant (every substitution over the exons) |
| **Initiation** | `INITIATION` | CPU | `INPUT` | Feature Kozak window |
| **Finish** | `COMPARE` → `SCORE_TAGS` (not modules) | CPU, per shard | all layers + catalog | comparison and tag rows per ORF |
| | `MERGE` (not a module) | CPU | protein-keyed layers + `SCORE_TAGS` | `prot_annot.parquet`, `all_paired.parquet` |

- **The same graph serves every run.** On a FASTA catalog (no ORFs) `COMPARE` and `SCORE_TAGS` run on zero rows, and `MERGE` writes only `prot_annot.parquet`.
- **The only module-to-module edges** are `ESMC → SAE` and `FOLD → SSE`, each a declared input.
- **No annotation process depends on another category.** Everything that combines categories (M1 constraint with variants, ΔLLR from `aa_logprobs` and the variant alleles, region biophysics) is in `COMPARE` (§3.5).

**Graph:**

```
starts.parquet ─► INPUT ─► catalog, proteins.fa, peptides
        ├─► FOLD (GPU) ─► SSE ─────────────────────────┐
        ├─► ESMC (GPU) ─► SAE (GPU) ───────────────────┤
        ├─► MASSSPEC (CPU 24, chunk × dataset) ────────┤
        ├─► LOCALIZATION, SIGNALP, TARGETP ────────────┤
        ├─► DOMAINS, BIOPHYSICS, MOTIFS ───────────────┼──────────────────────────────────────────┐
        ├─► CONSERVATION, CONSERVATIONFRAME ───────────┤                                          ▼
        ├─► VARIANTS, ALPHAMISSENSE ───────────────────┼─► COMPARE (per shard) ─► SCORE_TAGS ─► MERGE ─► prot_annot.parquet
        └─► INITIATION ────────────────────────────────┘                                          └─► all_paired.parquet
```

`nextflow run main.nf -preview -with-dag dag.html` (or `.mmd` for Mermaid) renders this graph without running anything. Each run also writes `-with-report`, `-with-timeline` and `-with-trace`.

**How each optimization maps onto Nextflow:**

| Optimization | Before (today / hand-built) | In Nextflow |
|---|---|---|
| PepQuery during the GPU phase (2A.1) | new Slurm array between A and C | `MASSSPEC` depends only on Input, so it starts with `FOLD`/`ESMC` |
| Both PepQuery datasets at once (2A.2) | `Popen` both, split `-cpu` | chunk × dataset tasks run concurrently by construction |
| Annotate waits only for what it needs | `afterany` on all of B | `SSE` waits for `FOLD`, `SAE` for `ESMC`, `COMPARE` for every layer; nothing else waits |
| CPU modules run during the GPU phase | a C1/C2 two-pass annotate split with a persisted intermediate | **not needed**: CPU modules depend only on Input, and `COMPARE` is the finishing pass |
| DeepLoc / SignalP / TargetP / InterProScan concurrently | concurrent subprocesses | separate processes |
| Annotation fanned across genes | Python worker pool | gene shards are channel items (`.collate(100)`), so each module × shard is a task |
| More GPU partitions for fold | resubmit by hand | `queue 'nvidia-A6000-20,nvidia-L40S-24,nvidia-RTX6000-24'` (Slurm takes the first with room) + `maxForks` under the per-user limits. The A6000 limit is 2 running jobs, so fold queued about 37 h behind embed in July. Check GPU memory for 1,024-aa proteins |
| Atomic outputs | temp file + `os.replace` | tasks write in their own `work/` folder; outputs are published only on success |
| Chunking / sharding | `split.py` + `awk` on its manifest | `.splitFasta(by: 1500)`, `.collate(100)` |
| Merge | `merge.py` + `merge.sbatch` | `MERGE` over `.collect()` of the `COMPARE` outputs; it writes the scoring/population sidecars `merge.py` misses today |
| Per-module timing | not logged | `-with-trace` / `-with-report` |
| Shard resume (contract carve-out) | `all_paired.parquet` as the done-marker | `-resume` within the campaign; `--rerun <module>` to force one (below) |

**Replaced, not built:** `00_prepare.sbatch`'s chain, `split.py`, `annotate_array.sbatch`, `merge.sbatch`, and the hand-built two-pass annotate split.

**Running and rerunning one module** (review item 4: modules run alone; change detection is not a priority):
- **Alone:** `swissiso module run <name>` outside Nextflow, or `nextflow run main.nf -entry <MODULE>` (plus `INPUT` if there is no catalog yet).
- **Forced:** `--rerun <module>` is fed into that process as an input, so its task hash changes and `-resume` reruns it and its downstream (e.g. `FOLD` → `SSE` → `COMPARE` → `SCORE_TAGS` → `MERGE`) while reusing every other module. Deleting `layers/<module>/` and resuming does the same.
- **No versioned reuse.** `-resume` is used only within one campaign. Changing one module's code (e.g. swapping ESMFold2 for Boltz-2) is handled with `--rerun <module>` in the same campaign; any other code or config change means a fresh campaign name, as today. Nextflow's hash does not see the imported `swissisoform` code, so the person changing a module names it in `--rerun`; nothing detects it automatically.
- **Contract:** CLAUDE.md's intra-campaign carve-out is extended from shard resume to module outputs. Nothing else in the Execution Contract changes.

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
3. Catalog + `layers/` + per-module CLIs + `COMPARE` (§3.7). Gate: the `cheeseman13` snapshot is unchanged when the modules run sequentially through the CLIs.
4. Extend CLAUDE.md's intra-campaign carve-out to module outputs; add `--rerun <module>`.
5. `main.nf`: `INPUT`, the 16 module processes in their subworkflows, `COMPARE`, `SCORE_TAGS`, `MERGE`, with Slurm profiles. Gate: `cheeseman13` via Nextflow matches the snapshot.
6. Rewrite the modules that must emit new positional data (§3.7 step 4). Gate: `all_paired.parquet` on `cheeseman13` is unchanged.
7. Run `cheeseman50`, then a campaign. Read `-with-trace` for per-module times; then add fold partitions and tune chunk sizes.

### 2E. Expected result

| | GPU phase | PepQuery | After the GPU phase | Campaign |
|---|---|---|---|---:|
| Today | 3.6 d | inside each shard | 3.4 d (annotate) | ≈ 7 d |
| Nextflow + 2A | 3.6 d | concurrent with GPU | Protein LM / structure modules + `COMPARE` (hours) | ≈ 4 d |
| + more GPU partitions for fold | < 3.6 d | concurrent with GPU | same | lower still |

A single run (`cheeseman50`, about 10.3 h today) gets about 4.5 h from concurrent dataset searches plus running the rest of annotation alongside PepQuery. These estimates assume PepQuery's stage fits inside the GPU phase, which the 2A benchmark will confirm.

---

## 3. Module outputs: positional layers, then aggregation, then compare

This section follows review items 1–3 and mat10d's proposed reader contract on PR #60:
- Every module writes **positional output** in one of five shared shapes, keyed by protein, genome or site.
- **Modules know nothing about isoforms:** no canonical vs isoform, no unique vs shared, no differential region. The test: annotate 500 arbitrary proteins, or a novel ORF with no baseline, and every module runs; there is just nothing to compare.
- **Aggregation** (a summary over a range) and **comparison** (isoform vs baseline) are separate steps that read the positional output.

```
INPUT ─► catalog ─► 16 module processes ─► layer files (Track / Feature / Variant / Pair / Scalar)
                                                │  read back through the readers as dataclasses
                     ┌──────────────────────────┴──────────────┐
                     ▼                                         ▼
            1. layers (MCP)           COMPARE → SCORE_TAGS ─► MERGE ─► 2. prot_annot.parquet
                                                                   └─► 3. all_paired.parquet
```

### 3.1 Key spaces and the catalog

A key is what a result is attached to and looked up by:

| Key | Identifies | Used by |
|---|---|---|
| `protein` (`protein_hash`, `hashing.py`) | one amino-acid sequence; identical sequences are computed once | sequence tools, ESM-C, SAE, structure, mass spec |
| `genome` (chrom, 0-based half-open, strand) | a stretch of DNA, whichever ORF crosses it | conservation, conservation frame, variants, AlphaMissense, Kozak |
| `site` (`tis_id`) | one start event | per-line Ribo-TISH measurements |

**The catalog** is the output of the §1 connector plus assembly (the Input step in §2B):
- `orfs.parquet`: one row per ORF, with these columns:
  - `tis_id`, gene, transcript, ORF type;
  - `protein_hash`, and the baseline's `protein_hash` (null when there is none);
  - the `orf_exons` mapping from protein positions to the genome;
  - the per-line `starts.parquet` measurements;
  - the `generef` context.
- `proteins.parquet`: one row per `protein_hash`, with the sequence and length.

A FASTA alone gives a valid catalog with no sites and no baselines. That is how any protein is annotated.

### 3.2 Reader contract

These are the types every module's output is read as. They follow AlphaGenome's `TrackData`, extended to protein coordinates and to outputs that are not tracks. The code goes in `src/swissisoform/layers/` (types, readers, writers):

```python
@dataclass(frozen=True)
class Interval:
    space: Literal["protein", "genome"]
    key: str                    # protein_hash, or chromosome
    start: int                  # 0-based, half-open
    end: int
    strand: Literal["+", "-"] | None = None    # genome only

@dataclass(frozen=True)
class TrackData:                # one value per position
    values: np.ndarray          # (n_positions, n_tracks); NaN where missing
    tracks: pd.DataFrame        # per track: name, module, units, status
    interval: Interval

@dataclass(frozen=True)
class FeatureData:              # intervals
    features: pd.DataFrame      # start, end, type, name, score, attrs
    module: str
    status: str
    interval: Interval

@dataclass(frozen=True)
class VariantData:              # point events with content
    variants: pd.DataFrame      # pos, ref, alt, source, variant_id, attrs
    module: str
    status: str
    interval: Interval

@dataclass(frozen=True)
class PairData:                 # residue x residue
    values: np.ndarray          # (n, n)
    name: str
    module: str
    status: str
    interval: Interval

@dataclass(frozen=True)
class ScalarData:               # whole-protein or whole-site values
    values: pd.Series           # name -> value
    module: str
    status: str
    key: str                    # protein_hash or site id
```

- **`status`** (`ok`, `not_run`, `no_data`, …) stays distinct from empty or zero.
- **Shared methods:** the position types share `slice(start, end)`, and `TrackData` also has `select(names)` and `summarize(how)`.

**Storage** sits behind the readers, so it can change without touching compare, the website or the MCP:
- **Feature, Variant and Scalar:** one Parquet file per module and shape, with the shape's columns plus `key` and `status`, sorted by `key` so that a read filters to one protein or region.
- **Dense tracks and PAE: NumPy `.npz` / `.npy`** (decided; no Zarr). The existing per-protein GPU caches stay as they are, behind the reader:
  - `plm_esmc/6B/<hash>.npz`: `llr`, `aa_logprobs`;
  - `sae_esmc/6B/<hash>.npz`: `idx`, `val`;
  - `structure/esmfold2/<hash>/`: `confidence.json`, `pae.npy`, `metrics.json`, `model.cif`.

  They are already keyed by `protein_hash` and load in one read. New dense tracks use the same form: `biophysics` one `.npz` per `protein_hash`; `conservation` one `.npz` per chromosome holding the bases under catalog exons, with a flat start/end + offsets index (PR #55's layout). Writes are atomic (temp file + `os.replace`, §2C).
- **Conservation:** the `CONSERVATION` process extracts per-base PhyloP and PhastCons over the catalog's exons from the BigWigs into a Track layer, so it is a layer file like every other module.
- **Anything shipped to the browser** uses PR #55's form: flat little-endian arrays plus offsets.

### 3.3 Module contract

This replaces the `ProteinModule` / `SiteModule` protocols and the `run(tis_sites)` wrappers in CLAUDE.md's Module Contract.

Each module:
1. **Is one Nextflow process** of the same name (§2B), whose outputs are exactly its layers.
2. **Declares** `MODULE_NAME`, `KEY_SPACE`, and `LAYERS`, giving each layer's name, shape, columns or tracks, units and meaning. The MCP's `list_layers()` and the column dictionary are generated from this.
3. **Runs alone:** `swissiso module run <name> --catalog <dir> --out <dir>/layers/<name>/` writes its layers and nothing else. It reads only the catalog and declared inputs (e.g. `varianteffect` ← ESM-C `aa_logprobs`).
4. **Never sees an isoform pairing:** it takes no canonical protein, differential region or ORF-type argument.
5. **Writes a `status` for every key,** including keys it could not evaluate.
6. **Exposes plain functions where needed.** Some values can't be averaged over a range (pI, for example), so the calculation must be callable on any sequence, and compare can recompute it on a sub-sequence.

### 3.4 The 16 modules, checked against the code

Each module is the Nextflow process of the same name in §2B. This is mat10d's table, checked against the modules on `feat/landing-table`. **Correction** marks a row that differs from the proposal.

| Module = process | Key | Layers | Today |
|---|---|---|---|
| `fold` / `FOLD` (GPU) | protein | Track pLDDT; Pair PAE; Scalar pTM, structure path | **On disk** (`structure/esmfold2/<hash>/`). `pae.npy` exists for 9,863 of 10,865 entries; the rest need a refold |
| `sse` / `SSE` (after `FOLD`) | protein | Feature: helix and strand elements, mean pLDDT | Computed from the CIF (`sse.py:42`), then tagged unique/shared. The tag moves to compare |
| `esmc` / `ESMC` (GPU) | protein | Track: log P(wild type); 20 tracks of log P per amino acid | **On disk** (`llr`, `aa_logprobs`); only region means are emitted. Embeddings stay a cache input for `sae`, not a layer |
| `sae` / `SAE` (GPU, after `ESMC`) | protein | Track (sparse): top-64 feature activations per residue | **On disk** (`idx`, `val`, both (L,64)) |
| `massspec` / `MASSSPEC` | protein | Feature: peptides with PSM count, hyperscore, p-value, dataset | **Dataset is lost:** aggregated across datasets (`massspec.py:535-601`). Uniqueness against the canonical (`:348-373`) moves to compare |
| `localization` / `LOCALIZATION` | protein | Scalar: DeepLoc class probabilities, membrane type, signal labels | Exists. **Correction:** DeepLoc2 gives no signal positions, so no Feature |
| `signalp` / `SIGNALP` | protein | Feature: signal peptide (0 → cleavage site), class, probability | Cleavage site kept as a raw string (`signalp.py:176-191`); parse to a position |
| `targetp` / `TARGETP` | protein | Feature: transit peptide (0 → cleavage site), class, probability | Same as `signalp` (`targetp.py:183-188`) |
| `domains` / `DOMAINS` | protein | Feature: InterProScan database, accession, name, e-value, InterPro id | Exists (`interproscan.py:404-465`) |
| `biophysics` / `BIOPHYSICS` | protein | Track: hydropathy, charge, disorder propensity, windowed entropy. Feature: low-complexity regions, homopolymer runs, prion-like regions. Scalar: pI, instability, aromaticity, length, LLPS, RG/FG density | **Scalars only** (18). Window entropy and the low-complexity mask are computed then discarded (`biophysics.py:197-218`); "prion-like" is a QNGSY composition fraction, not a region detector |
| `motifs` / `MOTIFS` | protein | Feature: motif hits | Exists (`hits`); no score |
| `conservation` / `CONSERVATION` | genome | Track: PhyloP (241 mammals), PhastCons (100 vertebrates), per base over the catalog's exons | The BigWigs are per base; the module reads only means |
| `conservationframe` / `CONSERVATIONFRAME` | genome | Feature: per-species frameshift, premature stop, start-codon loss, with position. Scalar: per-species identity, intact | **Not kept.** Per-species results (`frame.py:44-60`) are aggregated to clades; positions never recorded (`frame.py:210-225`) |
| `variants` / `VARIANTS` | genome | Variant: ClinVar, gnomAD, COSMIC alleles, frequency, significance | Exists as clinical hits. Consequence in an ORF's frame moves to compare |
| `alphamissense` / `ALPHAMISSENSE` | genome | Variant: pathogenicity per (chrom, pos, ref, alt), every substitution over the exons | **Correction:** genome-keyed, canonical transcript (`alphamissense.py:79-142`), not protein. Today looked up only for clinical variants |
| `initiation` / `INITIATION` | genome | Feature: Kozak window, match scores, GC | Exists as scalars (`initiation_context.py:110`); its canonical twin moves to compare |

**Not modules:**
- **Per-line Ribo-TISH values** (TIS counts, normalized counts, p-values, initiation efficiency), site-keyed: **Correction:** they come from `starts.parquet` through the connector (§1) and are catalog columns.
- **`generef`** (UniProt id, function, location, keywords): gene context, stored as catalog columns.
- **`core_identity`, `plm_vep`, `variant_intersection`, `varianteffect`:** canonical-vs-isoform logic, which moves into `COMPARE` (§3.5).

How far each module is from the target:
- **Reader only** (the data exists): `fold`, `esmc`, `sae`, `conservation`, `motifs`, `domains`, `localization`.
- **Small rewrite:** `signalp`, `targetp` (parse the cleavage site); `sse`, `initiation` (drop region tags); `variants`, `alphamissense` (look up over the exons, not only clinical hits).
- **Larger rewrite:** `biophysics` (per-residue tracks and intervals), `conservationframe` (per-species positions), `massspec` (keep the dataset, move uniqueness out).

### 3.5 Computed in compare, not emitted by modules

Anything defined by a difference, obtained by slicing the layers over ranges from the catalog:
- unique vs shared means and enrichment ratios (PhyloP, ESM-C constraint, pLDDT);
- region biophysics: the plain biophysics functions recomputed on the unique sub-sequence. This was "Scope-A" (`comparator.py:247`); it stays in compare and does not move into annotation;
- pLDDT deltas, TM-score and RMSD between two structures, PAE block means;
- SAE gained and lost features;
- variant counts and depletion in a region, variant consequence and ΔLLR in a given ORF's frame;
- peptides unique to the isoform (its digest minus the baseline's);
- secondary-structure elements tagged unique / shared / spans;
- core identity, the Kozak canonical twin, and per-clade frame integrity over the unique region.

To support this, the comparator functions and the aggregator take an **arbitrary range** as an argument, not only the precomputed differential region. The MCP needs this too (review item 7).

### 3.6 The three outputs

All three are generated from the same layer files, through the same readers.

| | Output | Generated by | For |
|---|---|---|---|
| 1 | **Layers,** `layers/<module>/` | The 16 module processes, written directly in their shapes' storage (§3.2) | The MCP and the website's tracks: positions, read on any range |
| 2 | **`prot_annot.parquet`** | `MERGE`, step 1: one row per `protein_hash` | Annotating any protein: one flat row per protein |
| 3 | **`all_paired.parquet`** | `COMPARE` → `SCORE_TAGS` → `MERGE`, step 2: one row per ORF | The website, exports and tags: isoform vs baseline |

1. **Layers.** These are the module output itself, not a copy. Readers return dataclasses for one key and range, which is all the MCP (`mcp/`, review item 7) and the website's track views (conservation, pLDDT; review item 9) need. Each layer is written with `status`, and a column dictionary is generated from the modules' `LAYERS`.
2. **`prot_annot.parquet`.** `MERGE` first summarizes every protein-keyed layer into one row per `protein_hash`, using the shared `aggregate(layer, interval, how)` function:
   - the Scalar layers as they are (pI, pTM, DeepLoc probabilities, …);
   - `protein_hash`, the sequence, and the input ID (the FASTA header, or `tis_id`s and gene for an isoform run), so a user goes from their own protein name to the hash;
   - whole-protein summaries of the Track and Feature layers (mean pLDDT, mean log P(wt), number of domains, peptides found, signal peptide present, …);
   - each module's `status`.

   It is a convenience table. Per-position data is read from the layers by `protein_hash` (`read_features("domains", protein_hash)`, `read_track("esmc", protein_hash, 0, L)`); the hash comes from this table or from `hashing.protein_hash(seq)`, since it depends only on the sequence (review item 2). The MCP calls the same `aggregate` on any range.
3. **`all_paired.parquet`, same columns as today.** `COMPARE` (per shard) takes each ORF and its baseline from the catalog, reads both through the readers, slices them to the unique and shared ranges and computes everything in §3.5. `SCORE_TAGS` adds `isoform_scoring_*` and `isoform_tags_*`. `MERGE` then builds one row per ORF: the isoform's `prot_annot` row as `isoform_*`, the baseline's as `canonical_*`, plus the site-keyed layers, the `cmp_*`/`diff_*` columns, scoring and tags; it joins the shards and writes the sidecars. Gate: the `cheeseman13` snapshot is identical, which proves compare and the `prot_annot`-derived columns reproduce today's values.

**One graph for every run:**
- **FASTA run** (`swissiso annotate --fasta x.fa`): the catalog has proteins but no ORFs, so `COMPARE` and `SCORE_TAGS` run on zero rows, and `MERGE` writes only `prot_annot.parquet`.
- **ORF with no baseline** (a novel ORF): `COMPARE` returns its row with `status = no_baseline` and null comparison columns, so it still appears in `all_paired.parquet`.
- **Isoform run:** both tables.

The MCP and the website call the same aggregate and compare functions, so their values cannot differ.

### 3.7 Order of work

1. **`layers/` and the catalog:** the dataclasses, a reader and a writer per shape, and the catalog from the §1 connector.
2. **Readers over data that already exists:** the "reader only" modules in §3.4. This tests the contract on real data without changing any module.
3. **Move the region logic into compare,** reading through the readers. Gate: `all_paired.parquet` on `cheeseman13` is unchanged.
4. **Rewrite the modules** that must emit new data (biophysics, conservation frame, mass spec, SignalP/TargetP), one at a time, each gated on the snapshot.
5. **`MERGE` writes `prot_annot.parquet` and builds `all_paired.parquet`'s `canonical_*` / `isoform_*` columns from it.**
6. **Remove** the region arguments, the `ProteinModule` / `SiteModule` protocols and the legacy `run()` wrappers, and update CLAUDE.md's Module Contract.

---

## 4. MCP

Review item 7 and mat10d's MCP guidelines on PR #60. **Purpose:** let a model answer questions about an ORF by reading the data itself (which domains sit where, how conserved those residues are, which variants fall there and what they change) instead of summaries, which lose this.

### 4.1 Placement

- **Package:** `src/swissisoform/mcp/`, present in the layout from the start and built last (after §3).
- **Read-only:** it never runs annotation modules.
- **Imports** the `layers/` readers, `aggregate` and `compare` only; nothing from the website or the LLM package. The LLM package's tool passes call the MCP's functions, not the other way round (§5.1).
- **Starting point:** `site/tools.py` and `site/structure_tools.py` move here (§5.1). What they do today is rebuilt on the readers instead of on `variants_long.parquet` and the fold cache:
  - `query_variants`, `variant_position_histogram`, `variant_effect_stats` (`tools.py:307, 359, 445`) → `get_variants` and `summarize`;
  - `plddt_profile`, `pae_block`, `secondary_structure`, `contacts` (`structure_tools.py:241, 281, 359, 449`) → `get_track`, `summarize`, `get_features`, `compare`.
- **Server:** the MCP Python SDK, serving one run directory (catalog + layers + `prot_annot.parquet` + `all_paired.parquet`), over stdio locally and HTTP for a shared deployment.

### 4.2 What it reads

The catalog (ORFs, proteins, ORF → genome mapping), every module's layers, and the two tables from §3.6.

### 4.3 Tools

A small generic set that works across modules, not one tool per module:

| Tool | Returns | Built on |
|---|---|---|
| `list_orfs(gene)` | ORFs for a gene: id, type, baseline if any | catalog `orfs.parquet` |
| `describe_orf(orf)` | Coordinates, length, baseline, and which layers exist with their status | catalog + each layer's `status` |
| `list_layers()` | Each layer's name, shape, key space, units and meaning | generated from the modules' `LAYERS` (§3.3) |
| `get_track(orf, layer, range)` | Per-position values | `read_track` → `TrackData` |
| `get_features(orf, range, types)` | Intervals overlapping the range (domains, motifs, secondary structure, peptides) | `read_features` → `FeatureData` |
| `get_variants(orf, range, filters)` | Variants with their consequence in that ORF's frame | `read_variants` → `VariantData`, consequence from `compare` |
| `align(orf, baseline)` | Shared, lost and gained ranges, in both ORFs' coordinates | `compare`'s range helper (from `plm/regions.py`) |
| `summarize(orf, layer, range)` | An aggregate over any range | `aggregate` |
| `compare(orf, baseline, range, layers)` | The comparator's functions on a chosen range | `compare` |
| `get_comparison(orf)` | The precomputed comparator row (scores, tags) | `all_paired.parquet` |

### 4.4 Rules

- **One implementation.** `summarize` and `compare` call the same functions the pipeline uses, so MCP values and website values cannot differ.
- **No baseline is a normal case.** For a novel ORF the positional tools work, and `compare` reports `no_baseline`.
- **Coordinates:** accept residue or genomic ranges, return both; 0-based half-open, as elsewhere. Conversion uses the catalog's `orf_exons` (`coords.py`).
- **"Not evaluated" stays distinct from zero or empty:** every result carries the layer's `status`.
- **Bounded output:** a maximum range per call, with paging for features and variants and downsampling for long tracks. Today's caps (`MAX_QUERY_LIMIT = 100`, `MAX_PROFILE_RESIDUES = 400`) are the starting values.
- **Provenance:** each result names the run and the module version it came from.

### 4.5 What it requires of §3

| Requirement | Where it is met |
|---|---|
| A schema per module that `list_layers` can be generated from | §3.3 rule 2 (`LAYERS`) |
| The ORF-to-genome mapping in the catalog | §3.1 (`orf_exons` in `orfs.parquet`) |
| Aggregator and comparator functions that take an arbitrary range | §3.5 |
| Run and module version on each layer | §3.2 (to add to the layer metadata) |

### 4.6 Order of work and check

1. Server skeleton and `list_orfs`, `describe_orf`, `list_layers` over the catalog.
2. The read tools (`get_track`, `get_features`, `get_variants`) on the readers; move `site/tools.py` and `site/structure_tools.py` here.
3. `align`, `summarize`, `compare`, `get_comparison`.
4. Point the LLM package's M/P tool passes at these functions.

**Check:** on `cheeseman13`, `compare(orf, baseline, unique range, …)` and `get_comparison(orf)` return the same values as `all_paired.parquet`, and `summarize` matches `prot_annot.parquet`.

---

## 5. Code cleanup

Each item is its own change, gated on the `cheeseman13` snapshot staying identical. 5.1 is required (review item 6); 5.2 can follow.

### 5.1 Required: move the LLM code into its own package

Today the LLM code sits in `src/swissisoform/site/`, mixed with website-data builders, and the website imports from it. The code and prompts are kept as they are; only where they live changes.

**Where it is now:**

| File | Lines | Contents |
|---|---|---|
| `site/llm.py` | 3364 | Prompt assembly, Anthropic calls (`llm.py:971, 1160`), category and synthesis passes |
| `site/evidence.py` | 2498 | Per-gene evidence records (`build_gene_record` `:408`); the criteria registry (`CRITERIA` `:624`, `CATEGORIES` `:1160`, `CRITERIA_METRIC_LABELS` `:1229`); the slicers (`slice_criterion` `:1842`, `slice_category` `:2451`); `write_variants_long` (`:509`) for the site |
| `site/grounding.py` | 801 | Alternative evidence payloads (prompt variants) |
| `site/tools.py`, `site/structure_tools.py` | 709, 711 | Tool readers for the M and P passes |
| `scripts/site/prompts/` | — | Prompt text and `output_schemas/*.json`, found through `ROOT = parents[3]` (`llm.py:40`, `grounding.py:53`) |

**Target:**

```
src/swissisoform/
  criteria.py        CRITERIA, CATEGORIES, metric labels, thresholds,
                     slice_criterion, format_metric                                    (from evidence.py)
  llm/
    records.py       build_gene_record, write_evidence_records                        (from evidence.py)
    slices.py        slice_category and the LLM-only shaping                           (from evidence.py)
    grounding.py                                                                       (from site/)
    prompts.py       prompt loading and assembly, TOOL_CATEGORY_PROMPTS                (from llm.py)
    client.py        Anthropic client, batches, schema validation                     (from llm.py)
    passes.py        category / synthesis passes, CLI entry                            (from llm.py)
    prompts/         prompt text + output_schemas, as package data                     (from scripts/site/prompts/)
  mcp/
    tools.py, structure_tools.py                                                       (from site/, §4)
  site/
    skeletons.py, variants_long.py                                                     (website data only)
```

- `criteria.py` holds the registry and the criterion slicer because the website (`app.py:43`), `figures/` and `llm/` all use them. The website then imports `criteria`, not `site.evidence`, and the `prepare_deploy.sh` copy list (`:79-96`) changes with it.
- The M/P tool readers go straight to `mcp/`; `llm/` imports them from there.
- `judge/` stays a separate package and imports `llm/`.
- Prompts load through `importlib.resources`, so nothing depends on the repo path.
- `anthropic` and `jsonschema` move from core dependencies to an `llm` extra.

**Callers to update:**
- `scripts/site/run_llm_interpretation.py`, `run_llm_variants.py`, `build_evidence_records.py` → `swissiso llm …`;
- `judge/reference.py`, `scripts/judge/build_requests.py`, `run_checks.py`;
- `website/src/swissisoform_site/data.py:40-43`, `app.py:43`;
- `figures/clustering_dims/feature_space/export_feature_catalog.py`, `figures/prompt_variants/assemble.py`;
- 11 test files under `tests/`.

**Check:** prompts saved with `--dry-run --save-prompts` on `cheeseman_test` are byte-identical before and after, and the cheeseman13 snapshot is unchanged.

### 5.2 Later

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
- **Site:** install the package into the website image instead of `prepare_deploy.sh` copying a hand-kept file list. The copy currently in the image is already stale.
- **Research code:** keep `judge/` and `figures/` apart from the pipeline, with no `sys.path` hacks.
- **Tests:** mirror `src/`, add a `realdata` marker, no tests that load scripts by path.
- **Hygiene:**
  - Delete `modules/conservation_homology.py` (dormant), the unused `PipelineConfig`/`ScoringConfig` fields, `scripts/_backfill_cols.py` and `scripts/_run_cheeseman13.sh`.
  - Decide whether `docs/architecture/` and `docs/plans/` should be tracked; `.gitignore` excludes them today.

---

## 6. Modal for the GPU stage, and possibly PepQuery

Review item 5: a full run in about a day, at $500–1,000 per run. GPU work and PepQuery are the two long poles; everything else stays on cluster CPUs.

### 6.1 What there is to move

| Stage | Work in the July campaign (10,851 proteins) | Why it is slow on the cluster |
|---|---|---|
| `ESMC` + `SAE` | 214.6 A6000 GPU-h (8 array tasks, longest 56.6 h) | 2 running A6000 jobs per user |
| `FOLD` | 117.8 A6000 GPU-h (8 tasks, longest 18.8 h) | same limit; fold queued ≈ 37 h behind embed |
| `MASSSPEC` (PepQuery) | 39,079 min summed over shards at 8 CPUs, ≈ 5,200 core-h upper bound | ran inside each annotate shard (§2A) |

### 6.2 What a Modal run entails

The science code does not change; Modal wraps the existing chunk entry points (`plm.cli`, `plm.sae`, `structure.cli`).
1. **Images** replace the conda envs: a GPU image (torch, transformers, esm, `swissisoform`) and, for PepQuery, a Java image.
2. **Volumes:**
   - model weights, downloaded inside Modal from Hugging Face (`biohub/ESMC-6B`, ESMFold2), so nothing is uploaded from the cluster;
   - for PepQuery: the spectra library (≈ 196 GiB) mirrored inside Modal from PepQueryDB's S3, plus the FM-index (`data/reference/pepquery/db/`).
3. **Functions:** `@app.function(gpu=…, volumes=…, timeout=…)` per process; fan-out is `.map()` over the chunk FASTAs.
4. **Data movement:**
   - in: the chunk FASTAs (MB);
   - out: only what the CPU side reads, `llr`, `aa_logprobs`, SAE `idx`/`val` and the fold outputs (≈ 8 GB today: 6.9 GB fold, 1.1 GB SAE, plus the two small ESM-C arrays);
   - the embeddings (`embedding`, `embedding_sae`: two (L, 2560) float32 arrays per protein, most of today's 49 GB ESM-C cache) stay on Modal. Only the SAE encode reads them (`sae.py:244`), so `ESMC` and `SAE` run in one Modal function.
   - Results are written to the cache atomically (temp file + `os.replace`, §2C).
5. **Nextflow:** there is no Modal executor. `ESMC`, `SAE` and `FOLD` (and `MASSSPEC`, if moved) run with `executor 'local'` in the driver job and call a Modal client that blocks until its chunk's results land; `-resume` and `--rerun` work as for any process.
6. **Admin:** Modal account and token as a secret, a concurrency cap and spend limit per run, and a check that ESM-C 6B's licence allows running it on Modal.

### 6.3 Cost and time (estimates to confirm with 6.4)

Assumes an H100 is 2–3× an A6000 for these models, and Modal list prices of about $4 per H100-hour, $0.05 per CPU core-hour and $0.008 per GiB-hour of memory; check current prices before committing.

| Stage | Compute | Cost | Wall time |
|---|---|---|---|
| GPU (`ESMC` + `SAE` + `FOLD`) | ≈ 110–170 H100-h | ≈ $450–700 | ≈ 3–4 h at 50 GPUs (3.6 d today) |
| PepQuery on Modal | ≈ 5,200 core-h + 64 GB per task | ≈ $500–600 | hours, at any width |
| PepQuery on cluster partition `24` (300 CPUs) | same | $0 | ≈ 17 h if packed, inside a day |

**Recommendation:** move the GPU stage to Modal first; that alone takes the campaign from ≈ 7 d to about a day, inside the budget. Keep PepQuery on partition `24` (§2A), running alongside the GPU stage, and move it to Modal only if the 2A benchmark shows it cannot finish in a day there. Both on Modal would sit at or above the top of the budget.

### 6.4 Before committing

1. **Benchmark** one 1,500-protein chunk of `ESMC`+`SAE` and of `FOLD` on H100 and A100 on Modal: time, peak GPU memory (1,024-aa proteins), and cost per protein. Check outputs are identical to the cluster cache for the same proteins.
2. **PepQuery:** the 2A benchmark (full log, `-cpu` scaling, memory) decides between partition `24` and Modal.
3. **Licence** for ESM-C 6B on third-party cloud GPUs.

### 6.5 Order of work

1. Modal images and Volumes; weights downloaded inside Modal.
2. Modal functions for `ESMC`+`SAE` and `FOLD` over the existing entry points; the 6.4 benchmark.
3. Wire them into the Nextflow processes (`executor 'local'` + blocking client).
4. Full campaign GPU stage on Modal; compare a sample of outputs to the July cache.
5. PepQuery on Modal only if 6.4 step 2 says so.
