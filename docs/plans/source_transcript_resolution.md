# Plan: Source-Transcript Resolution → high-confidence (TIS, mRNA, efficiency) dataset

> Agreed plan (approved 2026-06-16). Mirrors the working plan file; lives here per the
> "scope before plan" convention. Pairs with `docs/architecture/upstream_filtering_and_dedup.md`
> and `docs/architecture/ribotish_tis_assignment.md`.

## Context

**Problem.** Ribo-TISH assigns each translation-initiation site (TIS) to *every*
annotated transcript whose splice structure is compatible with the footprint reads
(see `docs/architecture/ribotish_tis_assignment.md`). It ignores which mRNA species
are actually present in our cells, so each TIS carries a polluted set of candidate
source transcripts (`all_transcripts`) with divergent 5′UTR/mRNA sequences. That
pollution is fatal for the end goal: a model predicting **translation-initiation
efficiency from mRNA sequence** needs each TIS paired with *the* sequence that
produced it.

**Approach.** Narrow each TIS's candidate transcripts with cell-line-specific
evidence, then enforce sequence purity:
1. **Transcript expression** (RNA-seq → per-transcript abundance) → drop
   annotated-but-unexpressed isoforms.
2. **CAGE 5′ grounding** → keep/snap transcripts whose TSS is supported by a CTSS cluster.
3. **Window-purity test** → if surviving candidates are **identical within ±W nt of the
   start codon**, keep the TIS with that consensus sequence; if they diverge inside the
   window, discard it.

**Key insight:** the window-agreement test means we do **not** need to fully resolve
which single isoform a footprint came from — if every surviving candidate shares the
same local sequence, the TIS efficiency maps to a unique sequence regardless. We only
discard when the local sequence is genuinely ambiguous.

**Outcome.** A curated table: one row per kept (TIS × cell line) with the
high-confidence source mRNA sequence, the ±W window, a per-transcript initiation
efficiency, and provenance/confidence — the training set for the model.

## Scope (initial pass)

**HeLa only.** HeLa is the one line with *both* expression arms — Arm A Illumina
(`BVTGPN_3_HeLa_1`, `BVTGPN_21_HeLa_2`) and Arm B long-read (ONT, `GSM8529146`) — so it
fully exercises the funnel **and** the Arm-A-vs-Arm-B disambiguation benchmark end-to-end
on a single cell line. All other lines (K562, U2OS, RPE1 Async/Que/Sen) are **future
work**: the code is written multi-sample-ready, but Phases 0–4 run on HeLa first. This
also defers the pending RPE1 Que/Sen FASTQ and the non-HeLa CAGE sourcing.

## RNA-seq data model (resolved 2026-06-16 by inspecting the actual files)

Three modalities are relevant; two feed transcript-existence as **interchangeable arms**
(benchmarked), one is excluded.

- **Arm A — standard Illumina RNA-seq (~94–100 nt, single-end)** [our data]. Spans
  junctions → isoform-level salmon quantification.
  - HeLa, K562, U2OS: `/lab/barcheese01/aTIS_data/rnaseq/BVTGPN_*_{line}_{rep}.fastq.gz`
    (2 reps each, 94 nt SE).
  - **RPE1_Async**: the generic `RPE1` BVTGPN samples (`BVTGPN_7_RPE1_1`,
    `BVTGPN_25_RPE1_2`) — **confirmed by user to be RPE1 Async**.
  - **RPE1_Que, RPE1_Sen**: separate ~100 nt RNA-seq — **path pending (user to provide)**.
- **Arm B — public long-read RNA-seq (PacBio Iso-Seq / ONT, full-length)** [other labs].
  Each read = one isoform → **unambiguous mRNA-species identification**. Coverage: **K562**
  (ENCODE/TALON, strong), **HeLa** (ONT PromethION, GSE277764 / GSM8529146); **U2OS / RPE1
  have no known public long-read data** → Arm A only there. Pulled from GEO/SRA and
  processed in-house (minimap2 + IsoQuant vs GENCODE v49); see "Expression evidence".
- **Excluded — footprint-modality "mRNA input"**
  (`…/Ribosomeprofiling/2021_TISseqV6_*/Data/Trim/*_trim.fastq.gz`): ~18–34 nt small
  fragments, SE, heavily rRNA/adapter-contaminated (~7% transcriptome alignment). Too
  short to discriminate isoforms. Retained only as a *candidate* fragment-size-matched
  efficiency denominator (see Efficiency); **not** an existence signal.

## Decisions (confirmed with user)

| Question | Answer |
|---|---|
| Initial scope | **HeLa only** (the one line with both arms → exercises funnel + benchmark); other lines are future work |
| Expression evidence | **Two interchangeable arms** — Arm A short-read salmon TPM (our Illumina RNA-seq) + Arm B public long-read isoform quant (in-house minimap2+IsoQuant). **Benchmark which better disambiguates TIS transcripts; use the superior per cell line.** CAGE + window-purity finalize |
| RNA-seq form | Arm A: Illumina SE ~94–100 nt, all 6 lines (RPE1 Que/Sen pending). Arm B: public PacBio/ONT pulled from GEO/SRA — K562 (ENCODE) + HeLa (GSE277764/GSM8529146); none yet for U2OS/RPE1 |
| Efficiency metric | **TIS reads / per-transcript RNA abundance** (per-mRNA initiation rate) |
| CAGE | Lab CTSS/peaks **on disk**, GRCh38 (path pending) |
| Deliverable | **Both** — standalone dataset workstream first, fold into core pipeline later |

## Required inputs (contracts — fill before Phase 2/3)

- **Illumina RNA-seq manifest** (Arm A, 6 lines × reps): BVTGPN paths for
  HeLa/K562/U2OS/RPE1_Async + **RPE1 Que/Sen ~100 nt path (pending)**. Build
  `data/reference/rnaseq_illumina_manifest.csv` (sample → fastq list), reusing the
  cell-line naming in `ribotish_replicate_manifest.csv`.
- **Long-read accession manifest** (Arm B): `data/reference/longread_manifest.csv`
  (cell_line, platform, accession, condition) — public GEO/SRA accessions (e.g. HeLa
  `GSM8529146`; K562 ENCODE). Pulled reproducibly by `scripts/setup/download_longread.sh`.
- **CAGE** CTSS clusters as BED per cell line (chrom,start,end,strand,score), GRCh38 /
  GENCODE v49 chrom naming (path pending).
- `gencode.v49.transcripts.fa` (comprehensive transcriptome) — verify/download; add to
  `scripts/setup/download_references.sh`. Genome FASTA + GTF already present.

## Architecture — the narrowing funnel

Driver reads `data/output/filtered/all_samples_combined.parquet`, groups rows by
**genomic init site** (`chrom:gstart:strand:codon`, via `_init_site_from_genome_pos`,
`combine.py:157`). Each group = one TIS with its candidate `Tid`s and per-cell-line
`{sample}_TISCounts`. Per cell line:

1. **Expression filter** — keep `Tid` expressed per the **chosen expression provider**
   (Arm A salmon `TPM ≥ τ` in **every** replicate, or Arm B long-read isoform abundance
   ≥ `τ_LR`). The provider is chosen per cell line by the benchmark.
2. **CAGE grounding** — keep `Tid` whose 5′ end (first-exon boundary from the exon
   skeleton, strand-aware) is within `D` bp (default 50) of a CTSS cluster; record the
   snapped CAGE TSS as the high-confidence 5′ boundary.
3. **mRNA window** — for each survivor, build the spliced mRNA, locate the start-codon
   cDNA index, slice `[idx−W, idx+W]` (default W=100; clamped at the 5′ end).
4. **Purity decision** — anchor all survivors at the start codon, compare outward
   base-by-base; `keep` if all identical within ±W, else `discard (ambiguous_window)`.
5. **Efficiency** — `init_efficiency = TIS_footprints / per-transcript RNA abundance`
   per cell line. When several survivors agree in-window (indistinguishable), **sum
   their TPM** for the denominator and keep the max-TPM survivor as representative.

## Expression evidence — Arm A: short-read salmon (concrete spec)

salmon (>=1.10, from the `swissisoform-v2` env) runs with selective alignment + decoys.
Inputs are **single-end, ~94–100 nt** Illumina reads (one FASTQ per replicate).
Short-fragment footprint reads are **not** used here (see RNA-seq data model).

**Pre-processing (before salmon).** Each replicate FASTQ is run through **fastp**:
adapter auto-detection + trimming, quality filtering (Q15), polyG/polyX tail trim, and
dropping reads <20 nt / failing quality — with a per-sample JSON/HTML QC report. salmon
then quantifies the *cleaned* reads. (The BVTGPN HeLa reads show adapter readthrough, so
this matters.) Implemented in `scripts/setup/run_salmon_hela.sbatch`.

**Index (`salmon index`)** — build once, reuse for all samples:
- Transcriptome = **GENCODE v49 comprehensive** `transcripts.fa` — must match the GTF
  Ribo-TISH used, so every candidate `Tid` has a target (else it gets no TPM and is
  wrongly dropped).
- **Decoy-aware**: `gentrome.fa = cat transcripts.fa genome.fa`; `decoys.txt` = genome
  contig names (`grep '^>' genome.fa | sed 's/>//; s/ .*//'`). Suppresses spurious
  intronic/unannotated mapping.
- `--keepDuplicates` (retain identical-sequence transcripts so no `Tid` is lost —
  these are exactly the ones that agree in the ±W window anyway), `-k 31` (fine for ≥94 nt).

**Quant (`salmon quant`)**, per replicate FASTQ:
- `-l A` to auto-detect strandedness → **lock** the resolved type from
  `lib_format_counts.json` (expect `SR`/`SF`/`U`) and reuse it across samples.
- `-r <fastq>` (single-end).
- **SE fragment-length is mandatory** — set `--fldMean`/`--fldSD` to the library insert
  (standard ~94 nt RNA-seq → sheared insert ~200–300; default 250/25 reasonable, confirm
  with the prep). TPM depends on effective length, which depends on this.
- `--seqBias` (hexamer-priming bias); `--gcBias` optional for SE (less reliable without
  paired fragments).
- `--rangeFactorizationBins 4` (salmon's recommendation for transcript-level resolution
  when isoforms share sequence — directly relevant to narrowing).
- `--numBootstraps 30` (per-transcript inferential variance → flags which isoform TPMs
  are trustworthy vs split among near-identical isoforms).
- `-p <threads> -o data/reference/salmon/{sample}_{rep}`.

**ID matching & aggregation**: salmon `Name` is the full GENCODE header
(`ENST…|ENSG…|…`); split on `|` → bare `ENST.version` to join on `Tid` (handle/drop
`_PAR_Y`). **Presence** = TPM ≥ τ in *every* replicate (`expressed_in_replicates`,
`require="all"`); the mean-TPM map (`load_salmon_replicates`) is kept for the later
abundance/efficiency step.

**QC sanity**: long reads should map at a high rate (in stark contrast to the ~7% of the
short-fragment footprint data); a low rate signals a wrong index/strandedness or
contamination. Log mapping rate per sample from `logs/salmon_quant.log` / `lib_format_counts.json`.

**Slurm sizing**: the decoy-aware index build is the heavy step (genome+transcriptome,
~16–24 GB RAM, ~30 min) — confirm partition/limits per cluster policy. Per-sample SE
quant of a ~1 GB FASTQ is light (~8 threads, a few GB, minutes); run as a `--array` over
the 6 lines × replicates.

*(Optional secondary signal)* observed splice junctions from a STAR pass on the Illumina
reads (`SJ.out.tab`) to corroborate expressed isoforms.

## Expression evidence — Arm B: public long-read RNA-seq (in-house processed)

Full-length long reads identify mRNA species **unambiguously** (one read = one isoform),
the highest-confidence isoform-existence signal.

- **Source (public GEO/SRA, pulled reproducibly):** a versioned accession manifest
  `data/reference/longread_manifest.csv` (cell_line, platform, accession, condition) drives
  `scripts/setup/download_longread.sh` (sra-toolkit 3.07 `prefetch` + `fasterq-dump`, ENA
  HTTP fallback, idempotent + provenance sidecar) so anyone can reproduce the pull.
  Coverage so far: **HeLa** — ONT PromethION polyA+, GSE277764 / **GSM8529146**
  (`SRX26164613`, untreated rep1; the series also has arsenite/heat-shock conditions);
  **K562** — ENCODE PacBio (TALON); **U2OS / RPE1** none known yet → Arm A only there.
- **Processing (HeLa, ONT cDNA — confirmed GSE277764):** `minimap2 -ax splice -k14
  --junc-bed <GENCODE v49 junctions>` (no `-uf`; cDNA) → sort → **IsoQuant** *reference-only*
  (`--no_model_construction --complete_genedb --stranded none --data_type nanopore`) vs the
  GENCODE v49 GTF → per-transcript counts. Annotated junctions (`paftools.js gff2bed`) guide
  noisy ONT splice mapping. Emits the same `{sample: {Tid: abundance}}` interface as Arm A;
  presence = counts ≥ 3. Runs in the dedicated `isoquant` env on Slurm.

## Expression evidence — benchmark (choose the arm)

Per the decision, **measure which arm better disambiguates TIS candidate transcripts** and
use the superior one (per cell line, since coverage differs).

- **Long-read as gold standard:** on lines with both (K562, maybe HeLa), the long-read
  isoform set is truth for "which isoforms exist"; measure short-read salmon's precision/
  recall against it.
- **Disambiguation metric (per arm × cell line)**, over multi-candidate TIS (`n_raw ≥ 2`):
  (a) candidate reduction `n_raw → n_after_expression`; (b) fraction resolved to a single
  **pure-window** source (combined with the window-purity test); (c) consistency with
  CAGE-supported TSS. Higher resolved-fraction *without* contradicting CAGE / long-read
  truth = superior.
- **Decision rule:** use long-read where it exists and wins; salmon where long-read is
  absent; if salmon tracks long-read well on K562, trust salmon on the uncovered lines,
  else flag those lower-confidence. Record the chosen provider per cell line in dataset
  provenance.

## New code — standalone subpackage `src/swissisoform/sourceseq/`

- **`mrna.py`** — `build_transcript_mrna(skeleton, genome) -> (cdna_seq, cds_start_idx)`
  (full spliced 5′UTR+CDS+3′UTR) and `extract_tis_window(skeleton, genome, start_genomic, W)`.
  Reuse FASTA-fetch + revcomp from `clinical/validate.py:269`
  (`build_coding_sequence_from_orf`) and `assembly.py:40` (`extract_kozak_context`);
  reuse `orf_exons_from_skeleton` + interval algebra (`coords.py:13,100-156`).
- **`expression.py`** — `load_salmon(quant_paths) -> {sample: {Tid: TPM/NumReads}}`;
  ENST-version normalization; replicate averaging.
- **`longread.py`** — load in-house long-read isoform quant (IsoQuant/FLAIR output) →
  `{sample: {Tid: abundance}}`, the same interface as `expression.py` so the funnel is
  provider-agnostic.
- **`benchmark.py`** — compare Arm A vs Arm B TIS-disambiguation power (metric in
  "Expression evidence — benchmark"); emit a report + the per-cell-line provider choice.
- **`cage.py`** — `load_ctss(bed)`; `tss_supported(skeleton, clusters, D)`; `snap_tss(...)`.
  Fills the missing CAGE/TSS handling.
- **`narrow.py`** — the funnel; consumes the combined table + `UpstreamReference`
  (`pipeline.py:179`, holds `exon_skeletons`); emits curated rows with per-stage
  candidate counts and discard reasons.
- **`efficiency.py`** — per-transcript initiation efficiency, superseding the gene-level
  `initiation_efficiency` (`assembly.py:461`).

## Scripts

**Built (HeLa initial pass):**
- `scripts/setup/create_conda_env.sh` — build the `swissisoform-v2` (+ `isoquant`) conda
  envs from `environment.yml` / `environment.isoquant.yml` (`--rebuild`, `--with-isoquant`).
- `scripts/setup/download_references.sh` — GENCODE v49 genome/GTF/translations **+ the
  transcriptome** (`gencode.v49.transcripts.fa`, added for the salmon index).
- `scripts/setup/run_salmon_hela.sbatch` — **fastp** QC/trim → decoy-aware GENCODE v49
  salmon index → quant of the two HeLa Illumina reps → `data/reference/salmon/HeLa_rep{1,2}/quant.sf`.
- `scripts/setup/run_longread_hela.sbatch` — pull HeLa ONT (from `longread_manifest.csv`)
  → minimap2 → IsoQuant (in the `isoquant` env) → `data/reference/longread/isoquant_HeLa/`.
- `data/reference/longread_manifest.csv` — HeLa ONT accessions
  (`GSM8529146`→`SRX26164613`, `GSM8529147`→`SRX26164614`).

**Planned (later phases / other cell lines):**
- `scripts/analysis/benchmark_expression_arms.py` — Arm A vs Arm B disambiguation report
  + provider choice per cell line.
- `scripts/export/build_source_transcript_dataset.py` — driver →
  `data/output/source_transcripts/tis_source_mrna.parquet`.

## Output schema — `tis_source_mrna.parquet`

`init_site, chrom, gstart, strand, start_codon, gene, cell_line,
candidate_transcripts_raw, n_raw, after_expression, after_cage, chosen_transcript,
n_kept, mrna_window (±W), mrna_full, cage_tss, tis_counts, transcript_tpm,
init_efficiency, window_divergence_nt, confidence, discard_reason`.

## Phasing (so work isn't blocked on pending paths)

- **Phase 0** — collect FASTQ manifest (RPE1 Que/Sen path) + CAGE path + build; verify/
  download `gencode.v49.transcripts.fa`; build salmon index.
- **Phase 1 (no new data)** — `mrna.py` window + purity logic, fully testable now with
  the genome FASTA + exon skeletons already present. The algorithmic core.
- **Phase 2** — Arm A: salmon quant of **HeLa** Illumina reads (`BVTGPN_3/21_HeLa`) +
  `expression.py` filter. (Other lines: future.)
- **Phase 2b** — Arm B: pull **HeLa** long-read (`GSM8529146`) via `download_longread.sh`,
  minimap2 + IsoQuant vs v49 → `longread.py` abundances.
- **Phase 2c** — benchmark Arm A vs Arm B disambiguation **on HeLa** (`benchmark.py`);
  record the chosen provider. (Generalize per-line in future.)
- **Phase 3** — `cage.py` grounding.
- **Phase 4** — `build_source_transcript_dataset.py`: assemble dataset + efficiency +
  confidence; export.
- **Phase 5 (later, "fold in")** — add a transcript-narrowing stage in `runner.py` that
  rewrites `all_transcripts` to the resolved set for the core pipeline/downstream modules.

## Parameters / defaults

`W = 100` nt (configurable; asymmetric upstream-weighted allowed). **Presence:** Arm A
**TPM ≥ `τ = 1.0` in every replicate** (stringent); Arm B IsoQuant **counts ≥ 3**.
`D = 50` bp CAGE distance. build = GRCh38 / GENCODE v49 (must match across genome,
transcriptome, GTF, CAGE).

## Reused functions (file:line)

| Use | Reuse |
|---|---|
| Spliced-sequence FASTA fetch + revcomp | `clinical/validate.py:269`, `assembly.py:40` |
| Exon walk + interval algebra | `coords.py:13,100-156` |
| Exon skeletons / reference tables | `io/gtf.py:184`, `pipeline.py:179` |
| Group rows → init site | `combine.py:157` (`_init_site_from_genome_pos`) |
| Per-cell-line TIS counts | `{sample}_TISCounts` cols in combined (`combine.py`) |

## Risks / to revisit

- **RPE1 Que/Sen long-read path pending** — Phases 2–4 run on HeLa/K562/U2OS/RPE1_Async
  first; add the other two when provided.
- **Generic RPE1 = Async** is a user assertion — confirm sample provenance (`aTIS_data/DATA.md`).
- **Single-end `--fldMean/--fldSD`** affects TPM via effective length — confirm library
  insert size or use defaults with a caveat.
- **Efficiency modality mismatch** — numerator is footprint counts, denominator is
  standard-RNA TPM (a translation-efficiency-style ratio). The footprint-matched
  small-fragment input is an alternative denominator but is rRNA-contaminated/short;
  decide if/when to compare.
- **Window purity may discard many TIS** — log per-stage yield to tune `W`.
- **CAGE build/chrom-naming mismatch** vs GENCODE v49 — validate early.
- **Long-read coverage gap** — no public long-read for U2OS/RPE1 → Arm A only there; the
  benchmark validates whether Arm A alone is trustworthy on the uncovered lines.
- **Cross-lab long-read** (other labs' cells/conditions, not ours) — use as isoform
  *existence* evidence, not condition-matched abundance.
- **Long-read tooling/compute** — minimap2 + IsoQuant install (conda) + Slurm.

## Verification

- **Unit tests** (`tests/test_sourceseq_*.py`, synthetic exon skeletons + tiny FASTA):
  full-mRNA extraction, ±W window (5′-clamp + minus strand), purity decision (identical
  vs diverging candidates), salmon and CTSS parsers with fixtures.
- **E2E** on the existing 13-gene set (**HeLa**) via the export driver; inspect a
  known multi-transcript TIS that agrees in-window (kept) vs one that diverges
  (discarded); confirm `init_efficiency = TIS_footprints/TPM` numerically.
- **Salmon sanity**: check mapping rate (expect high for long reads, unlike the 7% of the
  short-fragment data) and per-stage candidate-count logs.

## Environment (reproducible)

For collaboration the env is **derived from Matteo's `swissisoform-v2`** so the shared
base is identical: a minimal conda layer (Python 3.11.15 + `uv`/`openjdk`/`typst`) with
the scientific stack installed via **pip at Matteo's exact pinned versions** (pandas
3.0.2, numpy 2.4.4, pysam 0.23.3, biopython 1.87, scipy, pyarrow, …) plus
`pip install -e .`. On that base we add **only** the sequencing CLIs this workstream
needs: **salmon** (short-read quant), **fastp** (short-read QC/trim), **minimap2** +
**samtools** (alignment), **sra-tools** (`prefetch`/`fasterq-dump`).

**IsoQuant** (long-read quant) is a **separate env** (`environment.isoquant.yml`) because
its conda deps (pandas/pysam/gffutils) would conflict with the pip-pinned versions above;
the long-read sbatch activates it for the quant step only.

Reproduce from scratch:

```bash
bash scripts/setup/create_conda_env.sh --rebuild --with-isoquant   # miniforge (lab space) + both envs
source /lab/barcheese01/$USER/miniforge3/etc/profile.d/conda.sh
conda activate swissisoform-v2
```

Specs live at the repo root: `environment.yml` (main, derived from Matteo's) and
`environment.isoquant.yml`. Miniforge installs to `/lab/barcheese01/$USER/miniforge3`
(lab space, not `$HOME`, to avoid home quotas).

**Verified build (2026-06-16):** main env = Python 3.11.15, pandas 3.0.2, numpy 2.4.4,
pysam 0.23.3, scipy 1.17.1, biopython 1.87; tools salmon 2.0.1, fastp 1.3.3, minimap2
2.31, samtools 1.23.1, sra-tools 3.4.1. `isoquant` env = IsoQuant **3.13.0** (+ minimap2,
samtools, k8/paftools.js). Note the IsoQuant CLI is `isoquant` (not `isoquant.py`).

## Docs (per the "scope before plan" convention)

- This plan lives at `docs/plans/source_transcript_resolution.md` (tracked).
- After Phase 4, write `docs/architecture/source_transcript_resolution.md` (current state,
  code-grounded), cross-linked with the filtering + Ribo-TISH architecture docs.
