# Handoff: long-read arm & source-transcript resolution (HeLa)

> **Audience:** someone picking this up cold. Reading this top-to-bottom should let
> you understand *why* the work exists, *what* was done, *where* every artifact
> lives, and *what to do next* — without needing the chat history. Written
> 2026-06-20, HeLa pilot. Companion deep-dives: the method numbers live in
> `docs/analysis/disambiguation_method_comparison.md`; the strategy rationale in
> `docs/plans/union_strategy_unambiguous_tis.md`; the parent workstream plan in
> `docs/plans/source_transcript_resolution.md`.

---

## 1. The 60-second orientation

We are building a training set to model **translation-initiation efficiency from
mRNA sequence**. Each training row = one translation-initiation site (TIS) paired
with **the mRNA sequence around its start codon**.

The problem: **Ribo-TISH** (the TIS caller) assigns each TIS to *every annotated
transcript whose splice structure is compatible*, ignoring which mRNAs actually
exist in the cell. So each TIS arrives with a **polluted cloud of candidate
transcripts** that have *different 5′UTRs*. To train a sequence model we must
collapse that cloud to **one unambiguous start-codon sequence per TIS**, or
honestly discard the site.

This session: we built the machinery to do that collapse using **three
independent signals**, processed the **long-read (Arm B)** data end-to-end,
benchmarked it against short-read, and produced a tiered dataset + report.

**Headline result (HeLa):** of 9,920 multi-candidate TIS, the **three-way union
resolves 7,425 (74.8%)** to an unambiguous ±100 nt window; long-read inclusion
specifically adds **1,439** of those. Combined with the 7,859 single-candidate
TIS, that's **15,284 HeLa TIS with a trusted source mRNA**.

---

## 2. The three signals (the core idea)

A multi-candidate TIS can be made unambiguous three independent ways:

| Signal | "Resolved" means | Strength | Weakness |
|---|---|---|---|
| **Sequence-window purity** | all candidate transcripts are byte-identical within ±W nt of the start codon | highest (correct by construction; don't even need to know which isoform) | decays as W grows (more bases → more chance to differ) |
| **Short-read (salmon)** | only one candidate is expressed (TPM ≥ τ), or several that agree in-window | empirical, condition-matched | **read-dilution**: near-identical isoforms split reads below threshold → false drops & false single-source picks |
| **Long-read (IsoQuant)** | only one candidate is expressed (counts ≥ τ), or several that agree | direct (**one read = one isoform, no dilution**) | lower depth; only exists for HeLa/K562; cross-lab (existence, not condition-matched abundance) |

**The union** keeps a TIS if *any* signal resolves it. They're complementary:
sequence handles "all candidates agree"; expression handles "candidates differ but
only one is real." See §6 for why the union is *tiered*, not flat.

---

## 3. The data

| Data | What | Where |
|---|---|---|
| Ribo-TISH predictions | raw TIS calls per cell line (21-col TSV, incl. AASeq) | `data/reference/{line}_TIS_predict_all.txt` |
| Reference genome | GRCh38 primary assembly | `data/reference/Gencode_v49_GRCh38.primary_assembly.genome.fa` |
| GTF | GENCODE v49 annotation | `data/reference/gencode.v49.primary_assembly.annotation.gtf` |
| Transcriptome | **the candidate mRNA sequences** (= what RNA-seq mapped to) | `data/reference/gencode.v49.transcripts.fa` |
| Short-read (Arm A) | HeLa Illumina RNA-seq → salmon `quant.sf` (2 reps) | `data/reference/salmon/HeLa_rep{1,2}/quant.sf` |
| **Long-read (Arm B)** | HeLa ONT → IsoQuant transcript counts | `data/reference/longread/isoquant_HeLa/OUT/OUT.transcript_counts.tsv` |

**Key fact:** candidate mRNA *sequences* come straight from `transcripts.fa`
(keyed by versioned ENST). The exon skeleton (from the GTF) is needed only to
locate *where* the start codon sits inside each transcript (the genomic→cDNA
mapping). Long-read does **not** add new sequences — IsoQuant ran *reference-only*
(see §4), so it quantifies the same GENCODE transcripts.

---

## 4. The long-read analysis, start to finish

**Source:** GSE277764 (Cell 2025, PMID 40436014) — HeLa ONT PromethION polyA+
cDNA. Two untreated replicates: `GSM8529146` (`SRX26164613`) + `GSM8529147`
(`SRX26164614`). Manifest: `data/reference/longread_manifest.csv`.

**Pipeline** (`scripts/setup/run_longread_hela.sbatch`, then the resume):
1. **Pull** — `prefetch` + `fasterq-dump` (SRA-tools). Each SRX → one SRR run
   (`SRR30762267`, `SRR30762266`). *Gotcha fixed this session:* `prefetch` names
   output dirs by **SRR**, not SRX — the original loop dumped by SRX path and
   404'd. Fix: glob the actual `.sra` files (`sra/*/*.sra`).
2. **Align** — `minimap2 -ax splice -k14 --junc-bed <GENCODE v49 junctions>`
   (cDNA, no `-uf`) → sort → `HeLa.bam`. **95.9% mapped** (vs ~7% for the
   short-fragment footprint data — the QC sanity check passed).
3. **Quantify** — **IsoQuant 3.13.0**, *reference-only*
   (`--no_model_construction --complete_genedb --data_type nanopore --stranded none`)
   vs the v49 GTF → `OUT.transcript_counts.tsv` (509,653 transcripts; `feature_id`
   = versioned ENST, matches our candidates exactly; columns `feature_id`,`count`).
   Runs in the dedicated **`isoquant` conda env**.

**Job saga (so you understand the elapsed time):**
- `10178027` — first run, died at the SRX/SRR dump bug (above).
- `10180797` — after fix, the *quantification* phase hit the **12 h walltime**.
  Resumed via `scripts/setup/run_longread_hela_isoquant_resume.sbatch`
  (`isoquant --resume`, reuses per-chromosome collected chunks). The resume ran
  **~40 h** and COMPLETED — the tail (chr1/chr2) is a slow **serial** per-chromosome
  collection step, **not memory-bound** (peaked 47 GB of 128 GB). The resume sbatch
  walltime is now **4 days** to avoid repeat timeouts.

**Lesson for the next line (K562):** IsoQuant's collection phase is serial on the
biggest chromosomes; more RAM/threads won't speed it. If it's a problem, run
per-chromosome as a job array (each chromosome gets a fresh walltime).

---

## 5. The funnel (HeLa numbers, end to end)

```
Ribo-TISH raw calls ........................ 521,699   (transcript × start predictions)
  ▼ TSL/MANE reference-transcript gate (smaffa)        −397,825
TIS on reference transcripts (MANE/TSL1-3) . 123,874
  ▼ significance (TIS/Ribo p, Fisher q) + read-count + dedup; canonical imputation
Pass significance + read-count filter ...... 34,273
  ▼ collapse transcripts sharing one genomic start
Distinct initiation sites in HeLa .......... 17,779
  ▼ split by candidate count
   ├─ single-candidate: 7,859  (mRNA trivially known ✓)
   └─ multiple candidates: 9,920  (ambiguous — the work)
        ▼ union step 1: sequence purity OR short-read (salmon)
      resolved: 5,986
        ▼ + long-read recovers dilution losses          +1,439 kept
      three-way union resolved: 7,425  (−2,495 ambiguous, discarded)
  ▼ merge single + resolved
TIS with UNAMBIGUOUS mRNA .................. 15,284   ← the training set
```

The **TSL/MANE filter** (the first big cut) is *pre-existing* smaffa upstream
filtering (`src/swissisoform/filtering.py:identify_reference_transcripts`) — it
restricts candidates to well-supported annotations. Our three-signal disambiguation
is *additive*, operating on the survivors (everything below "Distinct initiation
sites").

---

## 6. Results & the key nuance (read this)

All at **W=100**, over the **9,920 multi-candidate** TIS:

| Method | resolved | lost | ambiguous |
|---|---|---|---|
| sequence-only | 3,474 (35.0%) | — | — |
| salmon (TPM≥0.1) | 4,146 (41.8%) | 5,111 | 662 |
| long-read (counts≥3) | 4,978 (50.2%) | 1,967 | 2,974 |
| union (seq + salmon) | 5,986 (60.3%) | — | — |
| union (seq + long-read) | 5,879 (59.3%) | — | — |
| **3-way union** | **7,425 (74.8%)** | — | 2,495 |

**The nuanced finding (don't miss this):** long-read resolves *more by expression*
and loses far fewer than salmon — but the two **pairwise unions nearly tie**.
Why? Long-read honestly keeps **genuinely co-expressed** isoforms (2,974 ambiguous
vs salmon's 662), whereas salmon's read-dilution *artificially* collapses to one.
Consequences:
- **~32% of salmon's single-source calls are likely dilution artifacts** (long-read
  finds >1 expressed isoform there).
- Salmon's low-confidence (sub-1-TPM) calls agree with long-read only **~62–64%**.

So **long-read is the better HeLa provider for *trustworthiness*, not raw count.**

**The three-way union is therefore TIERED, not flat** (`HeLa_tis_union3_W100.parquet`):

| Tier | TIS | Evidence | Trust |
|---|---|---|---|
| ① sequence-pure | 3,474 | identical in ±100 | highest |
| ② long-read / corroborated | 2,405 | direct molecular existence, or ≥2 methods | high |
| ③ salmon-only | 1,546 | salmon's single pick, uncorroborated | **flag — likely artifact** |

Source precedence per TIS: **long-read → short-read → sequence-representative**
(most-trustworthy available). The +1,546 that the 3-way adds over seq+long-read is
*entirely* Tier 3 — keep for recall, down-weight in the model.

**Window radius is a confidence/context dial.** Sequence purity: 93.7% @±5 → 35% @±100
→ 8.5% @±500. Expression is window-independent (flat floor). They cross at **W≈78**.
The 3-way union stays high even at large W (62.4% @±500) — it's robust to the choice.

---

## 7. Code written this session

All under `scripts/analysis/` unless noted. Pipeline order top-to-bottom. Anything
that loads the 3 GB GTF skeletons runs on **Slurm** (partition 20); the rest is
head-node-light.

| Script (+ sbatch) | Does | Key inputs → outputs |
|---|---|---|
| `candidate_mrna_divergence.py` (`run_…sbatch`) | pull candidate mRNAs, extract ±W window per candidate, run sequence-purity test | `init_site_skeleton.parquet` + GTF + genome + transcripts.fa → `HeLa_tis_window_summary.parquet`, `HeLa_tis_candidate_detail.parquet` |
| `disambiguate_expression.py` | filter candidates by expression, re-run purity, tag source abundance. **Provider-agnostic**: `--provider salmon` or `--provider isoquant` | candidate_detail + `quant.sf` / IsoQuant TSV → `HeLa_disambiguation_{salmon_tpm0.1,isoquant}.parquet` |
| `window_purity_vs_length.py` (`run_…sbatch`) | sweep W=5..500, 2-line purity curve (seq vs salmon) | → `HeLa_purity_vs_window.{csv,png,_radius.parquet}` |
| `window_methods_comparison.py` (`run_…sbatch`) | sweep W for **all** methods incl. long-read + 3-way union | + `--isoquant-table` → `HeLa_methods_vs_window.{csv,png,_radius.parquet}` |
| `build_hela_tis_dataset.py` | assemble seq+salmon union dataset (read counts, TIE, source, window) | → `HeLa_tis_disambiguated_union_W100.parquet` |
| `build_union3_dataset.py` | assemble **three-way** tiered dataset (resolved_by, tier, precedence source) | → `HeLa_tis_union3_W100.parquet` |
| `plot_source_transcripts.py` | render figures from on-disk outputs (no GTF) | → `data/figures/source_transcripts/*.png` |
| `make_source_transcript_report.py` | build the self-contained HTML report (figures embedded base64) | → `data/reports/source_transcript_resolution_report.html` |

Setup scripts (`scripts/setup/`): `run_longread_hela.sbatch` (pull→align→quant),
`run_longread_hela_isoquant_resume.sbatch` (resume, 4-day walltime),
`test_sra_connectivity.sbatch` (one-off network probe).

**Reused library core** (`src/swissisoform/sourceseq/`, pre-existing, do not
duplicate): `mrna.py` (`build_transcript_mrna`, `start_codon_index`,
`extract_tis_window`, `TisWindow`), `purity.py` (`divergence_radius`,
`purity_decision`), `expression.py` (`load_salmon_replicates`,
`load_isoquant_abundance`, `expressed_in_replicates`, `expressed_transcripts`).
GTF loaders: `io/gtf.py` (`load_exon_skeletons`, `load_transcript_annotations`).

**Skills added** (`.claude/skills/`): `cluster-job` (Slurm submit+watch
conventions), `source-transcript` (run the funnel for a cell line).

---

## 8. Outputs (what's on disk)

Under `data/output/source_transcripts/`:
- `HeLa_tis_window_summary.parquet` — per-TIS sequence purity (one row/TIS).
- `HeLa_tis_candidate_detail.parquet` — per (TIS × candidate) ±100 windows.
- `HeLa_disambiguation_salmon_tpm0.1.parquet` / `…_isoquant.parquet` — per-arm resolution + source tags.
- `HeLa_tis_disambiguated_union_W100.parquet` — seq+salmon union dataset.
- **`HeLa_tis_union3_W100.parquet`** — the three-way tiered dataset (the main deliverable). Columns: `init_site, gene, chrom, gstart, strand, start_codon, orf_type, n_candidates, hela_*_counts, resolved, resolved_by, n_methods, agreement_tier, agreement_label, source_transcript, source_evidence, source_salmon_tpm, source_longread_counts, source_agreement_salmon_vs_longread, tie_initiation_efficiency, start_codon_pos_in_window, mrna_window_100, full_mrna, full_mrna_unambiguous`.
- `HeLa_{purity,methods}_vs_window.{csv,png}` + `_radius.parquet` — the W-sweep curves (recompute curves from the radius parquets without re-walking the GTF).

Figures: `data/figures/source_transcripts/`. Report:
`data/reports/source_transcript_resolution_report.html` (open in a browser; the
flowchart + tiered examples + the benchmark live there).

---

## 9. Decisions made (and why)

- **Window W = 100 nt** default — a confidence/context dial; shorter keeps more TIS.
- **Salmon presence = TPM ≥ 0.1 in both reps** — TPM≥1 was *net-negative* (dilution
  drops 75%); 0.1 is the operating point, but salmon stays Tier-3-only.
- **Long-read presence = counts ≥ 3** (plan default; depth supports it).
- **Provider: long-read wins HeLa** (trustworthiness). Salmon only where long-read
  is absent (U2OS, RPE1×3), flagged lower-confidence.
- **Tiered union, not flat** — Tier 3 (salmon-only) carried but flagged.
- **Source precedence: long-read → short-read → sequence-rep.**

---

## 10. Gotchas / things that will bite you

- **A real bug we fixed:** the window-sweep `pure()` helper treated `None`
  (identical-to-±500) as *not* pure because `None`→`NaN` in the DataFrame. It
  undercounted the ~844 fully-identical TIS (reported "0% at ±500", should be 8.5%).
  Fixed (`pd.isna(r) or r > w`). Deliverable numbers were never affected (they use
  `purity_decision` on `None` directly). If you write new sweep code, watch this.
- **Two grains, ~4-TIS offset.** The deliverable extracts windows at ±100; the
  multi-W sweep at ±500, so the sweep can see a divergence *exactly* at radius 100
  that the deliverable can't. W=100 differs by ≤4 TIS between the two — negligible,
  documented in the comparison doc.
- **`build_hela_tis_dataset.py` is HeLa-hardcoded** (paths, salmon reps). Generalize
  before reusing for another line. `disambiguate_expression.py` is already
  provider/threshold-parameterized; `candidate_mrna_divergence.py` takes
  `--cell-line` but its sbatch hardcodes HeLa.
- **conda isn't on PATH.** Activate with
  `source /lab/barcheese01/$USER/miniforge3/etc/profile.d/conda.sh && conda activate swissisoform-v2`
  (long-read steps use the `isoquant` env).
- **Long-read is cross-lab, untreated.** Use it as *existence* evidence; keep salmon
  TPM as the condition-matched abundance for the efficiency denominator.

---

## 11. Reproduce from scratch

```bash
source /lab/barcheese01/$USER/miniforge3/etc/profile.d/conda.sh && conda activate swissisoform-v2

# Long-read: pull → align → IsoQuant (Slurm; ~40 h; resume if it times out)
sbatch scripts/setup/run_longread_hela.sbatch
#   if it times out: sbatch scripts/setup/run_longread_hela_isoquant_resume.sbatch

# 1. candidate mRNAs + sequence purity (Slurm)
sbatch scripts/analysis/run_candidate_mrna_divergence.sbatch
# 2. disambiguation per arm (head node)
python scripts/analysis/disambiguate_expression.py --provider salmon --min-tpm 0.1
python scripts/analysis/disambiguate_expression.py --provider isoquant \
  --isoquant-table data/reference/longread/isoquant_HeLa/OUT/OUT.transcript_counts.tsv --min-counts 3
# 3. window sweeps incl. long-read + 3-way line (Slurm)
sbatch scripts/analysis/run_window_methods_comparison.sbatch
# 4. datasets (head node)
python scripts/analysis/build_hela_tis_dataset.py
python scripts/analysis/build_union3_dataset.py
# 5. figures + report
python scripts/analysis/plot_source_transcripts.py --cell-line HeLa
python scripts/analysis/make_source_transcript_report.py
```

---

## 12. Next steps

**Immediate / unblocking:**
1. **Generalize the dataset builders** to non-HeLa lines (parameterize
   `build_hela_tis_dataset.py` / the candidate-mRNA sbatch by cell line + salmon
   paths + `present_{line}`).
2. **K562 long-read** — ENCODE PacBio (TALON) exists; pull + IsoQuant, run the same
   benchmark. K562 is the second line with both arms → confirms whether the HeLa
   provider decision generalizes.
3. **Process the other 4 lines** (U2OS, RPE1 Async/Que/Sen) — short-read salmon only
   (no public long-read). Apply sequence-purity + salmon, flagged lower-confidence.

**Phase 3–5 (from `source_transcript_resolution.md`):**
4. **CAGE 5′ grounding** (`sourceseq/cage.py`) — snap/validate transcript TSS against
   lab CTSS clusters (path still pending). Adds a 4th narrowing signal.
5. **Per-transcript efficiency** (`sourceseq/efficiency.py`) — formalize TIE
   (currently `norm_TIS / source TPM`); decide the denominator policy (salmon
   condition-matched vs long-read).
6. **Fold into the core pipeline** (`runner.py`) — rewrite `all_transcripts` to the
   resolved set so downstream annotation modules use the disambiguated source.
7. **Export the 6-cell-line dataset** (`build_source_transcript_dataset.py`).

**Analysis questions worth chasing:**
- Validate the **Tier-3 "dilution artifact" hypothesis** directly — for the 1,203
  salmon single-source calls long-read contradicts, are salmon's picks the
  dominant-isoform errors we suspect? Spot-check a few against the BAM.
- A **relative/dominant-isoform salmon rule** (vs absolute TPM) — the exploratory
  sweep suggested it resolves ~59%; wire it in and re-benchmark vs long-read.
- The **2,495 still-unresolved** multi-candidate TIS — are they genuinely ambiguous
  (co-expressed divergent isoforms, correctly discarded) or would CAGE/larger depth
  rescue some?

---

## 13. Provenance

- **Date:** session 2026-06-16 → 2026-06-20. **Cell line:** HeLa.
- **Env:** `swissisoform-v2` (Python 3.11.15) + `isoquant` (IsoQuant 3.13.0).
  Miniforge at `/lab/barcheese01/$USER/miniforge3`.
- **Key jobs:** salmon `10177954`; long-read `10178027`(bug)→`10180797`(resume,
  COMPLETED ~40 h); sweeps `10181103/10181384/10182033/…/10190524`.
- **Salmon QC:** 69.4%/69.6% mapping. **Long-read QC:** 95.9% mapping.
