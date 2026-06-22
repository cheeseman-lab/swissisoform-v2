# Disambiguation method comparison — how many TIS get an unambiguous mRNA

> **Purpose.** A durable, self-contained record of *what was tried* to disambiguate
> each Ribo-TISH start site to a single source-mRNA sequence, *the numbers each
> method gave*, *why the union strategy wins*, and *how we will fold in the
> long-read arm*. Written so a future session (or collaborator) can pick this up
> cold. Verified 2026-06-17 on HeLa. Pairs with the strategy memo
> `docs/plans/union_strategy_unambiguous_tis.md` and the workstream plan
> `docs/plans/source_transcript_resolution.md`.

## TL;DR

For each TIS that Ribo-TISH assigned to **multiple** candidate transcripts, we
need to collapse the candidates to **one** unambiguous mRNA sequence around the
start codon (else the site can't train a sequence→efficiency model). Three
methods, on **9,920 HeLa multi-candidate TIS**, at a **±100 nt** window:

| Method (W = 100) | TIS resolved | % of 9,920 |
|---|---|---|
| Sequence-window purity alone | 3,474 | 35.0% |
| Salmon expression alone (TPM ≥ 1, both reps) | 2,210 | 22.3% |
| Salmon expression alone (TPM ≥ 0.1, both reps) | 4,146 | 41.8% |
| **Union (sequence-pure OR expression-resolved @0.1)** | **5,986** | **60.3%** |

**The union wins decisively** — +1,840 TIS over the best single method (+44%
relative). It is *not* redundant: `5,986 = 3,474 sequence-pure + 2,512 resolved
only by expression`. Each method rescues sites the other cannot.

## Universe and grain (read this before trusting any number)

- **Cell line:** HeLa (the only line with both expression arms; pilot).
- **Grain:** *genomic initiation site* (`chrom:gstart:strand:codon`), i.e. "a TIS
  in Ribo-TISH." Transcripts sharing one start are the candidate set.
- **HeLa init sites detected:** 17,779. Of these:
  - **7,859 single-candidate** — one transcript → mRNA known trivially, *not* part
    of this comparison (they need no disambiguation).
  - **9,920 multi-candidate** — the test universe for every number here.
- **Two denominators appear in our outputs — don't conflate them:**
  - **`N = 9,920` multi-candidate** (the deliverable grain, `build_hela_tis_dataset.py`).
    A site whose candidates all yield one valid window counts as `single_candidate`
    → trivially pure.
  - **`N = 9,919` testable** (the window-sweep grain, `window_purity_vs_length.py`),
    which requires **≥2** candidates with a valid window — so it *excludes* the
    ~848 single-windowed-candidate sites. This is why sequence-only purity at W=100
    reads **3,474 (35.0%)** on the deliverable grain but **2,626 (26.5%)** on the
    testable grain. Same data, stricter universe.

## Methods tried

### 1. Sequence-window purity (no expression)
Build each candidate's spliced mRNA, anchor at the start codon, and keep the TIS
iff **all candidates are byte-identical within ±W nt**. No expression data; highest
confidence (correct by construction); identity-agnostic (we never learn *which*
isoform, and don't need to). Decays with W (see "Window length" below).
*Coordinate math validated: 0/32,889 candidates failed the reconstructed-mRNA ==
transcriptome-FASTA cross-check.*

### 2. Expression resolution (salmon, short-read)
Keep only candidates **expressed in HeLa** (salmon TPM ≥ τ in both reps), then
re-run purity on the survivors. A site resolves if it collapses to one expressed
isoform, or to several that agree in-window. Window-independent for the
single-survivor case.

**Threshold sensitivity (expression-only resolution, W=100, N=9,920):**

| salmon presence rule | resolved | note |
|---|---|---|
| TPM ≥ 1, both reps (plan default) | 2,210 (22.3%) | **net-negative vs sequence baseline** |
| TPM ≥ 0.1, both reps | 4,146 (41.8%) | operating point we adopted |
| dominant-isoform (≥10% of per-TIS max) ‡ | ≈5,900 (≈59%) | candidate, not yet wired as default |

‡ from an exploratory sweep using a *mean-TPM* rule; recompute with the per-rep
rule before adopting. **Why TPM ≥ 1 fails:** near-identical isoforms split salmon's
read mass (the equivalence-class dilution problem), so real isoforms fall below 1
TPM — the stringent cut drops *all* candidates at 7,462 sites, resolving fewer than
sequence purity alone. Loosening to 0.1 recovers most of that.

### 3. Union (the adopted strategy)
Keep a TIS if it is sequence-pure **OR** expression-resolved. Rationale and full
treatment in `docs/plans/union_strategy_unambiguous_tis.md`.

## Why the union is better (the decomposition)

At W=100, the two methods overlap only partially:

```
sequence-pure ............. 3,474
expression-only (new) ..... 2,512   ← sites where candidates DIVERGE in-window
                                       but only one is expressed
                            ------
union ..................... 5,986   (60.3%)
unresolved ................ 3,933   (39.7%)
```

- **Sequence handles "all candidates agree"** — even when several are expressed.
- **Expression handles "candidates differ, but only one is real"** — the 2,512
  sites sequence purity must discard.
- They are complementary because they answer different questions (sequence
  *identity* vs cell-line *existence*), so the union strictly dominates either.

## Window length × method (the full sweep)

`window_methods_comparison.py` counts unambiguous TIS for **every method** across
W = 5..500 (deliverable grain, N = 9,920):

| W (nt) | sequence-only | salmon TPM≥1 | salmon TPM≥0.1 | union @≥1 | **union @≥0.1** |
|---|---|---|---|---|---|
| 5 | 9,294 (93.7%) | 2,436 (24.6%) | 4,751 (47.9%) | 9,474 (95.5%) | **9,569 (96.5%)** |
| 10 | 8,865 (89.4%) | 2,431 (24.5%) | 4,724 (47.6%) | 9,150 (92.2%) | **9,318 (93.9%)** |
| 50 | 5,490 (55.3%) | 2,296 (23.1%) | 4,366 (44.0%) | 6,581 (66.3%) | **7,245 (73.0%)** |
| 100 | 3,470 (35.0%) | 2,210 (22.3%) | 4,145 (41.8%) | 4,969 (50.1%) | **5,982 (60.3%)** |
| 200 | 1,927 (19.4%) | 2,154 (21.7%) | 3,959 (39.9%) | 3,745 (37.8%) | **4,981 (50.2%)** |
| 500 | 844 (8.5%) | 2,121 (21.4%) | 3,839 (38.7%) | 2,859 (28.8%) | **4,313 (43.5%)** |

- **Union @0.1 dominates at every W.** At short windows (±5–10 nt) it keeps
  **94–96%** of all multi-candidate TIS unambiguous.
- **Sequence-only** is the steep curve: 93.7% at ±5 → 35.0% at ±100 → **8.5% at
  ±500** — it does *not* reach 0, because **844 sites are identical all the way out
  to ±500** (median first-divergence radius = 53 nt).
- **Salmon-only** is the flat floor (window-independent — it picks one isoform):
  ~22% at TPM≥1, ~39–48% at TPM≥0.1.
- **Sequence and salmon@0.1 cross at W ≈ 78 nt:** below it sequence wins, above it
  expression wins. Shorter windows keep far more sites (a context-vs-count trade).

Figure: `data/figures/source_transcripts/HeLa_methods_vs_window.png` (also embedded
in the HTML report).

> **Note on the 4-TIS offset.** At W=100 this table reads sequence-only 3,470 /
> union 5,982, vs 3,474 / 5,986 in the W=100 tables above. The multi-W sweep
> extracts windows to ±500, so it detects a divergence sitting *exactly* at radius
> 100, which the ±100-extracted deliverable cannot — a negligible boundary effect
> (the ±500 numbers are marginally more correct).

## Confidence tiers and honest caveats

The deliverable tiers every resolved site:

| Tier | TIS | Basis |
|---|---|---|
| 1 — sequence-pure | 3,474 | identical in ±100; isoform identity irrelevant |
| 2 — expression ≥ 1 TPM | 1,329 | one confidently-expressed isoform |
| 3 — expression < 1 TPM | 1,183 | one low-abundance isoform (provisional) |

- **Tier 3 is genuinely uncertain** — half of expression-resolved sources sit below
  1 TPM; the surviving isoform may be a minor one while the true source was diluted
  below threshold. This is exactly what long-read should adjudicate.
- The **±100 window** is unambiguous for all 5,986 resolved; the **full mRNA** is
  additionally unambiguous only for the 2,424 resolved to a *single* transcript.
- "Unresolved" (3,933) is partly correct abstention — Ribo-TISH over-assigns to
  unexpressed isoforms, so many of these *should* be dropped.

## The deliverable

`data/output/source_transcripts/HeLa_tis_disambiguated_union_W100.parquet` — one
row per multi-candidate TIS: metadata, Ribo-TISH read counts, per-transcript
initiation efficiency (TIE = normalized footprints / source TPM-sum), resolution
tier, source transcript, source TPM, and the ±100 nt start-codon mRNA window (+
full mRNA where unambiguous). Combined with the 7,859 single-candidate sites →
**13,845 HeLa TIS with an unambiguous source mRNA**.

## Long-read arm (Arm B) — RESULTS (2026-06-19)

The long-read arm (HeLa ONT, GSE277764 / GSM8529146, in-house minimap2 + IsoQuant
3.13.0 vs GENCODE v49, job `10180797`, ~40 h) is in. Presence rule **counts ≥ 3**
(49,555 transcripts present; `feature_id` is versioned ENST matching candidates
exactly). The head-to-head is **more nuanced than "long-read wins"**, and the
nuance is the interesting part.

**Head-to-head at W=100 (N=9,920):**

| Method | resolved | lost | ambiguous-window |
|---|---|---|---|
| sequence-only | 3,474 (35.0%) | — | — |
| salmon expr-only (TPM≥0.1) | 4,146 (41.8%) | 5,111 | 662 |
| **long-read expr-only (counts≥3)** | **4,978 (50.2%)** | **1,967** | 2,974 |
| union (seq OR salmon) | 5,986 (60.3%) | — | — |
| union (seq OR long-read) | 5,879 (59.3%) | — | — |

**The four findings:**
1. **By expression alone, long-read resolves more (50.2% vs 41.8%) and loses far
   fewer (1,967 vs 5,111).** Dilution recovery is real — one read = one isoform, so
   candidates aren't split below threshold. Mean expressed candidates/TIS: long-read
   **1.51** vs salmon **0.61**.
2. **…but the UNION is essentially tied (59.3% vs 60.3%).** Because long-read keeps
   genuinely co-expressed isoforms, it leaves **2,974** sites `ambiguous_window` vs
   salmon's 662. When several isoforms are *truly* co-expressed and diverge in-window,
   neither arm *should* pick one — long-read correctly reports the ambiguity that
   salmon's dilution hides.
3. **~32% of salmon's single-source calls are likely dilution artifacts.** Of
   salmon's 3,775 `single_candidate` calls, long-read finds **>1 expressed divergent
   isoform in 1,203 (32%)** — i.e. salmon "resolved" them only by dropping a real
   isoform. Those are probably *wrong* picks.
4. **Salmon's low-confidence (Tier-3, sub-1-TPM) calls disagree with long-read 38%
   of the time.** Of 2,044 salmon Tier-3 calls, 1,197 are also long-read-resolved;
   they agree on the source isoform in only **738 (62%)** — the other 459 name a
   *different* single isoform (long-read never called these ambiguous).
5. **Dilution recovery into the union:** of the 3,934 sites the salmon-union leaves
   unresolved, long-read resolves **1,439 (36.6%)**.

**Decision: long-read is the better HeLa provider — not for raw count, but for
trustworthiness.** The two arms resolve ~the same *number*, but long-read's
resolutions rest on direct molecular evidence, it recovers 1,439 dilution losses,
and it exposes that ~a third of salmon's confident single-source picks are
artifacts. Salmon's higher union count is partly inflated by *false* single-source
calls. So the chosen rule:
- **sequence-window purity** = threshold-free backbone everywhere (Tier 1);
- **long-read existence** where available (HeLa, K562) for the expression layer;
- **salmon** only where long-read is absent (U2OS, RPE1×3) — and flagged
  lower-confidence, since on HeLa it mis-resolves ~32% of single-source sites.

Outputs: `HeLa_disambiguation_isoquant.parquet`; long-read line added to
`HeLa_methods_vs_window.{csv,png}`.

## Reproduce

```bash
source /lab/barcheese01/$USER/miniforge3/etc/profile.d/conda.sh && conda activate swissisoform-v2
# 1. candidate mRNAs + window purity (Slurm; loads 3 GB GTF skeletons)
sbatch scripts/analysis/run_candidate_mrna_divergence.sbatch
# 2. expression disambiguation + source-TPM tags (head node)
python scripts/analysis/disambiguate_expression.py --provider salmon --min-tpm 0.1
# 3. window-length sweep + curve/figure (Slurm)
sbatch scripts/analysis/run_window_purity_vs_length.sbatch
# 4. union dataset, figures, report (head node)
python scripts/analysis/build_hela_tis_dataset.py
python scripts/analysis/plot_source_transcripts.py --cell-line HeLa
python scripts/analysis/make_source_transcript_report.py
```

## Provenance

- **Date:** 2026-06-17. **Cell line:** HeLa.
- **Salmon (Arm A):** job `10177954`, GENCODE v49 decoy-aware index, mapping rate
  69.4% / 69.6% across reps, library type `U`. Quant at
  `data/reference/salmon/HeLa_rep{1,2}/quant.sf`.
- **Long-read (Arm B):** job `10180797` (IsoQuant 3.13.0), **still running** at time
  of writing — quantification phase, no transcript table yet.
- **Key outputs** (under `data/output/source_transcripts/`):
  `HeLa_tis_window_summary.parquet`, `HeLa_tis_candidate_detail.parquet`,
  `HeLa_disambiguation_salmon_tpm0.1.parquet`, `HeLa_tis_disambiguated_union_W100.parquet`,
  `HeLa_purity_vs_window.{csv,png}`, `HeLa_purity_vs_window_radius.parquet`,
  `HeLa_methods_vs_window.{csv,png}`, `HeLa_methods_vs_window_radius.parquet`.
- **Correction (2026-06-17):** the window-sweep purity counter initially treated a
  `None` first-divergence radius (candidates identical out to ±500) as *not* pure,
  undercounting the fully-identical core at every W (it reported "0% at ±500" and a
  W≈58 crossover). Fixed in `pure()` (`pd.isna(r) or r > w`); corrected values are
  8.5% at ±500 and a **W≈78 crossover**. The deliverable/union numbers were never
  affected (they use `purity_decision` on `None` directly).
- **Figures:** `data/figures/source_transcripts/`. **Report:**
  `data/reports/source_transcript_resolution_report.html`.
- **Scripts:** `scripts/analysis/{candidate_mrna_divergence,disambiguate_expression,window_purity_vs_length,build_hela_tis_dataset,plot_source_transcripts,make_source_transcript_report}.py`.
