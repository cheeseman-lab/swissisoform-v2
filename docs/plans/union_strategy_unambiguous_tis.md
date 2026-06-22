# Strategy: maximize TIS sites with an unambiguous source-mRNA sequence

> Strategy memo, dated **2026-06-17**, verified against the HeLa salmon run
> (`10177954`) and the candidate-mRNA/window analyses run the same day.
> Companion to `docs/plans/source_transcript_resolution.md`; consumes the
> `sourceseq` Phase-1 core. The long-read arm (Arm B, IsoQuant job `10180797`)
> was still quantifying when this was written — the expression numbers here are
> salmon-only and will be re-scored against long-read.

## Goal

For the translation-initiation-efficiency model, every kept TIS must be paired
with **one** mRNA sequence around its start codon. A TIS that Ribo-TISH assigned
to several candidate transcripts is only usable if we can pin it to a single,
unambiguous local sequence. **This memo's objective: keep the *largest* number
of TIS that have an unambiguous source-mRNA sequence**, while being honest about
how confident each call is.

## Two resolution mechanisms

A multi-candidate TIS can be made unambiguous in two independent ways:

1. **Sequence-window purity** — if every candidate transcript is *byte-identical*
   within ±W nt of the start codon, the local sequence is unique regardless of
   which isoform a footprint came from. No expression data needed. (We don't
   resolve *which* isoform; we don't have to.)
2. **Expression resolution** — if cell-line RNA-seq shows only **one** candidate
   is expressed, that candidate is the source; if several are expressed but agree
   in-window, they're interchangeable. This needs an expression provider
   (short-read salmon / long-read IsoQuant).

### Why they are complementary (the key insight)

- Sequence purity rescues TIS where **all** candidates agree — even if several are
  expressed. It fails when candidates diverge in-window.
- Expression rescues TIS where candidates **diverge** in-window but only **one** is
  actually expressed. It fails when several expressed candidates still diverge,
  or when none clears the abundance threshold (dilution / low expression).

Each covers the other's blind spot, so their **union** strictly dominates either
alone.

## Evidence (HeLa, 9,920 multi-candidate TIS, init-site grain)

Coordinate math validated first: **0 sequence cross-check failures** across 32,889
candidates (skeleton-reconstructed mRNA == transcriptome FASTA).

**Sequence purity decays steeply with window length** (first-divergence radius
median 53 nt):

| ±W (nt) | 5 | 25 | 50 | 100 | 500 |
|---|---|---|---|---|---|
| sequence-pure | 93.7% | 73.8% | 55.3% | 35.0% | 8.5% |

(does not reach 0% — 844 sites stay identical out to ±500)

**Expression (salmon) is a window-independent floor and threshold-sensitive:**

| salmon presence rule | resolved |
|---|---|
| TPM ≥ 1 in all reps (stringent) | 22.3% (net-negative — dilution drops 75%) |
| TPM ≥ 0.1 in all reps | 41.8% (floor 38.1% = single expressed survivor) |
| dominant-isoform (≥10% of per-TIS max) | 59.4% |

The two cross at **W ≈ 78 nt**: below it sequence wins, above it expression wins.

**The union beats either mechanism (N=9,920):**

| ±W (nt) | sequence-pure | expression-resolved | **union** | union gain |
|---|---|---|---|---|
| 25 | 73.8% | 46.1% | **84.7%** | +1,078 |
| 50 | 55.3% | 44.0% | **73.0%** | +1,755 |
| 100 | 35.0% | 41.8% | **60.3%** | +1,837 |

## The strategy

**1. Union, don't choose.** Keep a TIS if its candidates are sequence-identical
over the needed window **OR** expression collapses it to a single source. Biggest
single lever (+2,000 TIS at ±50).

**2. Per-TIS adaptive window, not a global W.** Store each TIS's first-divergence
radius and keep it to the *maximal clean window it honestly has* — don't discard a
TIS with 40 nt of clean context just because a global W=100 was set. Adaptive
≥25 nt clean context keeps 74.9% by sequence alone; **union with expression →
8,462 (85.3%)** of the multi-candidate set.

**3. Use the shortest window the model truly needs.** Window length is a
context-vs-count dial; ±25–50 nt keeps far more TIS than ±100.

**4. Relative/dominant-isoform expression rule, not absolute TPM.** Absolute
thresholds collide with per-isoform dilution (near-identical isoforms split reads
below threshold). The dominant-isoform rule resolved 59% vs 35% baseline. Validate
against long-read where it exists (HeLa, K562); fall back to salmon-relative on the
four lines without long-read.

**5. Tier by confidence; discard nothing prematurely.**
- **Tier 1** — sequence-pure (isoform identity irrelevant). Highest confidence.
- **Tier 2** — expression-resolved, source ≥ 1 TPM (or long-read confirmed).
- **Tier 3** — expression-resolved, sub-1-TPM source (salmon-only). At TPM ≥ 0.1,
  **49% of resolved calls (2,044/4,146)** fall here — keep, but flag.

**6. Single-candidate TIS are free.** The ~61k single-candidate TIS (genome-wide)
are unambiguous by construction — this whole analysis only concerns the hard
multi-candidate ~13k.

**7. Long-read is the multiplier.** Arm B should both *validate* the low-confidence
salmon calls and *recover* dilution-lost TIS (lifting the expression floor and the
union). Defer the final expression-rule choice to that head-to-head.

## Pros and cons

**Pros**
- Maximizes retained TIS — union + adaptive window keeps ~85% of multi-candidate
  sites at ≥25 nt context vs ≤75% for either mechanism alone.
- Correctness is guaranteed for Tier 1 by construction (identical sequence).
- Adaptive window wastes no honestly-clean context; the model can use
  variable-length input or threshold on a minimum.
- Confidence tiering pushes the precision/recall decision downstream to modeling
  rather than hard-discarding now.
- Provider-agnostic: the expression half swaps salmon↔long-read with one flag.

**Cons / risks**
- Tier 3 (sub-1-TPM, salmon-only) calls are genuinely uncertain — the surviving
  isoform may be a minor one while the true source was diluted below threshold.
  *Mitigation:* tier flag + long-read validation.
- Expression resolution to a single isoform is a statistical estimate, not proof —
  salmon's EM can misattribute among near-identical isoforms. *Mitigation:* the
  dominant-isoform rule + long-read.
- Adaptive (variable-length) windows complicate any model that wants fixed-width
  input; a fixed Wmin cutoff trades some TIS for uniformity.
- Long-read coverage exists for only 2 of 6 lines, so the four uncovered lines
  remain salmon-only (lower confidence) for the expression half.
- Cross-lab long-read (other labs' HeLa) is existence evidence, not
  condition-matched abundance — keep salmon as the efficiency denominator.

## Code used

All reusable logic lives in `src/swissisoform/sourceseq/`; the analyses are
drivers under `scripts/analysis/`.

### Core library (`sourceseq`)

| Function | File:line | Role |
|---|---|---|
| `build_transcript_mrna` | `sourceseq/mrna.py:61` | spliced mRNA from exon skeleton + genome (cross-check vs FASTA) |
| `start_codon_index` | `sourceseq/mrna.py:83` | cDNA index of the start codon (genomic→mRNA) |
| `extract_tis_window` / `TisWindow` | `sourceseq/mrna.py:122` / `:101` | ±W window anchored at the start-codon A |
| `divergence_radius` | `sourceseq/purity.py:44` | smallest radius where candidates first disagree |
| `purity_decision` / `PurityResult` | `sourceseq/purity.py:73` / `:22` | keep/discard + reason + divergence_nt |
| `expressed_in_replicates` | `sourceseq/expression.py:99` | salmon presence (TPM ≥ τ in all reps) |
| `load_salmon_replicates` | `sourceseq/expression.py:46` | mean TPM per transcript (confidence tag / efficiency denom) |
| `load_isoquant_abundance` | `sourceseq/expression.py:61` | long-read presence (Arm B), same interface |
| `load_exon_skeletons` | `io/gtf.py:184` | per-transcript exon structure from the GTF |

The purity test, verbatim (`sourceseq/purity.py`):

```python
def divergence_radius(windows: list[TisWindow], w: int) -> int | None:
    for r in range(1, w + 1):
        downstream = [wn.downstream[r] if r < len(wn.downstream) else None for wn in windows]
        if not _all_equal(downstream):
            return r
        upstream = [wn.upstream[-r] if r <= len(wn.upstream) else None for wn in windows]
        if not _all_equal(upstream):
            return r
    return None
```

### Drivers (`scripts/analysis/`)

| Script (+ sbatch) | What it does | Output |
|---|---|---|
| `candidate_mrna_divergence.py` (`run_candidate_mrna_divergence.sbatch`) | pull candidate mRNAs from `transcripts.fa`, window them, run purity | `{cell}_tis_window_summary.parquet`, `{cell}_tis_candidate_detail.parquet` |
| `disambiguate_expression.py` | filter candidates by expression (`--provider salmon`/`isoquant`), re-run purity, **tag source TPM** | `{cell}_disambiguation_{provider}.parquet` |
| `window_purity_vs_length.py` (`run_window_purity_vs_length.sbatch`) | sweep W=5..500, per-TIS divergence radius, % + # curves | `{cell}_purity_vs_window{.csv,_radius.parquet,.png}` |
| `plot_source_transcripts.py` | render all figures from on-disk outputs (no GTF walk) | `data/figures/source_transcripts/*.png` |

### The union + adaptive selection (the strategy, as computed)

The whole strategy reduces to a per-TIS divergence radius (`r_seq`) plus an
expression-survivor count/radius (`n_surv`, `r_expr`), both already produced by
`window_purity_vs_length.py` into `{cell}_purity_vs_window_radius.parquet`:

```python
def pure(r, w):                      # identical to MAXW (None) or first divergence beyond w
    return (r is None) or (pd.notna(r) and r > w)

seq_ok  = c.r_seq.map(lambda r: pure(r, W))                       # mechanism 1
expr_ok = (c.n_surv == 1) | ((c.n_surv >= 2) & c.r_expr.map(lambda r: pure(r, W)))  # mechanism 2
keep    = seq_ok | expr_ok           # UNION

# adaptive: keep each TIS to its own clean radius instead of a global W
clean_radius = c.r_seq.fillna(MAXW)  # nt of unambiguous context this TIS honestly has
```

## Reproduce

```bash
source /lab/barcheese01/$USER/miniforge3/etc/profile.d/conda.sh
conda activate swissisoform-v2

# 1. candidate mRNAs + purity (Slurm: GTF skeleton load)
sbatch scripts/analysis/run_candidate_mrna_divergence.sbatch

# 2. expression disambiguation + source-TPM tags (head node; salmon TPM>=0.1)
python scripts/analysis/disambiguate_expression.py --provider salmon --min-tpm 0.1

# 3. window-length sweep + curve/figure (Slurm)
sbatch scripts/analysis/run_window_purity_vs_length.sbatch

# 4. figures
python scripts/analysis/plot_source_transcripts.py --cell-line HeLa
```

## Next steps

1. Re-run step 2 with `--provider isoquant` once Arm B lands; recompute the union
   and overlay long-read on `HeLa_purity_vs_window.png`.
2. Decide the expression rule (relative/dominant vs absolute) from the
   salmon-vs-long-read benchmark.
3. Implement union + adaptive-window + confidence-tier selection as the dataset
   builder (the `narrow.py` funnel / export driver in
   `docs/plans/source_transcript_resolution.md`), and run across all six lines.
