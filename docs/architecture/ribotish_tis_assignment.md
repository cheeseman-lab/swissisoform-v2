# Ribo-TISH: How TIS Reads Are Assigned to Transcripts — Current State

> **Scope:** Describes how **Ribo-TISH** (the upstream TIS caller that produces our
> `*_TIS_predict_all.txt` inputs) maps ribosome-profiling reads to annotated
> transcripts and emits per-TIS rows. This documents the upstream tool *as it
> exists*, to explain where the `Tid`/`all_transcripts` in our pipeline come from.
> No proposed changes. Pairs with [`upstream_filtering_and_dedup.md`](upstream_filtering_and_dedup.md).
> **Last verified:** 2026-06-16.

**Source:** `https://github.com/zhpn1024/ribotish.git` @ `d721b7a` (cloned to
`/lab/barcheese01/eblack/ribotish`; GPL-3.0). All `file:line` refs below are
relative to `ribotish/src/`. Re-clone:

```bash
git clone https://github.com/zhpn1024/ribotish.git /lab/barcheese01/eblack/ribotish
```

---

## 0. TL;DR

- Ribo-TISH `predict` is **annotation-driven**: it tests **every transcript in the
  GTF you pass** (`-g`), gene by gene. It does **not** assemble transcripts, and it
  takes **no RNA-seq input**.
- A read is "assigned" to a transcript by **splice-junction compatibility** +
  **P-site offset projection** into that transcript's cDNA coordinates — not by
  sequence matching.
- **Ambiguity is not resolved.** A TIS compatible with N transcripts produces **N
  output rows**, one `Tid` each (often identical `GenomePos`). This is exactly why
  our downstream `all_transcripts` / `dedupe_*` collapse exists.
- "Which transcripts exist in the cell" is **not** inferred from RNA-seq. The
  transcript catalog is the static GTF; cell-specificity comes only from which
  annotated transcripts carry compatible *ribosome-profiling* reads.

---

## 1. Inputs and iteration

```bash
ribotish predict -t ltm.bam -b chx.bam -g gene.gtf -f genome.fa -o pred.txt
#                └TIS riboseq┘ └CHX riboseq┘ └GTF annot┘ └genome┘
```
Inputs (`run/predict.py:9-13`): TIS-enriched riboseq BAM (`-t`, LTM/harringtonine),
ordinary riboseq BAM (`-b`, CHX), gene annotation (`-g`), genome FASTA (`-f`).
**No RNA-seq parameter exists.**

Iteration (`run/predict.py:226`, `:390`):
- `io.geneIter(args.genepath, …)` streams genes from the annotation.
- `_pred_gene` loops `for t in g.trans` — **every annotated transcript** of the
  gene. Transcripts shorter than `minTransLen = 50` nt are skipped
  (`zbio/ribo.py:14`, `run/predict.py:396`).

The transcript universe is whatever GTF is supplied. (Our pipeline supplies
GENCODE v49 — so it is the full standard annotation, not a cell-specific one.)

---

## 2. Read → transcript assignment (the core)

Reads arrive **already genome-aligned** in the BAM (alignment is done upstream by
the read aligner; Ribo-TISH only consumes the BAM). For each transcript `t`:

### 2.1 Splice-junction compatibility filter
`zbio/bam.py:478` `transReadsIter(bamfile, t, compatible=True, mis=compatiblemis)`
(and the cached path `transCounts`, `bam.py:567/609`) keep a read only if
`read.is_compatible(t, mis)` → `zbio/interval.py:207` `is_compatible`. That
compares the read's aligned blocks (from its CIGAR) against the transcript's exon
blocks over their overlap and **rejects the read if the symmetric block
difference exceeds `mis` bases** (default **2**).

- A read crossing an exon-exon junction the transcript lacks → incompatible → not
  counted for that transcript.
- CLI knobs (`run/predict.py:55-56`): `--nocompatible` disables the check entirely;
  `--compatiblemis` sets the tolerance.

### 2.2 P-site offset → cDNA codon
`zbio/ribo.py:55-81` (`Ribo.__init__`):
```python
off = offset(r, offdict)                  # ribo.py:24  P-site offset by read length (default 12 nt)
i   = trans.cdna_pos(r.genome_pos(off))   # ribo.py:76  genomic P-site → transcript cDNA index
self.cnts[i] += 1
```
`cdna_pos` (`zbio/interval.py:243`) walks the exon structure to convert a genomic
coordinate into a spliced **cDNA index**. The result `self.cnts` is the per-codon
P-site count profile *for that transcript*. The same genomic read can land in
several transcripts' profiles.

> **"Assigning a read to a transcript" = splice-compatibility + P-site projection
> into the transcript's cDNA frame.** It is alignment-geometry-based, not
> sequence-identity-based.

---

## 3. TIS calling on the transcript

For each transcript (`run/predict.py:413-497`):

1. **Spliced sequence** — `tsq = genome.transSeq(t)` (`predict.py:419`).
2. **ORF enumeration** — `orf.orflist(tsq, minaalen, …)` finds stop-to-stop ORFs
   and their candidate starts: ATG by default, plus near-cognate codons
   (CTG/GTG/TTG/ACG/…) when `--alt` is set (`zbio/orf.py:9-12`, `predict.py:450-453`).
3. **Per-start statistics** —
   - **TIS enrichment test**: `ttis.tis_test(tis, paras[ip][0], paras[ip][1])`
     (`ribo.py:315`) — negative-binomial test that the start codon position is
     enriched in the TIS (LTM/harr) track, against a background whose parameters
     are chosen by the transcript's density bin (§4).
   - **Frame test**: `tribo.frame_test(tis, stop)` (`ribo.py:104`) — in-frame vs
     out-of-frame P-site counts across the ORF (or `enrich_test` with `--enrichtest`).
   - **Combined**: Fisher's method over (TIS p, frame p) (`predict.py:483`).
   Thresholds: `--tpth`, `--fpth`, `--minpth`, `--fspth`, `--fsqth`
   (`predict.py:38-42`).
4. **TIS type** — `tisType(tis, stop, cds1, cds2)` (`predict.py:559-571`), relative
   to **that transcript's own annotated CDS** (`cds1=t.cds_start`, `cds2=t.cds_stop`,
   `predict.py:417-418`): `0 Annotated, 1 Truncated, 2 Extended, 3 5'UTR, 4 3'UTR,
   5 Inside, 6 Novel`.

### Output row — `getResult` (`run/predict.py:504-523`)
- `Tid = t.id` — the **specific annotated transcript** under test.
- `GenomePos` — cDNA start/stop mapped **back** to the genome via
  `t.genome_pos(tis)` / `t.genome_pos(stop)` (`predict.py:507-510`); format
  `chr:lo-hi:strand`.
- `TisType`, `AALen`, optional `Seq`/`AASeq`/`Blocks`.

The output column header is assembled at `predict.py:283-291`
(`Gid Tid Symbol GeneType GenomePos StartCodon Start Stop TisType …`), which is the
21-column `predict_all.txt` our pipeline reads.

---

## 4. Background / "expression" — riboseq, not RNA-seq

TIS significance is calibrated against a background that depends on transcript
**ribosome-profiling density**, not RNA-seq:
- `estimateTISbg` fits negative-binomial parameters per density quantile
  (`run/predict.py:139-163`).
- Transcripts are binned into `--nparts` quantiles (default 10) by `abdscore`
  (= log reads/kb, `zbio/ribo.py:91`); a transcript's bin index is `pidx(score, slp)`
  (`predict.py:413-415`).

So a transcript with no compatible riboseq reads simply yields zero counts and no
TIS calls — but it is **never removed using RNA-seq evidence.**

---

## 5. Ambiguity between transcripts — reported, not resolved

- **Per-transcript reporting.** Each annotated transcript is tested independently
  (`for t in g.trans`, `predict.py:390`) and emitted separately. A TIS at one
  genomic position compatible with N transcripts → **N rows**, each with its own
  `Tid`, frequently sharing one `GenomePos`. Reads compatible with multiple
  transcripts are counted toward each. There is **no "best transcript" selection.**
- **Post-hoc labeling only** (not dedup): after prediction, gene-wide `known_tis`
  and 3-frame `cds_regions` are built, and each TIS is tagged `:Known` (matches an
  annotated start anywhere in the gene) or `:CDSFrameOverlap` (overlaps a known CDS
  in-frame) (`predict.py:258-273`).
- **Genome multi-mapping** is a separate concern, bounded by `--maxNH` (default 5;
  reads aligning to >5 loci dropped) — unrelated to transcript ambiguity.

### Why this matters downstream
This per-transcript redundancy is the direct cause of the `all_transcripts` /
`n_transcripts` columns and the `dedupe_unique_proteins` / `dedupe_unique_init_sites`
collapses in our pipeline (see [`upstream_filtering_and_dedup.md`](upstream_filtering_and_dedup.md)
§3.9): they merge the multiple transcript rows Ribo-TISH emits for one shared
protein or genomic initiation site.

---

## 6. Correcting a common misconception

> "How does Ribo-TISH annotate the transcripts present in a cell from short-read RNA-seq?"

It does not. There is **no RNA-seq step** in `predict`. The transcript catalog is the
**static GTF** (`-g`); "presence" is only ever the existence of *ribosome-profiling*
reads compatible with an annotated transcript. To get a cell-line-specific
transcriptome you would assemble one from RNA-seq (e.g. StringTie) and pass it as
`-g` — but the standard workflow (and ours) uses GENCODE. In our pipeline, RNA-seq
is used **only** as the gene-level RPM normalization denominator, downstream of
Ribo-TISH (see [`upstream_filtering_and_dedup.md`](upstream_filtering_and_dedup.md) §3.4).

---

## 7. Source-file index (ribotish/src/)

| Concern | File:line |
|---|---|
| CLI / inputs / per-gene-per-transcript loop | `run/predict.py:9-13`, `:226`, `:390` |
| Read→transcript compatibility filter | `zbio/bam.py:478` (`transReadsIter`), `:567/609` (`transCounts`) → `zbio/interval.py:207` (`is_compatible`) |
| P-site offset + cDNA projection | `zbio/ribo.py:24` (`offset`), `:55-81` (`Ribo.__init__`), `zbio/interval.py:243` (`cdna_pos`) |
| ORF enumeration / start codons | `zbio/orf.py:9-12`, `run/predict.py:450-453` |
| TIS enrichment / frame / Fisher tests | `zbio/ribo.py:315` (`tis_test`), `:104` (`frame_test`), `run/predict.py:483` |
| Background density binning | `run/predict.py:139-163`, `zbio/ribo.py:91` (`abdscore`), `run/predict.py:413-415` (`pidx`) |
| TIS-type vs annotated CDS | `run/predict.py:559-571` (`tisType`), `:417-418` (`cds_start`/`cds_stop`) |
| Output row / GenomePos mapping | `run/predict.py:504-523` (`getResult`), `:283-291` (header) |
| Known-CDS post-hoc tagging | `run/predict.py:258-273` |
