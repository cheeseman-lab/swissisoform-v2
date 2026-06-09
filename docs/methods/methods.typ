#set document(
  title: "SwissIsoform v2 — Methods",
  author: "Matteo DiBernardo",
)
#set page(
  paper: "us-letter",
  margin: (x: 1in, y: 1in),
  numbering: "1",
)
#set text(font: "New Computer Modern", size: 11pt, lang: "en")
#set par(justify: true, leading: 0.65em, first-line-indent: 0em)
#show heading: set block(above: 1.4em, below: 0.8em)
#set heading(numbering: "1.1.1")

#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#align(center)[
  #text(size: 14pt, weight: "bold")[
    Methods: systematic annotation of alternative translation initiation sites
  ]
  #v(0.2em)
  #text(size: 10pt)[Draft — #datetime.today().display()]
]
#v(1em)

#let stage(body, fill: rgb("#e8ecf7")) = box(
  fill: fill,
  stroke: 0.6pt,
  inset: 8pt,
  radius: 3pt,
  width: 100%,
  body,
)

#let mod(name, what, fill: rgb("#e8f5ea")) = box(
  fill: fill,
  stroke: 0.5pt,
  inset: 6pt,
  radius: 2pt,
  width: 100%,
  [#text(size: 8.5pt, weight: "bold")[#name] \ #text(size: 7.5pt)[#what]],
)

#figure(
  {
    set text(size: 9pt)
    set par(leading: 0.4em, justify: false)

    // ── Inputs row ─────────────────────────────────────────────────────
    grid(
      columns: (1fr, 1fr, 1fr),
      column-gutter: 6pt,
      stage(fill: rgb("#f3efe0"))[
        *Ribosome profiling* \
        #text(size: 8pt)[
          Ribo-TISH `predict_all.txt` \
          (6 cell lines, 21 columns/row) \
          raw counts · p-values · AASeq
        ]
      ],
      stage(fill: rgb("#f3efe0"))[
        *RNA-seq depth* \
        #text(size: 8pt)[
          HTSeq-count (gene-level) \
          matched replicates × 6 cell lines \
          → total mapped reads for RPM
        ]
      ],
      stage(fill: rgb("#f3efe0"))[
        *Reference annotation* \
        #text(size: 8pt)[
          GENCODE v49 (GRCh38) \
          primary-assembly FASTA \
          GTF + protein FASTA
        ]
      ],
    )

    v(4pt)
    align(center)[↓]
    v(4pt)

    // ── Upstream per-sample ────────────────────────────────────────────
    stage[
      *1. Per-sample upstream* (§1.1–1.3) #h(1fr) _one invocation per cell line_

      #v(2pt)
      #text(size: 8.5pt)[
        TIS-type recategorization → RPM normalization against total mapped reads →
        reference-transcript pre-filter (MANE Select or TSL ∈ {1, 2, 3}) →
        mark non-reference / low-RPM / non-significant / distance-redundant events →
        impute canonical starts from GENCODE CDS + start_codon + pc_translations →
        drop transcripts still missing a canonical (`cds_start_NF`)
      ]
    ]

    v(4pt)
    align(center)[↓]
    v(4pt)

    // ── Cross-cell-line combine ────────────────────────────────────────
    stage[
      *2. Cross-cell-line combine* (§1.4)
      #v(2pt)
      #text(size: 8.5pt)[
        Union of distinct (gene symbol, transcript, ORF genomic span, start
        codon) tuples across the six samples. Each tuple carries a vector of
        per-cell-line expression metrics; zero counts fill non-observing
        samples. Differential expression is deferred downstream.
      ]
    ]

    v(4pt)
    align(center)[↓]
    v(4pt)

    // ── Gene assembly ──────────────────────────────────────────────────
    stage[
      *3. Gene assembly* (§2)
      #v(2pt)
      #text(size: 8.5pt)[
        One gene record per symbol. For each alternative TIS: select its
        *transcript-specific canonical* (not the gene-level longest),
        derive the *differential region* (N-terminal extension prefix,
        N-terminal truncation prefix, or whole ORF for uORFs / altORFs),
        extract the *13-nt Kozak window* (strand-aware, ATG at
        positions 9–11), and attach per-cell-line expression.
      ]
    ]

    v(4pt)
    align(center)[↓]
    v(4pt)

    // ── Symmetric annotation — modules ─────────────────────────────────
    stage(fill: rgb("#e6f3e8"))[
      *4. Symmetric annotation* (§3–4) #h(1fr) _protein methods run on both canonical and isoform_

      #v(4pt)

      #text(size: 8pt, weight: "bold")[Protein-level methods] #h(0.5em)
      #text(size: 7.5pt)[(run on canonical ×1 per gene AND on isoform ×1 per TIS)]

      #v(3pt)
      #grid(
        columns: (1fr, 1fr, 1fr),
        column-gutter: 4pt,
        row-gutter: 4pt,
        mod[Biophysics][18 descriptors: pI, GRAVY, instability, aromaticity, disorder (TOP-IDP), Shannon / window entropy, complexity, homopolymer, LLPS composite + components],
        mod[Linear motifs][14 ELM-derived SLiM patterns (CDK / ATM / 14-3-3 / SH3 / PIP / RING / ZnF etc.) — positional hits + density],
        mod[Subcellular localization][DeepLoc 2.1 (Fast mode) in isolated env; batched FASTA for amortized embedding cost],
        mod[Clinical variant burden][gnomAD v4.1 + ClinVar + COSMIC v102 via PyArrow pushdown; codon-level re-validation in isoform frame],
        mod[Domain architecture][InterProScan 6 (Pfam etc.) domain hits — positional, with diff-region gain / loss],
        mod[Proteomic detectability][In-silico tryptic digest (≤1 missed cleavage, 7–30 aa), uniqueness flag vs. canonical, optional PepQuery2 hit],
        mod[Signal / targeting peptides][SignalP 6.0 signal-peptide call + TargetP 2.0 SP / mTP / cTP targeting prediction],
        mod[Evolutionary conservation][Zoonomia phyloP (cactus241way) + phastCons (100way) BigWig at TIS codon / Kozak / unique region; HAL reading-frame integrity + AA percent identity],
        mod[Per-variant effect][AlphaMissense + ESM-2 650M masked-marginal ΔLLR per unique-region variant; ESM-2 constraint + gnomAD depletion / disease density per region],
      )

      #v(8pt)

      #grid(
        columns: (2fr, 1fr),
        column-gutter: 6pt,
        align: top,
        [
          #text(size: 8pt, weight: "bold")[Site-level methods] #h(0.5em)
          #text(size: 7.5pt)[(run ×1 per TIS)]
          #v(3pt)
          #grid(
            columns: (1fr, 1fr),
            column-gutter: 4pt,
            mod(fill: rgb("#eef2f5"))[ORF classification][Ribo-TISH label, isoform/canonical lengths, in-frame flag],
            mod(fill: rgb("#eef2f5"))[Kozak strength][Hamming distance to `gccgccRccATGG` under full, major, and partial weighting schemes; GC content],
          )
        ],
        [
          #text(size: 8pt, weight: "bold")[Gene-level methods] #h(0.5em)
          #text(size: 7.5pt)[(×1 per gene)]
          #v(3pt)
          #mod(fill: rgb("#faf0ea"))[Gene reference][HGNC symbol, OMIM associations, GENCODE biotype — attached once per gene]
        ],
      )
    ]

    v(4pt)
    align(center)[↓]
    v(4pt)

    // ── Output ────────────────────────────────────────────────────────
    stage(fill: rgb("#f7e9e7"))[
      *5. Differential annotation table* (§5)
      #v(2pt)
      #text(size: 8.5pt)[
        One row per alternative TIS with identity columns (gene, transcript,
        genomic coords, ORF type, Kozak), per-cell-line expression
        (`expr_{cell_line}_{metric}`), and paired annotation columns
        (`canonical_{method}_{field}`, `isoform_{method}_{field}`).
        Canonical–isoform differentials are derived post-hoc from this table.
      ]
    ]
  },
  caption: [Overview of the SwissIsoform v2 annotation pipeline. Sand-colored
  boxes denote external inputs and reference data; blue denotes per-sample
  and cross-sample preprocessing; green denotes the symmetric canonical /
  isoform annotation layer with its annotation methods grouped by dispatch
  protocol (protein-level methods run on canonical and isoform; site-level
  methods on each TIS; one gene-level method); red denotes
  the final differential-annotation table.],
) <fig:overview>

#v(1em)

= Translation initiation site identification

== Ribosome-profiling data and input preparation

Candidate translation initiation sites (TIS) were identified with
Ribo-TISH @ribotish applied to ribosome-profiling libraries from six
human cell lines (HeLa, K562, U2OS, and RPE1 in asynchronous, quiescent,
and senescent states). Ribo-TISH produced, for each sample, a tab-delimited
table of candidate initiation events in which each row specified a genomic
locus, strand, a supporting transcript identifier, the inferred start codon
and amino-acid sequence of the resulting open reading frame (ORF), together
with its raw read count (TISCounts), a two-sided enrichment p-value
comparing the ribosome footprint pile-up at the candidate codon against
local background, a frame-preference p-value, a Fisher-combined q-value,
and the initial ORF-type classification assigned by Ribo-TISH
(Annotated, Extended, Truncated, 5$prime$UTR, 3$prime$UTR, Internal, or
Novel). To normalize across samples of differing depth, we recoded
Ribo-TISH's composite labels into a controlled vocabulary and divided
each TIS's raw read count by the total number of uniquely-mapped RNA-seq
reads in the matched sample — a single grand-total constant summed over
all genes from the HTSeq-count @htseq table — scaled to reads per million
(RPM).

== Reference-transcript pre-filter and significance filtering

Per-sample processing of the normalized TIS table proceeds as a fixed
sequence of five operations applied in order. First, a permissive
_reference-transcript pre-filter_ selects the set of transcript
identifiers on which any downstream event may reside: a transcript
qualifies if it is MANE Select @mane, or if its GENCODE @gencode
transcript support level is 1, 2, or 3. This pre-filter runs with no
expression or significance thresholds, because its purpose is to set the
universe of permissible transcript models rather than to score events.
Second, every TIS event assigned to a transcript outside this universe
is marked with the drop reason _NotReferenceTranscript_. Third, events
with an RPM below 0.1 are marked _LowReadcounts_. Fourth, events failing
any of three statistical criteria — a TIS-enrichment test
$p lt.eq 0.01$, a frame-preference test $p lt.eq 0.01$, and a Fisher-
combined $q lt.eq 0.05$ — are marked _NotSignificant_. Fifth, a
distance-deduplication pass on the surviving events enforces a minimum
spacing of 30 nucleotides between retained TIS on any given transcript:
within each transcript, surviving events are sorted in descending RPM,
canonical starts (`Annotated` events) are selected preferentially, and
subsequent events whose transcript-relative start position falls within a
30-nucleotide window of a retained event are marked _UpstreamTIS_. The mask
is applied on the raw `Start` column (in plus-strand genomic coordinates),
so on minus-strand genes increasing `Start` is mRNA-upstream of the
retained event. TIS
rows are retained when they carry no drop reason across all five steps;
all thresholds are exposed as configurable parameters.

== Canonical-start imputation and uncanonical-transcript removal

Because the five-step filter can legitimately remove a transcript's own
canonical start (for instance, when an annotated event's re-detection
p-value does not clear the significance threshold in a particular
sample), the downstream comparison may be left without a canonical
reference against which to compute a differential region. To guarantee
that every surviving reference transcript carries a canonical record, we
applied a post-filter _canonical-imputation_ step: for every reference
transcript lacking an `Annotated` row in the filtered output, a
synthetic canonical TIS record was reconstructed from the GENCODE v49
annotation by combining the transcript's CDS coordinates, its
`start_codon` feature, and its protein sequence from the GENCODE protein
FASTA (`pc_translations`). The resulting row inherits the transcript's
coordinates, the trinucleotide of its start codon (extracted from the
primary-assembly GRCh38 FASTA), and the full amino-acid sequence of the
encoded protein, and is tagged with zero-valued read-count fields so
that downstream layers treat it as a reference fixture rather than an
empirical observation (the RNA-seq sample total carried for RPM
normalization is left at its non-zero value). Imputation supersedes — rather than augments — the
alternative convention of exempting `Annotated` rows from significance
filtering, and we run the filter in non-exempting mode to match the
upstream reference implementation exactly.

After imputation, transcripts that still lack an `Annotated` row are
removed. These are transcripts that GENCODE flags as `cds_start_NF`
(coding-sequence 5$prime$ end not defined) or that retain an intron on
a protein-coding gene: by construction their canonical start is not
defined, and any alternative-TIS annotation against them would produce
ill-defined differential coordinates. The output of this upstream
processing is, per sample, a filtered TIS table in which every distinct
reference transcript contributes exactly one `Annotated` row plus zero
or more alternative initiation rows (`Extended`, `Truncated`, `uORF`,
etc.), all in a shared schema.

== Cross-cell-line aggregation

Because each cell line was processed independently through the pipeline
above, a given alternative TIS may be supported in multiple cell lines
with independent expression estimates. We combined the six per-sample
tables by taking the union of distinct
$("gene symbol", "transcript", "ORF genomic span", "start codon")$ tuples
across samples and attaching, to each such tuple, a vector of
cell-line-specific expression metrics (raw count, RPM, one-sided
significance, and initiation efficiency — the ratio of the TIS read count
to the gene's RNA-seq count) carried forward
from whichever sample(s) observed the event. Here the ORF genomic span is
the full start-to-stop coordinate range emitted by Ribo-TISH (`GenomePos`).
Events observed in only a single cell line were retained but flagged with
zero expression in non-observing samples. Cross-cell-line differential
analysis is performed downstream
on this unified table rather than during upstream filtering, so that the
filter logic remains a pure within-sample operation.

= Construction of canonical–isoform pairs

== Gene and TIS domain representation

From the aggregated filtered table we constructed, for each gene symbol,
a structured _gene record_ containing: the gene's GENCODE identifier, a
gene-level representative canonical (defined as the longest annotated
protein among the reference transcripts surviving the upstream filter and
imputation, not over all GENCODE transcripts), the
protein sequence of that representative canonical, and a list of
alternative TIS records. Each TIS record carries the canonical protein of
its _own_ transcript — not the gene-level longest — because Ribo-TISH
classifies ORF type (e.g. `Extended`, `Truncated`) relative to the CDS of
the transcript on which the event was detected; comparing an alternative
start against a transcript whose CDS it does not modify would produce
spurious differential coordinates. Because the upstream canonical-imputation
step guarantees that every surviving reference transcript carries an
`Annotated` row, this per-transcript lookup is always defined; assembly
raises an error rather than substituting a gene-level canonical if a TIS's
own transcript is found to lack one.

== Differential-region derivation

For each alternative TIS we computed a _differential region_: the interval
of the isoform protein that is not shared with its transcript-specific
canonical. The interval was derived directly from the two amino-acid
sequences rather than from coordinates, which allowed the same logic to
handle frame-preserving extensions, frame-preserving truncations, and
out-of-frame or non-coding-frame alternative ORFs without special cases:

- *Extensions* (`extended`): the isoform is longer than
  its canonical; the differential region is the N-terminal segment of
  the isoform spanning positions $0..delta$, where $delta$ is the length
  difference, and the remainder of the isoform matches the canonical's
  full length exactly.
- *Truncations* (`truncated`): the canonical is longer
  than the isoform; the differential region is the N-terminal segment of
  the canonical that is _absent_ from the isoform, spanning the first
  $|delta|$ canonical positions.
- *CDS-frame-overlapping uORFs* (`uoorf`, from Ribo-TISH's
  `CDSFrameOverlap` types): these are first tested for a frame-preserving
  extension- or truncation-like relationship to the canonical by the same
  sequence-matching branch, and are treated as whole-isoform differential
  only when no such relationship is found.
- *Plain uORFs, altORFs, internal out-of-frame ORFs, and 3$prime$UTR ORFs*
  (`uorf`, `alt_orf`, `internal_oof`, `3utr_orf`): no meaningful
  shared frame exists; the entire isoform protein is treated as
  differential.

The derivation also accepts cross-transcript canonicals by verifying the
tail of the shorter sequence against the corresponding tail of the longer
at $gt.eq 95%$ identity over the final $gt.eq 50$ residues (an exact match
is required when fewer than 50 residues remain); failures
trigger a warning and fall back to treating the entire isoform as
differential. This guard protects against silently emitting spurious
differential regions when a TIS's transcript changes frame mid-protein
relative to the gene-level canonical.

== Kozak-context extraction

For each TIS we extracted a 13-nucleotide window spanning positions
$-9$ to $+4$ of the mRNA, with the initiation codon occupying
positions 9 through 11 (0-indexed). Extraction was strand-aware: on plus
strand the window was read directly from the primary-assembly GRCh38
FASTA; on minus strand the reverse-complement of the corresponding
plus-strand interval was taken so that the final string is read
5$prime$ → 3$prime$ in mRNA orientation. This window matches the
consensus encoding `gccgccRccATGG` used for Kozak scoring and is
populated whether the start codon is ATG or a near-cognate (CTG, GTG,
TTG, ACG, AAG); the identity of the trinucleotide at positions 9–11 is
preserved in the TIS record independently of the window itself.

= Symmetric annotation of canonical and isoform proteins

The central operation of the pipeline is the _symmetric application_ of
protein-level annotation methods to both the canonical and the
alternative isoform protein of each initiation event. The principle is
that any method capable of describing a protein property — biophysical,
evolutionary, functional, or clinical — should be run identically on
both sequences, with no a-priori assumption about which end of the
comparison represents the "reference". Canonical annotations are
computed once per gene and cached; isoform annotations are computed once
per TIS. The comparator layer, described in §5, then derives
canonical-versus-isoform differentials post-hoc from these two complete
annotation records.

Methods fall into three dispatch classes, distinguished by the scope of
information each requires:

- *Protein-level methods* receive only an amino-acid sequence, and
  optionally the gene symbol and the canonical protein against which
  uniqueness is to be assessed. Such methods are run twice per TIS:
  once on the canonical protein and once on the alternative isoform
  protein.
- *Site-level methods* receive the full TIS record, including its
  coordinates, ORF-type classification, and Kozak window. They are run
  once per TIS and write only into the isoform-level annotation slot,
  because site-level context (ORF type, initiation-codon identity, Kozak
  strength) is a property of the alternative event rather than of the
  gene.
- *Gene-level methods* receive the gene symbol and any gene-level
  metadata, and write into a gene-level annotation slot independent of
  any individual TIS.

= Annotation methods

== Biophysical descriptors

For each protein sequence we computed eighteen biophysical descriptors
that summarize the global physicochemical character of the polypeptide:
length, theoretical isoelectric point (pI), the Kyte–
Doolittle hydropathy index (GRAVY) @kd, aromatic fraction, the fraction
of charged residues, the instability index, a TOP-IDP disorder propensity
score derived from the intrinsic-disorder amino-acid propensities of
Campen et al. @topidp, the fraction of disorder-promoting residues, and a
family of sequence-complexity descriptors: the whole-sequence Shannon
entropy and its mean over a sliding 20-residue window, a
normalized-complexity score, the fraction of residues in low-complexity
regions (Shannon entropy below 2.2 bits over a sliding 12-residue window),
the amino-acid diversity of the N-terminal window, and the longest
homopolymer run. Liquid-liquid phase-separation (LLPS) propensity is
reported as a composite score together with its three sequence-derived
components — the prion-like-domain content (Q/N/G/S/Y fraction), the
$pi$-$pi$ interaction propensity (F/Y/W/R/H/Q/N), and the
RG/FG-motif density — combined with disorder and low-complexity fraction
under fixed weights. The instability index is computed with BioPython's
ProtParam module @biopython; pI is computed from an EMBOSS-style $p K$
table by binary search; the remaining descriptors requiring residue-level
propensity tables (TOP-IDP, LLPS components, the Shannon-entropy
complexity family) were implemented directly against published
residue-level propensity values. Residue-level vectors suitable for
downstream positional comparisons (per-residue hydropathy and disorder
propensity) are retained alongside the scalar summaries.

== Linear-motif scanning

Short linear motifs (SLiMs) were identified by regular-expression search
against fourteen canonical patterns curated from the ELM database
@elm and earlier SLiM literature: CDK-substrate consensus (`[ST]P`),
ATM/ATR-substrate consensus (`[ST]Q`), 14-3-3 Mode I and Mode II binding
motifs, EB1 SxIP-class microtubule-plus-end tracking, RGG
methylation/RNA-binding motifs, SH3-domain class-I and class-II binding,
heme-regulatory motif (`C[^C].{2}C[^C]H`, a Cys-spaced HRM) and
cytochrome-c heme-binding motif (CXXCH),
PCNA-interacting PIP-box and APIM motifs, C2H2 zinc-finger and RING-
finger E3-ligase consensus patterns. Hits were recorded with their
start and end coordinates within the protein, the literal matching
substring, and the motif name; the summary counts motif-type densities
(hits per residue) and the number of residues involved in overlapping
hits.

== Subcellular localization

Subcellular localization was predicted with DeepLoc 2.1 @deeploc,
operating in its "Fast" mode. Because DeepLoc's runtime dependencies are
pinned to a Python 3.8
environment that conflicts with the rest of the analysis stack, DeepLoc
was executed within an isolated conda environment whose stdout was
consumed by the main pipeline. To amortize the substantial fixed cost of
ESM embedding, all unique canonical and isoform protein sequences in a
batch were written to a single FASTA, DeepLoc was invoked once, and the
per-sequence predictions (primary localization, signal-peptide / NLS /
mitochondrial / peroxisomal sub-predictions, and membrane-type call)
were joined back onto the input records by sequence identity. A
protein-hash-keyed lookup served as a deduplicating memoization layer,
so that identical sequences arising in different genes or cell lines
were predicted exactly once per analysis run. Per-residue ESM embeddings
were not retained; only the final categorical and probabilistic outputs
were written to the annotation table.

== Clinical variant burden and codon-level consequence validation

Clinical annotations drew on three primary variant databases, each
preprocessed from its authoritative source to a columnar parquet with
filter pushdown:

- *gnomAD v4.1 exomes* @gnomad: per-chromosome VCF files were parsed,
  filtered to `PASS` records with a canonical-transcript VEP annotation
  bearing a non-null gene symbol, and projected to a tabular
  representation preserving genomic coordinates, allele identifiers,
  allele frequency, consequence term, HGVSp / HGVSc strings, and
  canonical-frame protein position. Per-chromosome tables were then
  concatenated into a single gene-indexed parquet via streaming
  row-group appends, which bounds peak memory by the largest per-chrom
  partition rather than the full ≈$10^8$-record table.
- *ClinVar* @clinvar: the `variant_summary.txt.gz` table was filtered to
  the GRCh38 assembly. When parsing alleles, we preferred the
  VCF-format columns (`ReferenceAlleleVCF`, `AlternateAlleleVCF`,
  `PositionVCF`) because these are guaranteed to be in genomic
  plus-strand orientation; the non-VCF allele columns and the HGVSc
  title parse are both transcript-direction, which would silently
  mis-validate minus-strand genes. Clinical-significance terms were
  preserved verbatim from the submission.
- *COSMIC v102 (GRCh38)* @cosmic: the Genome Screens Mutant,
  NonCoding Variants, and Complete Targeted Screens Mutant VCF
  distributions were downloaded via the Sanger authenticated API and
  stream-parsed to per-VCF parquet partitions in a single directory
  dataset. Stream parsing writes one RecordBatch per fixed-row-count
  buffer directly to the parquet writer, which avoids accumulating
  hundreds of millions of row dictionaries in memory.

Retrieval for a query gene is a filter-pushdown read: PyArrow's dataset
layer evaluates the `gene_symbol` predicate at the parquet level and
materializes only the matching rows. All three databases return hits in
a unified schema carrying source, variant identifier, chromosome,
1-based genomic position, reference and alternate alleles, consequence
term, raw HGVSp and HGVSc strings, allele frequency where applicable,
and clinical-significance label where applicable.

Because HGVSp strings in the source databases are expressed in the
frame of the canonical transcript, they are _not_ valid indicators of
protein position for alternative TIS events, which may re-frame the
protein entirely (extensions) or eliminate the N-terminus (truncations,
uORFs). We therefore implemented a dedicated codon-level consequence
validator that re-maps every queried variant from its genomic position
to an isoform-specific coding position using the GTF-derived CDS
exon-boundary table for the TIS's transcript. The mapping is
strand-aware: on plus strand, genomic positions within CDS exons are
assigned sequential 0-indexed coding positions in 5$prime$ → 3$prime$
order; on minus strand, exons are walked in descending genomic order
and the genomic positions within each exon are traversed high-to-low,
with the reference and alternate bases complemented before codon
assembly. For single-nucleotide variants, the codon containing the
variant position was extracted from the isoform's reference coding
sequence, translated, and compared with the translation of the mutant
codon to assign a consequence in $\{$`synonymous_variant`,
`missense_variant`, `stop_gained`, `stop_lost`, `reference_mismatch`$\}$;
the loss-of-function gate downstream keys on the exact `stop_gained`,
`stop_lost`, and `frameshift_variant` terms. For
in-frame indels we assigned `inframe_insertion` or `inframe_deletion`
from the length differential; frameshifts were labelled `frameshift_variant`.
Multi-nucleotide substitutions (equal-length reference and alternate
alleles both longer than one base) were assigned the label `mnv` without
a single-codon translation, because a full implementation would require
walking multiple codons and is deferred. Variants outside the isoform's
coding interval — including variants that fall upstream of truncated
TIS starts, downstream of extended stops, or within introns — were
assigned protein position `None`; these variants are _retained_ in the
per-TIS hit list for re-validation against other transcripts rather than
discarded, and are counted in the per-protein clinical summary.

The per-protein clinical annotation therefore consists of: a positional
list of validated variants, each with genomic coordinates, isoform-
specific protein position, isoform-specific amino-acid change, source
database, clinical significance, and population allele frequency; and a
per-protein summary reporting total validated variant count, counts by
source, counts by consequence class, and counts of variants annotated
pathogenic or likely pathogenic in ClinVar.

== Protein-domain architecture

Protein domains were annotated with InterProScan 6 @interproscan, run as a
Nextflow pipeline that aggregates member-database signatures (Pfam and
others) into InterPro entries. Because the per-invocation startup cost of
the pipeline is large, all unique canonical and isoform protein sequences
in a batch were written to a single FASTA, InterProScan was invoked once,
and the resulting hits — each carrying its member database, signature
accession, optional InterPro cross-reference, and start / end coordinates
within the protein — were joined back onto the input records by sequence
identity. The per-protein annotation records the positional hit list and a
summary with a `status` field, so that a protein with zero domains (`ok`,
empty list) is distinguished from a protein whose scan did not complete.
For functional criterion F3 (§5) we count only _real_ functional domains
— hits carrying a genuine InterPro accession, excluding disorder- and
structure-only signatures (MobiDB-lite, coils, low-complexity, SignalP,
Phobius, TMHMM) — and require that such a domain start inside the
differential region and be absent from the other form's domain set,
rather than merely repositioned.

== Signal and targeting peptides

Two further precompute-and-lookup modules annotate N-terminal sorting
signals, whose presence or absence is precisely the kind of property an
alternative N-terminus can change. SignalP 6.0 @signalp6 predicts the
five signal-peptide types using a protein language model; TargetP 2.0
@targetp2 predicts the broader targeting-peptide classes — signal peptide
(SP), mitochondrial transit peptide (mTP), and chloroplast transit peptide
(cTP) — and, where present, the predicted cleavage site. Both tools are
run once over the batch of unique sequences in their own pinned
environments and joined back by sequence identity. For each protein we
record the predicted class, the class probabilities, and the cleavage site;
running SignalP and TargetP together lets functional criterion F4 (§5)
detect a gained or lost targeting signal that either tool alone would miss.

== Evolutionary conservation

Evolutionary conservation is annotated by two complementary site-level
methods that operate in genomic coordinates rather than on the protein
sequence, so that conservation is measured exactly over the isoform's
own open reading frame.

*Nucleotide-level conservation.* The first method reads pre-computed
basewise conservation scores from two BigWig tracks: phyloP
@phylop derived from the Zoonomia 241-mammal Cactus alignment
@zoonomia (the `cactus241way` track), and phastCons @phastcons from the
UCSC 100-vertebrate alignment (`phastCons100way`). For each TIS the method
queries both tracks at the initiation codon (3 nt) and across the 13-nt
Kozak window, and — using the per-ORF genomic exon intervals derived by the
assembly layer's transcript-skeleton walker — computes length-weighted mean
scores over the isoform-unique region, the canonical-shared region, and the
unique/shared enrichment ratio. A status field distinguishes `ok`,
`not_run` (no BigWig track or no configuration), and `no_skeleton` (the
ORF exon intervals were unavailable for the region computation), so that a
genuine zero score is not confused with a missing measurement.

*Reading-frame integrity.* The second method assesses whether the
isoform-unique open reading frame is preserved as a coding frame across
the placental-mammal radiation. The unique-region exon intervals are
extracted from the Zoonomia Cactus alignment HAL with `hal2maf`, and for
each aligned species the method tests start-codon conservation, scans for
frameshifting indels and premature stop codons, and records amino-acid
percent identity. Per-species calls are aggregated into the mean
amino-acid percent identity, the fraction of primate and of mammalian
species with an intact frame, and the
deepest-diverging species (with its phylogenetic depth read from the HAL's
own species tree) whose frame remains intact. The primate and mammalian
mean percent identities feed existence criteria E1 and E2 (§5), with the
intact-frame fraction carried alongside as context. The method
distinguishes `not_run` (no
HAL available), `no_skeleton` / `no_unique_region` (no region to query),
`no_alignment` (the region did not align), and `ok`.

== In-silico proteomic detectability

We assessed the mass-spectrometric detectability of every protein by
in-silico tryptic digestion, cleaving the sequence after every lysine
or arginine residue that is not immediately followed by proline
(KP / RP exceptions) and allowing up to one missed cleavage. Peptides
shorter than 7 or longer than 30 residues — outside the typical tandem
mass-spectrometry observation range — were discarded. Each retained
peptide was recorded with its sequence, start position, end position,
length, and a uniqueness flag indicating whether the peptide sequence
occurs anywhere in the canonical protein's tryptic digest. The
uniqueness flag is the primary handle for downstream experimental
validation: unique peptides are the candidates that could, in principle,
distinguish the alternative isoform from the canonical in a
proteomics experiment. When pre-computed PepQuery2 @pepquery validation
results were provided, peptides already observed in a reanalysis of
public proteomics datasets were additionally flagged as experimentally
validated. For the canonical protein annotation, uniqueness is
undefined (there is no alternative sequence to compare against) and the
flag is recorded as `None` rather than `False`, to distinguish the
semantically-undefined case from an empirically-negative one.

== Per-variant effect scoring

Beyond counting clinical variants, we estimate the predicted functional
effect of each variant that falls in the isoform-unique region, combining
two complementary per-variant predictors and aggregating them over the
region. Each unique-region clinical hit is scored on two independent
axes:

- *AlphaMissense* @alphamissense supplies DeepMind's calibrated missense
  pathogenicity (a 0–1 score plus a `likely_pathogenic` / `ambiguous` /
  `likely_benign` class) looked up by genomic coordinate. AlphaMissense is
  defined in the frame of the canonical transcript, so it applies to
  shared-region and truncation-unique variants but not to
  extension-unique variants, which fall outside the canonical CDS.
- *ESM-2 650M* @esm2 supplies a masked-marginal change in
  log-likelihood, $Delta "LLR" = log P("alt") - log P("wt")$, evaluated at
  the variant's residue from the per-position amino-acid distribution of a
  single full-protein forward pass (cached on disk by protein hash). More
  negative values indicate a substitution less tolerated by the language
  model.

A variant is flagged damaging on either of two independent branches: a
loss-of-function consequence (frameshift, stop-gained, splice, or
start-lost) is damaging on its own — neither missense predictor can see it
— or a missense variant is damaging when AlphaMissense calls it
`likely_pathogenic` or its ESM-2 $Delta "LLR"$ is at or below
$-7.5$, the threshold Brandes et al. @brandes use for the analogous
language-model LLR. Because gnomAD is a population-tolerance catalogue
rather than a disease one, a predicted-damaging gnomAD variant observed at
an allele frequency at or above $10^(-3)$ is gated out of the damaging
flag (ACMG allele-frequency benign evidence); ClinVar and COSMIC variants
are never gated. These per-variant damaging scores are retained as
context, reported separately for germline (gnomAD) and disease
(ClinVar + COSMIC) sources.

The signals that feed functional criteria F5 and F6, however, are
_densities_, not raw counts, so that the comparison between the
isoform-unique and canonical-shared regions is normalized by region size.
A dedicated genomic-intersection method tags each clinical variant by its
membership in the isoform-unique versus canonical-shared exon intervals
(from the assembly layer's transcript-skeleton walker) and exposes the
nucleotide length of each region. From these it derives two ratios:
a germline _depletion ratio_,
$(n_"gnomAD,unique" \/ "nt"_"unique")\/(n_"gnomAD,shared" \/ "nt"_"shared")$,
where a value below one means common germline variation _avoids_ the
unique region (purifying constraint), and a disease _enrichment ratio_,
$(n_"disease,unique" \/ "nt"_"unique")\/(n_"disease,shared" \/ "nt"_"shared")$,
where a value above one means ClinVar / COSMIC variants _concentrate_ in
the unique region. Either ratio is `None` when a denominator is zero or
missing. These two densities, together with the ESM-2 constraint signal
below, are the basis for F5 (germline tolerance / constraint) and F6
(disease density enrichment) respectively (§5); the earlier
damaging-count formulation is superseded.

A separate language-model module (`plm_vep`) records the ESM-2
sequence-_constraint_ profile of the proteins themselves — the mean
wild-type-residue log-probability (a constraint signal, with no alternate
allele) over the isoform-unique and canonical-shared regions, their
enrichment ratio, and the count of strongly-constrained positions in each
region — distinct from the per-variant $Delta "LLR"$ scores above, which
carry the allele change. The unique/shared constraint enrichment ratio
is one of the two inputs to F5.

== ORF classification and initiation-context descriptors

Two site-level annotators record metadata specific to the alternative
initiation event. ORF type is assigned in two stages. The upstream filter
first reduces Ribo-TISH's compound `TisType` strings to a five-value
controlled vocabulary (`Annotated`, `Extended`, `Truncated`, `uORF`,
`Other`) used for filtering and imputation. The output column then carries
a finer eight-value enum (`annotated`, `extended`, `truncated`, `uorf`,
`uoorf`, `internal_oof`, `3utr_orf`, `alt_orf`), where the `uoorf`
category captures Ribo-TISH's `CDSFrameOverlap` types. The
ORF-classification method exposes this label together with the lengths of
the isoform and of its transcript-specific canonical, and a boolean
flag indicating whether the alternative start maintains the canonical
reading frame. The initiation-context method scores the 13-nucleotide
Kozak window against three weighting schemes derived from @kozak:
a full-length Hamming distance against the `gccgccRccATGG` consensus,
a "major-position" Hamming distance restricted to the strongly
conserved positions ($-3$ and $+4$), and a partial-weighting score that
applies weight 1 to the major positions and 0.1 to the minor positions.
The same method records the GC content of the window and the
trinucleotide at the initiation codon.

== Gene-level reference annotation

A gene-level method attaches to each gene a summary of external reference
information indexed by gene symbol: the HGNC-approved gene symbol and
alternative symbols, OMIM disease associations and phenotype labels
where applicable, and the GENCODE biotype of the gene. This annotation
is computed once per gene and shared across all TIS records of that
gene; it is not subjected to canonical-versus-isoform comparison.

= Evidence scoring

The annotations above are summarized per TIS by a dual-axis evidence-
scoring framework that runs after the comparator (§5), since several
criteria read the canonical-versus-isoform comparison. The two axes are
deliberately independent: an _existence_ score asks whether the
alternative isoform is a real biological entity, and a _functional_ score
asks whether it changes protein function. Each axis is the count of
satisfied criteria among six.

Every criterion returns one of three states: `True` (evidence present),
`False` (evidence genuinely absent), or `None` (cannot be evaluated —
the upstream annotation module did not run or did not produce the required
field). `None` results are excluded from both the score numerator and the
per-axis _evaluable_ count, so that a low score driven by missing data is
distinguishable from one driven by genuine non-evidence. The six existence
criteria are:

- *E1 — primate coding conservation*: the mean amino-acid percent
  identity of the isoform-unique region across the primate radiation
  (from the reading-frame integrity method, §3.4) is at or above a
  threshold ($"E1" gt.eq 0.80$, provisional). The fraction of primate
  species with an intact reading frame is retained as context for the
  reason string but is no longer the score basis.
- *E2 — mammalian coding conservation*: the same, over the mammalian
  radiation, on mean amino-acid percent identity
  ($"E2" gt.eq 0.50$, provisional).
- *E3 — coding-level nucleotide selection*: the mean phyloP over the
  isoform-unique region is at or above an absolute purifying-selection
  threshold ($"E3" gt.eq 2.0$, strong purifying selection). The
  criterion is an absolute statement about the unique region's coding
  conservation; the unique-versus-shared enrichment ratio is reported as
  context only and the threshold does not imply the unique region is more
  conserved than the shared core.
- *E4 — multi-cell-line support*: the TIS is detected in at least a
  minimum number of cell lines ($gt.eq 3$).
- *E5 — initiation efficiency*: the maximum per-cell-line initiation
  efficiency (the ratio of a TIS's read count to its gene's RNA-seq
  count) exceeds a threshold.
- *E6 — mass-spectrometric validation*: at least one isoform-unique
  tryptic peptide is matched in public MS spectra by a pre-computed
  PepQuery2 @pepquery search. In-silico detectability alone is _not_
  treated as evidence; E6 is `None` until the PepQuery2 precompute exists.

The six functional criteria are:

- *F1 — structured, biophysically distinct differential region*: the
  differential region is both folded and physicochemically distinct from
  the canonical. The criterion is `True` only when the mean
  predicted-structure pLDDT over the differential region is at or above
  $0.70$ _and_ a biophysical-distinctness flag is set — the latter when
  the canonical-versus-isoform comparison shows a GRAVY delta
  $gt.eq 0.3$, a charged-fraction delta $gt.eq 0.05$, or a disorder delta
  $gt.eq 0.05$ (provisional cutoffs). It is `None` when the structure
  prediction is unavailable or the biophysical comparison is missing; the
  reason names which half passed.
- *F2 — localization change*: the comparator flags a change in any
  localization feature (DeepLoc primary-compartment prediction, sorting
  signals, or membrane-type call) between canonical and isoform.
- *F3 — domain gain / loss*: at least one _real_ InterPro functional
  domain (a hit carrying a genuine InterPro accession, excluding
  disorder- and structure-only signatures such as MobiDB-lite, coils,
  low-complexity, SignalP, Phobius, and TMHMM) starts inside the
  differential region and is absent from the other form's domain set
  (§3.5), rather than merely repositioned.
- *F4 — targeting change*: SignalP or TargetP disagrees on the canonical
  versus the isoform (§3.6).
- *F5 — germline tolerance / constraint*: the isoform-unique region shows
  germline constraint by either of two independent signals — an ESM-2
  unique/shared constraint enrichment at or above a threshold ($2.0$,
  provisional), or a gnomAD germline depletion ratio below a threshold
  ($0.80$, provisional), the latter indicating that common germline
  variation avoids the unique region (per-variant effect scoring, §4.5).
  It is `None` only when both inputs are missing.
- *F6 — disease density enrichment*: the disease (ClinVar / COSMIC)
  variant density of the isoform-unique region, relative to the
  canonical-shared region, is at or above one ($"F6" gt.eq 1.0$,
  zero-point), so that disease variants concentrate in the unique region
  rather than merely being present (clinical genomic intersection). It is
  `None` when the enrichment ratio is undefined.

Two boolean flags, `existence_high_confidence` and
`functional_high_confidence`, mark isoforms whose score reaches a
high-confidence cutoff: an existence score of at least five satisfied
criteria, and a functional score of at least three. These cutoffs and the
$gt.eq 3$ cell-line threshold for E4 are fixed; they are no longer relaxed
by run-script overrides, so the curated 13-gene set is scored under the
same thresholds as a genome-wide run would be.

Thresholds fall into two classes. _Principled anchors_ are fixed on
biological grounds and are not tuned to any dataset: the E3 phyloP cutoff
of $2.0$ (strong purifying selection), the F1 pLDDT cutoff of $0.70$, the
F6 disease-enrichment zero-point of $1.0$, the $gt.eq 3$ cell-line
requirement for E4, and the existence / functional high-confidence cutoffs
of five and three. _Provisional thresholds_ — the E1 / E2 percent-identity
cutoffs ($0.80$ / $0.50$), the F5 constraint-enrichment ($2.0$) and
depletion-ratio ($0.80$) cutoffs, and the F1 biophysical-distinctness
deltas — are config-driven placeholders pending calibration on a
genome-wide run; the curated 13-gene cohort validates the scoring _logic_
rather than these calibration points, since a reviewer-selected gene set
is not representative of the genome-wide distribution.

= Differential annotation table

The output of the pipeline is a tabular representation in which each
row corresponds to exactly one alternative TIS. The columns of each row
fall into four groups:

1. *Identity*: gene symbol, gene identifier, canonical transcript
   identifier, TIS identifier (composed as
   `chromosome:position:strand:codon:transcript_id` so that distinct
   TIS events at the same codon on different transcripts remain
   distinguishable), TIS-supporting transcript identifier, chromosome,
   position, strand, initiation codon, ORF type, isoform and canonical
   protein lengths, differential region sequence and coordinates, and
   Kozak window.
2. *Per-sample expression*: for each cell line, the raw read count,
   RPM-normalized count, one-sided initiation-significance p-value, and
   initiation efficiency (TIS read count over gene RNA-seq count), laid
   out as wide columns
   (`expr_{cell_line}_{metric}`) so that cross-cell-line comparisons
   translate to column arithmetic.
3. *Canonical annotations*: for each protein-level annotation method,
   the complete output of that method applied to the TIS's
   transcript-specific canonical protein, flattened as
   `canonical_{method}_{field}` columns.
4. *Isoform annotations*: symmetrically, the complete output of each
   protein-level method applied to the alternative isoform protein, the
   output of each site-level method, and the output of each
   gene-level method, flattened as `isoform_{method}_{field}` columns.

This two-pane layout realizes the central design decision of the
pipeline: canonical-versus-isoform differentials are computed
_post-hoc_ from two complete and independently-derived annotation
records, rather than stored as pre-subtracted deltas. Scalar differentials
(e.g. $Delta "pI"$ between canonical and isoform) reduce to column
subtraction; positional differentials (e.g. ClinVar variants falling
within a truncation's lost region) are derived by intersecting each
positional hit list against the differential-region coordinates carried
in the identity columns. Storing both panes together ensures that the
same pair of annotation records is equally available for aggregate
analyses (pathway-level burden of alternative extensions), for
per-event analyses (whether a given truncation disrupts a specific
motif), and for re-comparison under alternative definitions of the
reference (for instance, asking whether an alternative isoform's
localization matches that of a paralogue canonical).

= Reference data and versioning

Reference data supporting the pipeline were built from primary sources
with machine-readable provenance: every reference artifact (ClinVar
parquet, gnomAD parquet, COSMIC parquet directory, AlphaMissense table,
Zoonomia phyloP / phastCons BigWig tracks and Cactus HAL, GENCODE
FASTA/GTF, DeepLoc environment) was recorded alongside a sidecar
document capturing the source URL, version, timestamp of retrieval,
size in bytes, and SHA-256 checksum for each file. GENCODE v49 (GRCh38
primary assembly) was used throughout for gene, transcript, CDS, and
protein-translation annotation; the Zoonomia 241-mammal `cactus241way`
phyloP track, the UCSC `phastCons100way` track, and the Zoonomia Cactus
HAL provided conservation data; gnomAD v4.1 exome
sites, ClinVar's current `variant_summary.txt.gz` release, COSMIC
v102 (GRCh38), and AlphaMissense hg38 provided variant data; DeepLoc 2.1
provided localization predictions.

#pagebreak()

#heading(level: 1, numbering: none)[References]

#bibliography("methods.bib", title: none, style: "nature")
