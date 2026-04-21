"""Reading-frame integrity analysis over MAF-aligned sequences.

Per-species checks:

- start codon conserved (first reference codon is ATG in target)
- frame intact (no insertion or deletion runs with length not divisible by 3,
  and no premature in-frame stop)
- premature stop (any in-frame stop before the last reference codon)
- AA percent identity vs. the reference translation

All inputs are MAF-aligned strings (reference + target, equal length,
containing ``-`` gap characters). The reference species provides the frame.
"""

from __future__ import annotations

from dataclasses import dataclass

_STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})

_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


@dataclass
class SpeciesFrameResult:
    """Per-species output of :func:`analyze_species`.

    ``aligned`` is False when the target has zero non-gap bases overlapping
    the reference region. All downstream flags are then ``None``.
    """

    species: str
    aligned: bool
    start_codon: str | None = None
    start_codon_conserved: bool | None = None
    frame_intact: bool | None = None
    premature_stop: bool | None = None
    n_frameshifts: int | None = None
    pident: float | None = None
    coverage: float | None = None


def _gap_runs(seq: str) -> list[tuple[int, int]]:
    """Return ``(start, length)`` for every maximal run of ``-`` in *seq*."""
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for i, ch in enumerate(seq):
        if ch == "-":
            if run_start is None:
                run_start = i
        elif run_start is not None:
            runs.append((run_start, i - run_start))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(seq) - run_start))
    return runs


def _count_frameshifts(ref_seq: str, target_seq: str) -> int:
    """Count indel runs whose lengths are not divisible by 3.

    Walks the alignment once. A run of columns where ``ref == '-'`` and
    ``target != '-'`` is a target insertion; a run where ``target == '-'``
    and ``ref != '-'`` is a target deletion. Each run contributes one
    frameshift when its length is not a multiple of 3. Runs where both are
    gaps are ignored.
    """
    assert len(ref_seq) == len(target_seq)

    frameshifts = 0
    state: str | None = None  # 'ins', 'del', or None
    run_len = 0

    def close():
        nonlocal frameshifts, run_len, state
        if state is not None and run_len % 3 != 0:
            frameshifts += 1
        run_len = 0
        state = None

    for r, t in zip(ref_seq, target_seq):
        r_gap = r == "-"
        t_gap = t == "-"
        if r_gap and t_gap:
            close()
            continue
        if r_gap and not t_gap:
            if state != "ins":
                close()
                state = "ins"
            run_len += 1
            continue
        if not r_gap and t_gap:
            if state != "del":
                close()
                state = "del"
            run_len += 1
            continue
        close()
    close()
    return frameshifts


def _project_target(ref_seq: str, target_seq: str) -> str:
    """Return target characters at columns where ``ref != '-'``.

    Resulting length equals the reference's ungapped length, so codon
    indexing follows the reference frame. Target gaps survive as ``-``.
    """
    return "".join(t for r, t in zip(ref_seq, target_seq) if r != "-")


def _translate(seq: str) -> str:
    """Translate *seq* in frame 0; codons with gaps or ambiguous bases → ``X``.

    Stop codons translate to ``*``. Trailing partial codon is dropped.
    """
    out: list[str] = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i : i + 3].upper()
        if "-" in codon or any(c not in "ACGT" for c in codon):
            out.append("X")
        else:
            out.append(_CODON_TABLE.get(codon, "X"))
    return "".join(out)


def _aa_pident(ref_aa: str, target_aa: str) -> float | None:
    """Matched / comparable residues; returns ``None`` if none comparable.

    A position is comparable when neither translation is ``X``. Stop codons
    (``*``) count as residues — matching stops are matches, mismatched
    stops are mismatches.
    """
    comparable = 0
    matches = 0
    for r, t in zip(ref_aa, target_aa):
        if r == "X" or t == "X":
            continue
        comparable += 1
        if r == t:
            matches += 1
    if comparable == 0:
        return None
    return matches / comparable


def analyze_species(
    species: str,
    ref_seq: str,
    target_seq: str,
) -> SpeciesFrameResult:
    """Score reading-frame integrity for one target species.

    Args:
        species: Name to attach to the result.
        ref_seq: Reference (e.g. hg38) aligned sequence — ``-`` allowed.
        target_seq: Target species' aligned sequence — ``-`` allowed,
            same length as ``ref_seq``.

    Returns:
        :class:`SpeciesFrameResult`. Flags are ``None`` when the target
        contributes zero non-gap bases overlapping the reference.
    """
    if len(ref_seq) != len(target_seq):
        raise ValueError(
            f"MAF rows must be equal length; ref={len(ref_seq)} target={len(target_seq)}"
        )

    target_projected = _project_target(ref_seq, target_seq)
    target_nongap = target_projected.replace("-", "")
    ref_ungapped = ref_seq.replace("-", "")

    if not target_nongap:
        return SpeciesFrameResult(species=species, aligned=False, coverage=0.0)

    coverage = len(target_nongap) / len(ref_ungapped) if ref_ungapped else None

    start_codon = target_projected[:3].upper() if len(target_projected) >= 3 else None
    start_conserved = start_codon == "ATG" if start_codon is not None else None

    frameshifts = _count_frameshifts(ref_seq, target_seq)

    # Premature stop: any stop codon within the first (N-1) reference codons,
    # where N is the number of full reference codons.
    n_ref_codons = len(ref_ungapped) // 3
    premature_stop = False
    last_scored = max(0, n_ref_codons - 1)
    for i in range(last_scored):
        codon = target_projected[i * 3 : i * 3 + 3].upper()
        if len(codon) < 3 or "-" in codon or any(c not in "ACGT" for c in codon):
            continue
        if codon in _STOP_CODONS:
            premature_stop = True
            break

    frame_intact = frameshifts == 0 and not premature_stop

    ref_aa = _translate(ref_ungapped)
    target_aa = _translate(target_projected)
    pident = _aa_pident(ref_aa, target_aa)

    return SpeciesFrameResult(
        species=species,
        aligned=True,
        start_codon=start_codon,
        start_codon_conserved=start_conserved,
        frame_intact=frame_intact,
        premature_stop=premature_stop,
        n_frameshifts=frameshifts,
        pident=pident,
        coverage=coverage,
    )


def aggregate_species_results(
    results: list[SpeciesFrameResult],
) -> dict[str, float | int | None]:
    """Roll per-species results into primate/mammalian-style aggregates.

    Returns keys matching the output-column scheme from
    ``conservation_path12_spec.md``:

    - ``n_species_aligned``
    - ``n_species_intact_frame``
    - ``frac_intact``
    - ``start_codon_conserved`` (fraction of aligned species)
    - ``mean_pident`` (over aligned species)
    """
    aligned = [r for r in results if r.aligned]
    n_aligned = len(aligned)
    if n_aligned == 0:
        return {
            "n_species_aligned": 0,
            "n_species_intact_frame": 0,
            "frac_intact": None,
            "start_codon_conserved": None,
            "mean_pident": None,
        }

    n_intact = sum(1 for r in aligned if r.frame_intact)
    n_start = sum(1 for r in aligned if r.start_codon_conserved)
    pidents = [r.pident for r in aligned if r.pident is not None]
    mean_pident = sum(pidents) / len(pidents) if pidents else None
    return {
        "n_species_aligned": n_aligned,
        "n_species_intact_frame": n_intact,
        "frac_intact": n_intact / n_aligned,
        "start_codon_conserved": n_start / n_aligned,
        "mean_pident": mean_pident,
    }
