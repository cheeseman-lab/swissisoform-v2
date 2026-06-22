"""Benchmark the two expression arms on TIS-transcript disambiguation.

Compares Arm A (short-read salmon present-set) and Arm B (long-read IsoQuant
present-set) on how well each narrows a TIS's candidate transcripts. For every
TIS with multiple Ribo-TISH candidate transcripts, an arm's present-set either:

- **resolves** it to a single surviving candidate,
- **loses** it (zero survive — over-pruned), or
- leaves it **ambiguous** (>=2 survive — hand off to the window-purity test).

The superior arm resolves more TIS to a single confident source while *losing*
fewer. Long-read (full-length, unambiguous isoform identity) is also treated as
a gold standard for which transcripts exist. Pure set logic — no I/O — so it is
unit-testable; the driver `scripts/analysis/benchmark_expression_arms.py` wires
in the real present-sets. See ``docs/plans/source_transcript_resolution.md``.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def per_tis_candidates(combined: pd.DataFrame, sample: str) -> dict[str, set[str]]:
    """Map each genomic init site to its candidate transcript set, for one sample.

    The init site is ``chrom:pos:strand:codon`` where ``pos`` is the start-codon
    A (lo for ``+``, hi for ``-`` — the ``assembly.py`` convention). Transcripts
    sharing an init site are the competing candidate sources for that TIS.

    Args:
        combined: ``all_samples_combined`` frame with ``Tid``, ``GenomePos``,
            ``StartCodon``, and ``present_{sample}`` columns.
        sample: Cell line name (e.g. ``"HeLa"``).

    Returns:
        ``{init_site: {Tid, ...}}`` over rows present in ``sample``.
    """
    df = combined[combined[f"present_{sample}"].fillna(False).astype(bool)]

    def _key(gp: str, codon: str) -> str:
        chrom, coords, strand = gp.rsplit(":", 2)
        lo, hi = coords.split("-")
        pos = lo if strand == "+" else hi
        return f"{chrom}:{pos}:{strand}:{codon}"

    out: dict[str, set[str]] = {}
    for gp, codon, tid in zip(df["GenomePos"], df["StartCodon"].astype(str), df["Tid"].astype(str)):
        out.setdefault(_key(gp, codon), set()).add(tid)
    return out


def arm_disambiguation(
    candidates: Mapping[str, set[str]],
    present: set[str],
    min_candidates: int = 2,
) -> dict[str, float]:
    """Summarize how one arm's present-set narrows multi-candidate TIS.

    Args:
        candidates: ``{init_site: candidate Tids}`` from :func:`per_tis_candidates`.
        present: The arm's set of present (expressed) transcript ids.
        min_candidates: Only score TIS with at least this many candidates.

    Returns:
        Counts and percentages for lost (0 survive) / resolved (1) /
        ambiguous (>=2), plus the mean number of surviving candidates.
    """
    groups = [v for v in candidates.values() if len(v) >= min_candidates]
    n = len(groups)
    lost = resolved = ambiguous = surv_total = 0
    for cand in groups:
        s = len(cand & present)
        surv_total += s
        if s == 0:
            lost += 1
        elif s == 1:
            resolved += 1
        else:
            ambiguous += 1
    pct = lambda x: 100.0 * x / n if n else float("nan")  # noqa: E731
    return {
        "multi_candidate_tis": n,
        "lost_0": lost,
        "resolved_1": resolved,
        "ambiguous_2plus": ambiguous,
        "pct_lost": pct(lost),
        "pct_resolved": pct(resolved),
        "pct_ambiguous": pct(ambiguous),
        "mean_survivors": surv_total / n if n else float("nan"),
    }


def present_set_agreement(
    a: set[str], b: set[str], *, a_name: str = "A", b_name: str = "B"
) -> dict:
    """Overlap of two present-sets; precision/recall treating ``b`` as gold standard."""
    inter = a & b
    prec = len(inter) / len(a) if a else float("nan")
    rec = len(inter) / len(b) if b else float("nan")
    jac = len(inter) / len(a | b) if (a or b) else float("nan")
    return {
        f"n_{a_name}": len(a),
        f"n_{b_name}": len(b),
        "overlap": len(inter),
        f"{a_name}_only": len(a - b),
        f"{b_name}_only": len(b - a),
        f"precision_{a_name}_vs_{b_name}": prec,
        f"recall_{a_name}_vs_{b_name}": rec,
        "jaccard": jac,
    }


def compare_arms(
    candidates: Mapping[str, set[str]],
    arm_a_present: set[str],
    arm_b_present: set[str],
    *,
    a_name: str = "salmon",
    b_name: str = "isoquant",
) -> dict:
    """Full head-to-head: present-set agreement + per-arm disambiguation."""
    agreement = present_set_agreement(arm_a_present, arm_b_present, a_name=a_name, b_name=b_name)
    return {
        "agreement": agreement,
        a_name: arm_disambiguation(candidates, arm_a_present),
        b_name: arm_disambiguation(candidates, arm_b_present),
    }
