"""Benchmark Arm A (salmon) vs Arm B (IsoQuant) on HeLa TIS disambiguation.

Short-read salmon vs long-read IsoQuant on narrowing each TIS's candidate
transcripts. Runs Arm A immediately (its quant.sf are ready); prints the full
head-to-head once Arm B's IsoQuant output lands. Each arm is scored over the
multi-candidate HeLa TIS: of the Ribo-TISH candidate transcripts at each init
site, how many survive the arm's present-set -> resolved to 1 / lost to 0 /
still ambiguous.

Usage:  python scripts/analysis/benchmark_expression_arms.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from swissisoform.sourceseq.benchmark import (
    arm_disambiguation,
    per_tis_candidates,
    present_set_agreement,
)
from swissisoform.sourceseq.expression import (
    expressed_in_replicates,
    expressed_transcripts,
    load_isoquant_abundance,
)

ROOT = Path(__file__).resolve().parents[2]
COMBINED = ROOT / "data/output/filtered/all_samples_combined.parquet"
SALMON = ROOT / "data/reference/salmon"
ISOQUANT_DIR = ROOT / "data/reference/longread/isoquant_HeLa"
SAMPLE = "HeLa"
SALMON_MIN_TPM = 1.0
ISOQUANT_MIN_COUNT = 3.0


def _fmt(d: dict) -> str:
    return (
        f"  multi-candidate TIS : {d['multi_candidate_tis']:,}\n"
        f"    resolved -> 1     : {d['resolved_1']:,} ({d['pct_resolved']:.0f}%)\n"
        f"    lost -> 0         : {d['lost_0']:,} ({d['pct_lost']:.0f}%)\n"
        f"    still ambiguous   : {d['ambiguous_2plus']:,} ({d['pct_ambiguous']:.0f}%)\n"
        f"    mean survivors    : {d['mean_survivors']:.2f}"
    )


def find_isoquant_counts(d: Path) -> Path | None:
    # reference transcript counts (we disabled novel models with --no_model_construction)
    for pattern in ("**/*transcript_counts.tsv", "**/*transcript_grouped_counts.tsv"):
        hits = [h for h in d.glob(pattern) if "model" not in h.name]
        if hits:
            return sorted(hits)[0]
    return None


def main() -> None:
    combined = pd.read_parquet(
        COMBINED, columns=["Tid", "GenomePos", "StartCodon", f"present_{SAMPLE}"]
    )
    cand = per_tis_candidates(combined, SAMPLE)
    n_multi = sum(1 for v in cand.values() if len(v) >= 2)
    print(f"HeLa init-sites: {len(cand):,}  |  multi-candidate (>=2): {n_multi:,}\n")

    reps = sorted(str(p) for p in SALMON.glob("HeLa_rep*/quant.sf"))
    arm_a = expressed_in_replicates(reps, min_tpm=SALMON_MIN_TPM, require="all")
    a = arm_disambiguation(cand, arm_a)
    print(f"=== Arm A: salmon (TPM>={SALMON_MIN_TPM:.0f} both reps), present={len(arm_a):,} ===")
    print(_fmt(a))

    iq = find_isoquant_counts(ISOQUANT_DIR)
    if iq is None:
        print("\n=== Arm B: IsoQuant -- output not found yet ===")
        print(f"  (re-run this script once the long-read job writes {ISOQUANT_DIR}/OUT/)")
        return

    arm_b = expressed_transcripts(load_isoquant_abundance(iq), min_value=ISOQUANT_MIN_COUNT)
    b = arm_disambiguation(cand, arm_b)
    hdr = f"Arm B: IsoQuant ({iq.name}, counts>={ISOQUANT_MIN_COUNT:.0f}), present={len(arm_b):,}"
    print(f"\n=== {hdr} ===")
    print(_fmt(b))

    ag = present_set_agreement(arm_a, arm_b, a_name="salmon", b_name="isoquant")
    print("\n=== present-set agreement (IsoQuant long-read = gold standard) ===")
    n_s, n_i, ov = ag["n_salmon"], ag["n_isoquant"], ag["overlap"]
    print(f"  salmon={n_s:,}  isoquant={n_i:,}  overlap={ov:,}  jaccard={ag['jaccard']:.2f}")
    print(f"  salmon-only={ag['salmon_only']:,}  isoquant-only={ag['isoquant_only']:,}")
    prec, rec = ag["precision_salmon_vs_isoquant"], ag["recall_salmon_vs_isoquant"]
    print(f"  salmon precision vs isoquant={prec:.2f}  recall={rec:.2f}")

    print("\n=== verdict (more resolved + fewer lost = better disambiguation) ===")
    rs, rl = a["pct_resolved"], b["pct_resolved"]
    ls, ll = a["pct_lost"], b["pct_lost"]
    print(f"  resolved-to-1 : salmon {rs:.0f}%  vs  isoquant {rl:.0f}%")
    print(f"  lost-to-0     : salmon {ls:.0f}%  vs  isoquant {ll:.0f}%")


if __name__ == "__main__":
    main()
