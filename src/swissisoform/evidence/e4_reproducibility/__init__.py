"""E4 — reproducibility (multi-cell-line). Plumbing: site.expression."""

from __future__ import annotations

from swissisoform.config import ScoringConfig
from swissisoform.evidence.common import CriterionResult
from swissisoform.models import TranslationInitiationSite


def score(site: TranslationInitiationSite, cfg: ScoringConfig) -> CriterionResult:
    """E4: TIS detected in at least ``min_cell_lines`` cell lines.

    Sensitivity note: with ``cfg.min_cell_lines == 1`` (used by single-sample
    diagnostic presets) this criterion is degenerate — every TIS in the
    dataset is detected in ≥1 cell line by construction, so E4 returns True
    universally and contributes a constant +1 to every existence score. For
    multi-sample / production runs raise it to 2-3 (the default is 3) so it
    actually discriminates.
    """
    n = len(site.expression)
    passed = n >= cfg.min_cell_lines
    return CriterionResult(
        "E4_multi_cell_line",
        passed,
        f"n_cell_lines={n} (threshold {cfg.min_cell_lines})",
    )
