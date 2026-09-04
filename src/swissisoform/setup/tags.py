"""Build the frozen tag registry from the reviewed candidate table (setup-time).

Freezes a tag vocabulary — every tag's kind, metric, direction and cutoff — as
provisioned reference data under ``data/reference/tags/<version>/``. The runtime
read-side is :mod:`swissisoform.tags.registry`; the evaluator is
:mod:`swissisoform.tags.evaluate`.

The mechanical fields are **re-derived, not parsed back out of the CSV**. The
reviewed table (``figures/tag_vocab/tag_candidates.csv``) is authoritative for
exactly two things a human decided — ``decision`` and ``proposed_label`` — and
carries the rest only as a rendering (``direction`` survives only inside the
``test`` string). So the builder re-runs the sweep's own
``propose -> apply_filters -> choose_cutoff`` against the same frozen
distributions and keeps the rows the reviewer did not reject. Nothing is
recovered by string-matching.

Two cutoff sources, one flag:

``--cutoffs config``
    Criterion tags carry their live ``ScoringConfig`` thresholds, so firing is
    identical to ``EvidenceScoringModule`` today. Build this first: the criterion
    tags must reproduce ``isoform_scoring_criteria`` row for row before any cutoff
    is allowed to move, so that a calibration finding can never be confused with a
    wiring bug.
``--cutoffs distribution``
    Criterion tags carry the cutoff the frozen distribution put them at.

Either way a criterion enters as a ``derived`` tag that runs its own scorer, with
the cutoff handed to it through ``cutoff_overrides``. Only the numbers move; the
gates the scorer applies around them do not. Swept tags have no ``ScoringConfig``
equivalent and carry their distribution cutoff under both flags.

Driven by the thin CLI ``scripts/setup/build_tag_registry.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq

from swissisoform import distributions as dist_mod
from swissisoform.tags import candidates as cand_mod
from swissisoform.tags import derived as derived_mod
from swissisoform.tags import seeds
from swissisoform.tags.registry import (
    KIND_BOOL,
    KIND_DERIVED,
    KIND_LLM,
    KIND_THRESHOLD,
    REGISTRY_COLUMNS,
    REGISTRY_FILE,
    SIDECAR_FILE,
    axis_for,
    from_frame,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = ROOT / "figures" / "clustering_dims" / "feature_space" / "feature_catalog.csv"
DEFAULT_CANDIDATES = ROOT / "figures" / "tag_vocab" / "tag_candidates.csv"
DEFAULT_DIST_VERSION = "v3"
DEFAULT_RUN = "full_catalog"

# The sweep's `kind` vocabulary is narrower than the registry's — it has no notion
# of a derived tag, because it can only propose what it can cut.
KIND_FROM_SWEEP = {"code": KIND_THRESHOLD, "bool": KIND_BOOL, "llm": KIND_LLM}

# `decision` values that keep a row. Blank means "not yet rejected", which is
# where the whole table sits until the review pass happens; a registry built now
# is therefore a default rather than a selection, and the version number is what
# makes that safe.
REJECT_DECISIONS = frozenset({"remove", "drop"})


class TagBuildError(RuntimeError):
    """Raised when the candidate table and the sweep cannot be reconciled."""


def _rel(path: Path) -> str:
    """Repo-relative path when inside the repo, else absolute."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_columns(run: str | None, parquet: Path | None) -> tuple[set[str], str]:
    """Column names of a run, for the boolean stream's availability check.

    Only the schema is read: the builder needs to know which boolean columns exist,
    not what is in them, and the genome-wide parquet is ~2 GB.
    """
    if parquet is not None:
        return set(pq.read_schema(parquet).names), parquet.parent.name
    run_dir = ROOT / "data" / "output" / (run or "")
    merged = run_dir / "all_paired.parquet"
    if merged.exists():
        return set(pq.read_schema(merged).names), run or ""
    shards = sorted((ROOT / "data" / "output").glob(f"{run}_shard_*/all_paired.parquet"))
    if shards:
        return set(pq.read_schema(shards[0]).names), run or ""
    raise SystemExit(f"no all_paired.parquet under {run_dir} or {run}_shard_*/")


def accepted_ids(candidates_csv: Path) -> tuple[dict[str, str], dict[str, int]]:
    """``({tag_id: reviewed_label}, counts)`` for the rows the reviewer kept.

    A blank ``decision`` counts as kept — see :data:`REJECT_DECISIONS`.
    """
    table = pd.read_csv(candidates_csv)
    for column in ("tag_id", "decision", "proposed_label"):
        if column not in table.columns:
            raise TagBuildError(f"{candidates_csv} has no {column!r} column")
    decision = table["decision"].astype("string").str.strip().str.lower()
    keep = table[~decision.isin(REJECT_DECISIONS)]
    labels = {
        str(r["tag_id"]): ("" if pd.isna(r["proposed_label"]) else str(r["proposed_label"]))
        for _, r in keep.iterrows()
    }
    counts = {
        "rows_in_table": int(len(table)),
        "rejected": int(len(table) - len(keep)),
        "accepted": int(len(keep)),
    }
    return labels, counts


def _dedupe(cands: Iterable[cand_mod.Candidate]) -> list[cand_mod.Candidate]:
    """First-wins dedupe by ``tag_id``, asserting the duplicates really are equal.

    ``propose`` emits some ids twice — the sweep's complementary-direction dedupe
    happens later, inside ``build_table``, which the builder does not call. Every
    such pair is identical in everything that decides firing, so first-wins is
    right; taking the first *silently* would not be, because a genuine collision
    (two different tags slugging to one id) would then be invisible.
    """
    seen: dict[str, cand_mod.Candidate] = {}
    for cand in cands:
        prior = seen.get(cand.tag_id)
        if prior is None:
            seen[cand.tag_id] = cand
            continue
        identity = ("metric", "direction", "cutoff", "valid_for", "kind")
        if any(getattr(prior, f) != getattr(cand, f) for f in identity):
            raise TagBuildError(
                f"tag_id {cand.tag_id!r} proposed twice with different definitions: "
                f"{ {f: getattr(prior, f) for f in identity} } vs "
                f"{ {f: getattr(cand, f) for f in identity} }"
            )
    return list(seen.values())


def sweep(
    catalog: pd.DataFrame,
    dist: dist_mod.Distributions,
    columns: set[str],
    *,
    band: tuple[float, float] = cand_mod.DEFAULT_BAND,
) -> list[cand_mod.Candidate]:
    """Re-run the candidate sweep up to (not including) the review table."""
    by_feature = cand_mod._catalog_index(catalog)
    funnel = cand_mod.Funnel()
    proposed = cand_mod.propose(catalog, dist, columns)
    kept = cand_mod.apply_filters(proposed, dist, by_feature, funnel)
    return [cand_mod.choose_cutoff(c, dist, band) for c in kept]


def _warn_if_truncating(criterion_id: str, field: str, value: float, cutoffs: str) -> None:
    """Warn when a distribution cutoff lands between integers on an int threshold.

    A percentile of a discrete distribution is generally not an integer, and the
    evaluator coerces to the field's declared type — so a calibrated
    ``min_cell_lines`` of 1.12 becomes 1, and a ``massspec_unique_peptides_min``
    of 0.07 becomes 0, i.e. "at least zero peptides", which fires on everything.
    Counting criteria need their cutoffs chosen on the integers, not swept; until
    they are, this says so at build time rather than at read time.
    """
    if cutoffs != "distribution":
        return
    from swissisoform.config import ScoringConfig

    declared = {f.name: f.type for f in dataclasses.fields(ScoringConfig)}
    if declared.get(field) == "int" and value != int(value):
        print(
            f"  WARNING {criterion_id}: {field}={value:.4g} is not an integer and will "
            f"truncate to {int(value)}. This threshold counts things; a swept "
            "percentile is not a meaningful cutoff for it."
        )


def criterion_rows(by_id: dict[str, cand_mod.Candidate], *, cutoffs: str) -> list[dict[str, Any]]:
    """One ``derived`` row per scored criterion — all sixteen.

    Every criterion is carried as ``derived`` rather than as a threshold on its
    headline metric. That is not a stylistic choice: measured on cheeseman50, the
    threshold form disagreed with the scorer on 5 of the 13 criteria the sweep can
    express, and in each case the threshold had dropped a gate — P1 reads a
    populated ``plddt_diffregion_mean`` on a protein whose fold status is
    ``too_long`` (scorer: not-evaluable, threshold: False), M1 is undefined for
    separate ORFs, M2/P2 gate on their own status fields, and M1/S2 are either-or
    roll-ups over two and three inputs. Turning "could not evaluate" into
    "evidence absent" is precisely the failure issue #30 exists to remove.

    The cutoffs still come from the registry, as ``cutoff_overrides`` — a
    ``{ScoringConfig field: value}`` map the evaluator folds into the config it
    hands the scorer. So a calibrated version moves a criterion's number while
    keeping every gate around it.
    """
    rows: list[dict[str, Any]] = []
    for criterion_id, label in seeds.CRITERION_LABELS.items():
        if criterion_id not in derived_mod.SCORER_BY_CRITERION:
            raise TagBuildError(f"{criterion_id} has a label but no scorer")
        category = criterion_id[0]
        branches = seeds.seeds_for_criterion(criterion_id)
        overrides: dict[str, float] = {}
        notes: list[str] = []
        for seed in branches:
            if seed.note:
                notes.append(seed.note)
            if not seed.config_field:
                continue
            cand = by_id.get(cand_mod._slug(f"{criterion_id}__{seed.metric}"))
            value = (
                cand.cutoff
                if cutoffs == "distribution" and cand is not None and cand.cutoff is not None
                else cand_mod._current_cutoff(seed.config_field, seed.literal)
            )
            if value is not None:
                _warn_if_truncating(criterion_id, seed.config_field, float(value), cutoffs)
                overrides[seed.config_field] = float(value)
        # A citation needs one number. An either-or criterion has no single one,
        # so it gets none rather than an arbitrary branch's.
        metrics_ = [s.metric for s in branches if s.metric]
        single = metrics_[0] if len(metrics_) == 1 else ""
        rows.append(
            {
                "tag_id": cand_mod._slug(criterion_id),
                "category": category,
                "axis": axis_for(category),
                "label": label,
                "kind": KIND_DERIVED,
                "metric": single,
                "direction": branches[0].direction if single else "",
                "cutoff": overrides.get(branches[0].config_field or "") if single else None,
                "cutoff_source": cutoffs,
                "cutoff_pctile": None,
                "cutoff_overrides": json.dumps(overrides, sort_keys=True) if overrides else "",
                "valid_for": "|".join(seeds.ALL_ORF_TYPES),
                "criterion_id": criterion_id,
                "source": "criterion",
                "blocked": "",
                "note": " ".join(notes),
            }
        )
    return rows


def candidate_row(cand: cand_mod.Candidate, label: str) -> dict[str, Any]:
    """One registry row from one swept (non-criterion) candidate.

    Swept tags have no ``ScoringConfig`` equivalent, so their cutoff is the
    distribution's under either ``--cutoffs`` mode; only the criterion tags differ.
    """
    return {
        "tag_id": cand.tag_id,
        "category": cand.category,
        "axis": axis_for(cand.category),
        "label": label or cand.label,
        "kind": KIND_FROM_SWEEP.get(cand.kind, cand.kind),
        "metric": cand.metric,
        "direction": cand.direction,
        "cutoff": cand.cutoff,
        "cutoff_source": cand.cutoff_source,
        "cutoff_pctile": cand.cutoff_pctile,
        "cutoff_overrides": "",
        "valid_for": "|".join(cand.valid_for),
        "criterion_id": cand.criterion_id,
        "source": cand.source,
        "blocked": cand.blocked,
        "note": cand.note,
    }


def _check_scorer_names() -> None:
    """Fail the build if a criterion was renamed out from under the derived map.

    ``SCORER_BY_CRITERION`` is hand-written and its keys are the strings the
    scorers put in ``CriterionResult.name`` — not derivable from the package name
    (M1 lives in ``m1_germline_constraint`` but is called
    ``M1_pathogenic_variant_enrichment``). A rename would otherwise leave a derived
    tag quietly scoring a different criterion, in every run, with no error.
    """
    from swissisoform.config import ScoringConfig
    from swissisoform.models import ORFType, TranslationInitiationSite

    probe = TranslationInitiationSite(
        tis_id="probe",
        gene_name="probe",
        transcript_id="probe",
        chrom="chr1",
        position=1,
        strand="+",
        start_codon="ATG",
        orf_type=ORFType.TRUNCATED,
    )
    bad = derived_mod.check_names(probe, ScoringConfig())
    if bad:
        raise TagBuildError(
            "criterion id(s) in tags/derived.py no longer match what the scorer "
            f"reports: {', '.join(bad)}"
        )


def build(
    *,
    catalog_csv: Path,
    candidates_csv: Path,
    columns: set[str],
    dist_version: str,
    cutoffs: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Assemble the registry table. Returns ``(frame, counts)``."""
    _check_scorer_names()
    catalog = pd.read_csv(catalog_csv)
    dist = dist_mod.load(dist_version)
    labels, counts = accepted_ids(candidates_csv)

    swept = _dedupe(sweep(catalog, dist, columns))
    by_id = {c.tag_id: c for c in swept}
    missing = sorted(set(labels) - set(by_id))
    if missing:
        raise TagBuildError(
            f"{len(missing)} accepted tag(s) no longer proposed by the sweep — the "
            f"candidate table and the code have diverged: {', '.join(missing[:6])}"
        )

    # Criterion-stream candidates are superseded by the derived rows below: the
    # same claim, with the gates the threshold form cannot carry. Keeping both
    # would put two tags in the vocabulary for one criterion, one of them wrong
    # on exactly the rows where being wrong matters most.
    rows = [
        candidate_row(by_id[tid], labels[tid]) for tid in labels if by_id[tid].source != "criterion"
    ]
    rows.extend(criterion_rows(by_id, cutoffs=cutoffs))
    frame = pd.DataFrame(rows, columns=list(REGISTRY_COLUMNS))
    frame = frame.sort_values(["category", "kind", "tag_id"], kind="stable").reset_index(drop=True)

    kinds = frame["kind"].value_counts().to_dict()
    counts.update(
        {
            "tags": int(len(frame)),
            "code_fired": int(((frame["kind"] != KIND_LLM) & (frame["blocked"] == "")).sum()),
            **{f"kind_{k}": int(v) for k, v in kinds.items()},
        }
    )
    return frame, counts


def write_sidecar(
    out_dir: Path,
    *,
    catalog_csv: Path,
    candidates_csv: Path,
    dist_version: str,
    cutoffs: str,
    source_run: str,
    counts: dict[str, int],
) -> None:
    """Write ``_setup.json`` — provenance, plus the caveats a reader must know."""
    payload: dict[str, Any] = {
        "artifact": "tag registry (frozen vocabulary + cutoffs)",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cutoff_source": cutoffs,
        "distributions_version": dist_version,
        "source_run_for_columns": source_run,
        "candidates_csv": _rel(candidates_csv),
        "candidates_csv_sha256": _sha256(candidates_csv),
        "feature_catalog": _rel(catalog_csv),
        "feature_catalog_sha256": _sha256(catalog_csv),
        "counts": counts,
        "caveats": [
            "A blank `decision` in the candidate table counts as accepted, so a "
            "registry built before the review pass is a default (everything not "
            "yet rejected), not a selection.",
            "--cutoffs config only moves the criterion tags; swept tags have no "
            "ScoringConfig equivalent and keep their distribution cutoff either way.",
            "Derived tags call their criterion's scorer, so their value equals the "
            "criterion's by construction and no cutoff of theirs lives here.",
            "LLM tags carry no state from code. A consumer must render them as "
            "unanswered, never as off.",
        ],
    }
    (out_dir / SIDECAR_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    """Build one frozen tag-registry version. Refuses to clobber without --force."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--version", default="v1", help="Version directory name (default: v1)")
    p.add_argument(
        "--cutoffs",
        choices=("config", "distribution"),
        default="config",
        help="Where criterion cutoffs come from (default: config — reproduces today's scoring)",
    )
    p.add_argument(
        "--dist-version",
        default=DEFAULT_DIST_VERSION,
        help=f"Frozen distributions version to cut against (default: {DEFAULT_DIST_VERSION})",
    )
    p.add_argument("--run", default=DEFAULT_RUN, help="Run whose schema names the columns")
    p.add_argument("--parquet", type=Path, default=None, help="Explicit all_paired.parquet")
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="Feature catalog CSV")
    p.add_argument(
        "--candidates", type=Path, default=DEFAULT_CANDIDATES, help="Reviewed candidate CSV"
    )
    p.add_argument("--out", type=Path, default=None, help="Override the output directory")
    p.add_argument("--force", action="store_true", help="Overwrite an existing version")
    args = p.parse_args(list(argv) if argv is not None else None)

    out_dir = args.out or (ROOT / "data" / "reference" / "tags" / args.version)
    if (out_dir / REGISTRY_FILE).exists() and not args.force:
        raise SystemExit(
            f"{out_dir} already holds a built version. The registry is frozen on "
            "purpose — bump --version for a new one, or pass --force to rebuild in place."
        )

    columns, source_run = run_columns(args.run, args.parquet)
    frame, counts = build(
        catalog_csv=args.catalog,
        candidates_csv=args.candidates,
        columns=columns,
        dist_version=args.dist_version,
        cutoffs=args.cutoffs,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / REGISTRY_FILE, index=False)
    write_sidecar(
        out_dir,
        catalog_csv=args.catalog,
        candidates_csv=args.candidates,
        dist_version=args.dist_version,
        cutoffs=args.cutoffs,
        source_run=source_run,
        counts=counts,
    )
    # Read it straight back: a registry that cannot be loaded is not a registry,
    # and the failure belongs at build time rather than mid-run.
    reg = from_frame(args.version, pd.read_parquet(out_dir / REGISTRY_FILE), {})
    print(
        f"wrote {out_dir}  ({len(reg)} tags, {len(reg.code_fired())} code-fired, "
        f"cutoffs={args.cutoffs}, distributions={args.dist_version})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
