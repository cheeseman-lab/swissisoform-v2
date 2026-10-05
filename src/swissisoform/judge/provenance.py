"""What a judgment was made on, recorded so it can be checked later.

A results file is only interpretable against the exact text the judge read. The
arm outputs are overwritten in place on every rerun, and ``run_judge.py``
resumes by id, so without a fingerprint a results file can silently mix
judgments from two request builds -- which is what the v3 fit did: 14,700 of
its 25,200 rows were carry-overs from v2, judged against a different reference,
and the judged outputs had been regenerated on disk by the time it was read.

Each request carries the sha256 of both responses it shows, and the build as a
whole a ``build_id`` derived from every prompt. ``run_judge.py`` copies both into
each result, so ``analyze.py`` can refuse a results file that is not all one
build, and can say which judged responses no longer match the corpus on disk.

The other half is the arm side. The shared reference is only fair if it carries
the inputs the arms actually ran on, and the v3 one did not: every reference was
built with tag registry v1 while the tags arms ran on v2/v3, so a cutoff the
arms were shown ("vs cutoff 11.38") read as fabricated against it in 22-25% of
tags outputs. ``run_llm_variants.py`` now records each arm run's registry and
distribution versions and a content digest of its source data in
:data:`ARM_PROVENANCE`; the request builder reads those, builds the reference with
the same versions, and refuses arms whose provenance disagrees.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Written beside requests.jsonl: every request without its prompt.
INDEX_NAME = "requests_index.jsonl"

# One per arm output directory: what each run of that arm was given.
ARM_PROVENANCE = "_arm_provenance.json"

# The per-isoform run stamps llm.py writes beside each artifact.
_STAMPS = ("categories.meta.json", "synthesis.meta.json")


def text_sha(text: str) -> str:
    """sha256 of one judged text, as hex."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_digest(requests: Iterable[tuple[str, str]]) -> str:
    """A build id from ``(request_id, prompt)`` pairs, independent of their order.

    Content-derived rather than minted, so rebuilding byte-identical requests
    keeps the id -- and results made against them stay valid -- while any change
    to a response, a reference or the rubric yields a new one.
    """
    digest = hashlib.sha256()
    for rid, prompt in sorted(requests):
        digest.update(rid.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text_sha(prompt).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def check_results(results: Iterable[dict], index: dict[str, dict], build_id: str) -> dict[str, int]:
    """Count results that cannot be tied to the current request build.

    Args:
        results: Parsed ``results.jsonl`` rows.
        index: ``{request_id: request-without-prompt}`` from :data:`INDEX_NAME`.
        build_id: The build ``requests_meta.json`` names.

    Returns:
        ``n_results`` plus three failure counts: ``not_in_requests`` (an id the
        current build does not contain), ``other_build`` (a row stamped with a
        different build, or with none -- every pre-provenance results file), and
        ``sha_mismatch`` (the row names different response text than the request).
    """
    out = {"n_results": 0, "not_in_requests": 0, "other_build": 0, "sha_mismatch": 0}
    for row in results:
        out["n_results"] += 1
        request = index.get(row.get("id", ""))
        if request is None:
            out["not_in_requests"] += 1
            continue
        if row.get("build_id") != build_id:
            out["other_build"] += 1
            continue
        if (row.get("sha_a"), row.get("sha_b")) != (request.get("sha_a"), request.get("sha_b")):
            out["sha_mismatch"] += 1
    return out


def stale_on_disk(index: Iterable[dict], current: dict[tuple[str, str, str], str]) -> dict:
    """Judged responses whose text on disk has since changed or disappeared.

    Args:
        index: Request rows (with ``slug``, ``unit``, ``arm_a/b``, ``sha_a/b``).
        current: ``{(arm, slug, unit): sha256}`` of the corpus as it is now.

    Returns:
        ``{"n_judged": int, "n_stale": int, "stale_by_arm": {arm: count}}``,
        counted over distinct ``(arm, slug, unit)`` outputs.
    """
    judged: dict[tuple[str, str, str], str] = {}
    for row in index:
        for side in ("a", "b"):
            key = (row.get(f"arm_{side}", ""), row.get("slug", ""), row.get("unit", ""))
            judged[key] = row.get(f"sha_{side}", "")
    stale: dict[str, int] = {}
    for key, sha in judged.items():
        if current.get(key) != sha:
            stale[key[0]] = stale.get(key[0], 0) + 1
    return {
        "n_judged": len(judged),
        "n_stale": sum(stale.values()),
        "stale_by_arm": dict(sorted(stale.items())),
    }


def file_sha(path: Path) -> str | None:
    """sha256 of a file's bytes, or None when it does not exist."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def records_digest(records_dir: Path) -> str | None:
    """One digest over every evidence record, by file name and content."""
    paths = sorted(records_dir.glob("*.json")) if records_dir.is_dir() else []
    if not paths:
        return None
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((file_sha(path) or "").encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_fingerprint(corpus_dir: Path) -> dict[str, str | None]:
    """Content digests of what an arm run and the reference are built from.

    ``llm_evidence/`` is what both read; the two parquets are where it came from
    (``all_paired``) and what the M tool loop queries directly (``variants_long``).
    """
    return {
        "records": records_digest(corpus_dir / "llm_evidence"),
        "all_paired": file_sha(corpus_dir / "all_paired.parquet"),
        "variants_long": file_sha(corpus_dir / "variants_long.parquet"),
    }


def record_arm_run(out_dir: Path, entry: dict[str, Any]) -> Path:
    """Append one run's provenance to an arm's :data:`ARM_PROVENANCE` file.

    Appended rather than overwritten: ``--only-category`` regenerates part of
    an arm, so its current outputs can come from several runs, and each run's
    inputs have to stay on record for :func:`effective_runs` to check.
    """
    path = out_dir / ARM_PROVENANCE
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}
    stamped = {"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    data["runs"].append({**entry, **stamped})
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def effective_runs(out_dir: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """The recorded runs whose outputs are still on disk, and stamps with no record.

    Returns ``(runs, unrecorded)``: provenance entries whose ``run_id`` appears in
    at least one per-isoform ``*.meta.json`` stamp, and stamped run ids that have
    no provenance entry at all (outputs written before recording existed, or by
    something other than ``run_llm_variants.py``).
    """
    stamped: set[str] = set()
    for name in _STAMPS:
        for stamp in out_dir.glob(f"*/{name}"):
            try:
                run_id = json.loads(stamp.read_text(encoding="utf-8")).get("run_id")
            except (OSError, json.JSONDecodeError):
                run_id = None
            stamped.add(run_id or "")
    path = out_dir / ARM_PROVENANCE
    recorded = json.loads(path.read_text(encoding="utf-8"))["runs"] if path.exists() else []
    runs = [r for r in recorded if r.get("run_id") in stamped]
    return runs, stamped - {r.get("run_id") for r in recorded}


def reconcile(
    arm_runs: dict[str, tuple[list[dict[str, Any]], set[str]]],
    reference_sources: dict[str, str | None],
    *,
    tag_version: str | None = None,
    dist_version: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Decide the reference's versions from the arms, listing every disagreement.

    Args:
        arm_runs: ``{arm: effective_runs(arm_dir)}``.
        reference_sources: :func:`source_fingerprint` of the data the reference
            will be built from.
        tag_version: Explicit tag registry; it must still match what the
            ``tags`` arms ran with, or that is listed as a problem.
        dist_version: Explicit distribution version, checked the same way
            against the ``dist`` arms.

    Returns:
        ``(resolved, problems)``. ``resolved`` holds ``tag_version`` and
        ``dist_version`` (None when no arm used that grounding and nothing was
        given) and a per-arm summary for the record. ``problems`` is empty only
        when every arm's current outputs come from recorded runs, on the same
        source data as the reference, and the ``tags`` / ``dist`` arms agree on
        one registry / distribution version.
    """
    problems: list[str] = []
    tag_seen: set[str] = set()
    dist_seen: set[str] = set()
    per_arm: dict[str, Any] = {}

    for arm, (runs, unrecorded) in sorted(arm_runs.items()):
        if unrecorded:
            problems.append(
                f"{arm}: outputs from {len(unrecorded)} run(s) with no provenance record "
                f"({', '.join(sorted(r or '(unstamped)' for r in unrecorded)[:3])})"
            )
        if not runs and not unrecorded:
            problems.append(f"{arm}: no recorded run produced its outputs")
        for run in runs:
            for key, value in (run.get("sources") or {}).items():
                if value != reference_sources.get(key):
                    problems.append(
                        f"{arm}: run {run.get('run_id')} used a different {key} "
                        "than the reference is built from"
                    )
        category = [r for r in runs if r.get("pass") == "category"]
        tag_seen |= {r.get("tag_registry") for r in category if r.get("grounding") == "tags"}
        dist_seen |= {r.get("dist_version") for r in category if r.get("grounding") == "dist"}
        per_arm[arm] = {
            "run_ids": sorted({r.get("run_id") for r in runs}),
            "unrecorded_run_ids": sorted(unrecorded),
            "tag_registry": sorted({str(r.get("tag_registry")) for r in category}),
            "dist_version": sorted({str(r.get("dist_version")) for r in category}),
        }

    def pick(kind: str, seen: set[str], explicit: str | None) -> str | None:
        if len(seen) > 1:
            problems.append(
                f"{kind}: arms ran with {sorted(map(str, seen))}; the reference can carry one"
            )
        if explicit is not None:
            if seen and seen != {explicit}:
                problems.append(
                    f"{kind}: {explicit!r} requested but arms ran with {sorted(map(str, seen))}"
                )
            return explicit
        return next(iter(seen)) if len(seen) == 1 else None

    resolved = {
        "tag_version": pick("tag registry", tag_seen, tag_version),
        "dist_version": pick("distribution version", dist_seen, dist_version),
        "per_arm": per_arm,
    }
    return resolved, problems
