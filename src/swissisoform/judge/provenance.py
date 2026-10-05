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
"""

from __future__ import annotations

import hashlib
from typing import Iterable

# Written beside requests.jsonl: every request without its prompt.
INDEX_NAME = "requests_index.jsonl"


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
