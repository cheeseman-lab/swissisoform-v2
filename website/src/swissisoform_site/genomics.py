"""Interval-algebra helpers for the gene-page viewer — site-local copy.

The website is a separate self-contained deployable: ``prepare_deploy.sh`` only
vendors ``swissisoform.site.evidence`` from the backend, so the site can't import
``swissisoform.coords`` at runtime. These mirror ``swissisoform.coords``
(``interval_intersection`` / ``interval_length`` + the private ``_normalize``);
they operate on protein-residue-frame intervals for the combined gene figure's
domain deduplication. Half-open ``[start, end)`` intervals.
"""

from __future__ import annotations


def _normalize(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent intervals; drop empties."""
    cleaned = [(s, e) for s, e in intervals if e > s]
    if not cleaned:
        return []
    cleaned.sort()
    merged: list[tuple[int, int]] = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_length(intervals: list[tuple[int, int]]) -> int:
    """Total length of half-open intervals."""
    return sum(end - start for start, end in intervals)


def interval_intersection(
    a: list[tuple[int, int]],
    b: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return ``a ∩ b`` as a sorted, merged list of half-open intervals."""
    a_norm = _normalize(a)
    b_norm = _normalize(b)
    result: list[tuple[int, int]] = []
    i = j = 0
    while i < len(a_norm) and j < len(b_norm):
        a_start, a_end = a_norm[i]
        b_start, b_end = b_norm[j]
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if start < end:
            result.append((start, end))
        if a_end <= b_end:
            i += 1
        else:
            j += 1
    return result
