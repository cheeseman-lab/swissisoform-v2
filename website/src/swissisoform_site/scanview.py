"""A VCF scan, from the uploaded bytes to the JSON the browser keeps.

**The server holds nothing.** The upload is staged to a temp file only because
``scan()`` reads a path, and the file is gone before the response is built. The
response carries the whole result — enriched hits included — and the browser keeps
it in ``sessionStorage``; every page that shows scan context reads it from there.
That is issue #33's step 1: with no stored digest there is no token, no TTL and no
per-upload field whose ownership has to be argued.

Everything a page used to compute from the stored digest on each view is computed
here once, at scan time, because the browser has no ORF index:

* each hit's figure ``x`` (``frame.plotly_x``) and its ``protein_change`` label;
* the results page's ``derived`` numbers — the assembly check and the funnel's
  ``passing`` / ``in_orf_alleles`` steps.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from swissisoform.variantclass import PROTEIN_LETTERS_1TO3
from swissisoform.variantquery.consequence import (
    MISSENSE,
    START_LOST,
    STOP_GAINED,
    STOP_LOST,
    SYNONYMOUS,
)
from swissisoform.variantquery.frame import plotly_x
from swissisoform.variantquery.portable import from_portable
from swissisoform.variantquery.scan import scan
from swissisoform.variantquery.vcf import VcfLimitExceeded, check_size

#: Shape of the response body, which the browser stores as-is. ``scanstate.js``
#: refuses a stored scan whose schema it does not know, so a deploy that changes
#: these fields cannot have an old tab read new keys (or the reverse).
#:
#: **Bump this whenever the response's structure changes.** ``c2``: hits carry
#: ``tis_slug``.
SCAN_SCHEMA = "c2"

#: Hit fields no page reads. Dropped to keep a capped scan inside the browser's
#: ~5 MB ``sessionStorage`` quota: ``vclass`` is derivable from ``ref``/``alt``, and
#: ``transcript_id`` is already the last field of ``tis_id``.
_DROPPED_HIT_FIELDS = frozenset({"vclass", "transcript_id"})

_CHUNK = 1 << 20

#: Reader caps → (error code, status). Both are 4xx and both are the uploader's
#: file to fix; the codes stay distinct from the compressed-body ``too_large`` so
#: a caller can tell which limit it hit. Shared by the server route and the
#: browser scan, so both report a bad file the same way.
LIMIT_RESPONSES = {
    "decompressed_bytes": ("expands_too_large", 413),
    "line_bytes": ("line_too_long", 422),
}

#: What reading a file that cannot be read as a VCF raises: a truncated or corrupt
#: gzip (``BadGzipFile`` is an ``OSError``, a cut-off stream an ``EOFError``, a bad
#: block ``zlib.error``). Reported to the user as ``unreadable`` (422) on both
#: paths, never as a traceback or a bare 500. Malformed *lines* are not here: the
#: scanner counts those as rejections, so a ``ValueError`` out of it is a bug and
#: stays one.
UNREADABLE = (OSError, EOFError, zlib.error)


def unreadable_message(exc: BaseException) -> str:
    """The one sentence an unreadable file gets, on the server and in the browser."""
    return f"Could not read this file as a VCF: {exc}"


_SLUG_SEPARATORS = re.compile(r"[:.]+")


class StagingError(Exception):
    """The upload could not be written to the temp file."""


@dataclass(frozen=True, slots=True)
class StagedUpload:
    """An upload on disk for the length of one request."""

    path: Path
    #: SHA-256 of the bytes as uploaded — plain or gzipped, never re-encoded.
    vcf_sha256: str


@contextmanager
def staged_upload(
    stream: BinaryIO, *, directory: str | Path | None = None
) -> Iterator[StagedUpload]:
    """Copy an upload to a private temp file for ``scan()``, deleted on exit.

    The bytes are copied **verbatim**: ``open_vcf`` detects gzip by magic bytes, so
    there is nothing to normalise, and the hash rides along in the same pass. The
    file is created ``0600`` by ``mkstemp`` and unlinked however the block exits —
    a finished scan, a ``VcfLimitExceeded`` or anything else.

    Args:
        stream: The uploaded file object. Werkzeug has already spooled anything over
            500 KB to an anonymous temp file, so this is not held in memory.
        directory: Where to stage; the platform temp dir when ``None``.

    Raises:
        StagingError: If the temp file cannot be created (an unwritable, missing or
            full temp dir) or the copy fails. Raised before the block runs, so only a
            failure of the staging itself is reported as one.
    """
    try:
        fd, name = tempfile.mkstemp(prefix="swissisoform-scan-", suffix=".vcf", dir=directory)
    except OSError as exc:
        raise StagingError(f"could not stage upload: {exc}") from exc
    path = Path(name)
    try:
        digest = hashlib.sha256()
        try:
            with os.fdopen(fd, "wb") as out:
                while chunk := stream.read(_CHUNK):
                    digest.update(chunk)
                    out.write(chunk)
        except OSError as exc:
            raise StagingError(f"could not stage upload: {exc}") from exc
        yield StagedUpload(path=path, vcf_sha256=digest.hexdigest())
    finally:
        path.unlink(missing_ok=True)


def with_frame(protein_change: str | None, frame: str) -> str:
    """Name the frame a residue number counts against, in the same breath as the number.

    Both variant populations on the gene figure carry protein notation, and the two
    are numbered in **different coordinate systems**: a scan hit is numbered against
    one ORF, while an annotated variant's HGVSp is copied from ClinVar / gnomAD /
    COSMIC and numbered against that source's canonical transcript. For an
    alternative-TIS isoform those disagree — the isoform starts elsewhere, so
    canonical residue N is not the isoform's residue N.

    They previously hedged asymmetrically, and the wrong way round: the scan side
    named its frame while the annotated string beside it appeared bare and therefore
    more authoritative, when it is the one whose numbering does not apply to the
    isoform being drawn. Routing both through here means neither can be rendered
    without saying which protein it counts against.
    """
    if not protein_change or not frame:
        return protein_change or ""
    return f"{protein_change} · {frame} frame"


def hgvsp_from_hit(aa_ref: str, aa_alt: str, residue: int, consequence: str) -> str:
    """HGVS protein notation for one scan hit, or ``""`` when none is derivable.

    An uploaded variant has no upstream annotator — the pipeline's own variants carry
    whatever VEP / ClinVar / COSMIC wrote — so the notation is built here from the
    classifier's amino acids. Three-letter codes and the same conventions those
    sources use, since both kinds share a tooltip:

    ========================  ==========================
    missense                  ``p.Arg253Glu``
    synonymous                ``p.Arg253=``
    stop gained               ``p.Arg253Ter``
    stop lost                 ``p.Ter253Arg``
    start lost                ``p.Leu1?``
    multi-residue (MNV)       ``p.Phe225_Glu226delinsSerLys``
    ========================  ==========================

    **The residue is numbered against the ORF the hit names, not a transcript.** The
    same nucleotide is a different residue in every ORF containing it, so the caller
    must show the frame alongside — a bare ``p.Arg253Glu`` next to a ClinVar string
    would otherwise read as the same coordinate system when it is not.

    Args:
        aa_ref: Reference residue(s), one letter each; empty for indels, which the
            classifier resolves by length without reading sequence.
        aa_alt: Alternate residue(s), same length as *aa_ref*.
        residue: 0-based residue of the first affected codon.
        consequence: The classifier's term, which decides the notation's shape.

    Returns:
        The notation, or ``""`` when there are no amino acids to name.
    """

    def three(one: str) -> str:
        return "Ter" if one == "*" else PROTEIN_LETTERS_1TO3.get(one.upper(), one)

    if not aa_ref or len(aa_ref) != len(aa_alt):
        return ""

    first = residue + 1
    if consequence == "start_lost":
        # What changed is whether the codon still initiates, not the residue — the
        # same "?" VEP writes for p.Met1?.
        return f"p.{three(aa_ref[0])}{first}?"
    if len(aa_ref) > 1:
        last = first + len(aa_ref) - 1
        inserted = "".join(three(a) for a in aa_alt)
        return f"p.{three(aa_ref[0])}{first}_{three(aa_ref[-1])}{last}delins{inserted}"
    if aa_ref == aa_alt:
        return f"p.{three(aa_ref)}{first}="
    return f"p.{three(aa_ref)}{first}{three(aa_alt)}"


def tis_slug(tis_id: str | None) -> str:
    """URL-safe form of a tis_id: each run of ``:`` and ``.`` becomes ``-``.

    The one copy of the rule for the site: ``data.tis_slug``, the ``slugify``
    template filter and the hits' ``tis_slug`` (which ``variants.js`` puts in isoform
    links) all call this. It must also agree with the LLM dispatcher's
    ``swissisoform.site.llm._tis_slug``, which names the on-disk output directories.
    Only ``:`` and ``.`` are replaced; a ``+`` strand is kept, since URLs carry it
    fine in a path segment.
    """
    return _SLUG_SEPARATORS.sub("-", str(tis_id or "unknown"))


def enrich_hits(hits: list[dict[str, Any]], index: Any) -> list[dict[str, Any]]:
    """Scan hits → the hits the browser stores, with what only the index can say.

    Adds three fields to each hit:

    * ``x`` — the gene figure's x for this hit, in the figure's own coordinates
      (canonical residue 1 at x = 0), so the browser plots it as-is. ``None`` when
      the hit cannot be placed on a protein axis: no residue (e.g. an indel crossing
      an intron) or an ORF the index no longer has. Such hits still belong in the
      results table; they just draw no marker.
    * ``protein_change`` — HGVS notation with its frame named, or a sentence saying
      why there is none. ``""`` when there is no residue at all.
    * ``tis_slug`` — the isoform's URL slug (:func:`tis_slug`), so the browser links
      to the isoform page without a copy of the rule.

    The residue is meaningless without the frame — a truncation's lost N-terminus is
    numbered against the canonical — which is why ``x`` comes from
    ``frame.plotly_x`` and not from ``residue`` alone.
    """
    out: list[dict[str, Any]] = []
    for hit in hits:
        row = {k: v for k, v in hit.items() if k not in _DROPPED_HIT_FIELDS}
        residue = hit.get("residue")
        frame = hit.get("frame") or ""
        x = None
        protein_change = ""
        if residue is not None:
            record = index.by_tis_id(hit.get("tis_id", "")) if index is not None else None
            if record is not None:
                x = plotly_x(record, int(residue), frame)
            consequence = hit.get("consequence") or "other"
            protein_change = hgvsp_from_hit(
                hit.get("aa_ref") or "", hit.get("aa_alt") or "", int(residue), consequence
            )
            if not protein_change:
                # Absent is a real answer for an indel — the classifier resolves those
                # by length without reading sequence — so say which, rather than
                # leaving a blank line in the tooltip that reads as a bug.
                protein_change = f"residue {int(residue) + 1} — no amino-acid change resolved"
                note = hit.get("consequence_note") or ""
                if note:
                    protein_change += f" ({note})"
            protein_change = with_frame(protein_change, frame)
        row["x"] = x
        row["protein_change"] = protein_change
        row["tis_slug"] = tis_slug(hit.get("tis_id"))
        out.append(row)
    return out


#: Terms whose hits actually had their REF compared against the reference. The
#: classifier reaches ``reference_mismatch`` only on the substitution path
#: (clinical/validate.py) — intronic spans and indels return before that check —
#: so counting them in the denominator would dilute a real mismatch below the
#: threshold on any VCF with many indels or intronic-in-ORF hits.
_REF_CHECKED_TERMS = frozenset(
    {
        MISSENSE,
        SYNONYMOUS,
        STOP_GAINED,
        STOP_LOST,
        START_LOST,
        # Lives as a literal in clinical/validate.py, like "intronic" does in
        # variantquery/scan.py's note table; there is no constant to import.
        "reference_mismatch",
    }
)

#: A REF disagreement or two is ordinary — a miscalled base, a variant normalised
#: against a different representation. A *majority* of them is what an hg19 file
#: looks like scanned against GRCh38 coordinates, which is the one failure mode a
#: user cannot see for themselves: the scan "succeeds" and reports plausible
#: garbage. Below the floor the fraction is noise, so it stays a plain tally.
_ASSEMBLY_MIN_CLASSIFIED = 10
_ASSEMBLY_SUSPECT_FRACTION = 0.5


def assembly_check(counts: dict[str, Any]) -> dict[str, Any]:
    """REF-mismatch tally, and whether it looks like the wrong genome build.

    The classifier already computes ``reference_mismatch`` per hit; this only reads
    the tally it was throwing away.
    """
    consequences = counts.get("consequences") or {}
    mismatches = int(consequences.get("reference_mismatch") or 0)
    checked = sum(int(n or 0) for term, n in consequences.items() if term in _REF_CHECKED_TERMS)
    fraction = mismatches / checked if checked else 0.0
    return {
        "mismatches": mismatches,
        "ref_checked": checked,
        "fraction": fraction,
        "percent": round(fraction * 100),
        "suspect": (checked >= _ASSEMBLY_MIN_CLASSIFIED and fraction >= _ASSEMBLY_SUSPECT_FRACTION),
    }


def derived_counts(counts: dict[str, Any]) -> dict[str, Any]:
    """The results page's computed numbers, which used to be rebuilt on every view."""
    alleles = counts.get("alleles") or 0
    non_pass = counts.get("skipped_non_pass") or 0
    return {
        "assembly": assembly_check(counts),
        "passing": max(alleles - non_pass, 0),
        # Alleles that landed in an ORF, derived by subtraction so the funnel stays
        # monotonic. counts["hits"] is NOT usable here: it counts (variant, isoform)
        # pairs, so one allele in three isoforms is 3 hits and the last step would
        # appear to grow. Subtraction is also independent of the hit-list cap.
        "in_orf_alleles": max(
            alleles
            - non_pass
            - (counts.get("off_catalog_contig") or 0)
            - (counts.get("no_orf") or 0),
            0,
        ),
    }


def build_response(result: Any, *, index: Any, vcf_sha256: str | None) -> dict[str, Any]:
    """``ScanResult`` → the response body the browser stores.

    The filename is deliberately absent: the browser already has it from the
    ``File`` it uploaded, so the server never needs to hold or echo it.
    """
    digest = result.to_dict()
    counts = digest["counts"]
    return {
        "schema": SCAN_SCHEMA,
        "counts": counts,
        "genes": digest["genes"],
        "provenance": {
            "vcf_sha256": vcf_sha256,
            "index_version": index.version,
            "catalog_genes": index.n_genes,
            "catalog_isoforms": index.n_isoforms,
        },
        "derived": derived_counts(counts),
        "hits": enrich_hits(digest["hits"], index),
    }


def load_portable_index(path: str | Path) -> Any:
    """The browser's index: :mod:`variantquery.portable` bytes read from ``path``."""
    return from_portable(Path(path).read_bytes())


def scan_file(path: str | Path, index: Any, *, max_seconds: float = 0) -> dict[str, Any]:
    """Scan a VCF already on disk, returning the response body.

    The browser scan's entry point (Pyodide, ``static/js/scan_worker.js``): the
    worker writes the dropped file into Pyodide's in-memory filesystem and calls
    this, so the result is the very dict the server route returns. The wall-clock
    cap is off by default — it exists for gunicorn's request timeout, and a Web
    Worker has none. The record and byte caps in ``scan`` / ``vcf`` still apply.

    ``provenance.vcf_sha256`` is ``None``: hashing would read the whole file a
    second time in WebAssembly for a value only the server's log reads. The
    server route hashes while it stages the upload, for free.

    Raises:
        VcfLimitExceeded: As ``scan`` does; see :func:`scan_file_json`.
    """
    result = scan(path, index, max_seconds=max_seconds)
    return build_response(result, index=index, vcf_sha256=None)


def _refusal(error: str, message: str) -> str:
    """The server route's ``{error, message}`` shape, as JSON."""
    return json.dumps({"error": error, "message": message}, separators=(",", ":"))


def scan_file_json(path: str | Path, index: Any) -> str:
    """:func:`scan_file` as JSON, with a refused file as ``{error, message}``.

    One string across the Pyodide boundary, rather than a proxied dict, so the
    worker posts exactly what ``JSON.parse`` gives the page. The error shapes are
    the server route's, so ``vcf_drop.js`` reports both paths the same way: a cap
    as its ``LIMIT_RESPONSES`` code, a file that cannot be read as ``unreadable``.
    """
    try:
        body = scan_file(path, index)
    except VcfLimitExceeded as exc:
        return _refusal(LIMIT_RESPONSES[exc.kind][0], str(exc))
    except UNREADABLE as exc:
        return _refusal("unreadable", unreadable_message(exc))
    return json.dumps(body, separators=(",", ":"))


def size_refusal(n_bytes: int) -> str:
    """The refusal for a file too large to scan, as JSON — or ``""`` when it may be.

    Called by the browser worker *before* it copies the file into memory, so a
    multi-gigabyte VCF is refused at once instead of filling the tab first. The
    same error :func:`scan_file_json` would give after the copy.
    """
    try:
        check_size(n_bytes)
    except VcfLimitExceeded as exc:
        return _refusal(LIMIT_RESPONSES[exc.kind][0], str(exc))
    return ""
