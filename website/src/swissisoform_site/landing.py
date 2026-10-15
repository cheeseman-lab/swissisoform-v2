"""Landing-page index: one compact, column-projected payload for the gene table.

The landing table shows every gene and, on expand, every isoform with its
CDLMPS lineup, LLM synthesis tags and fired pipeline tags. At catalogue scale
(tens of thousands of isoforms) that cannot go through :func:`data.load_all`,
which materialises every column of every row (hit lists included) as Python
objects. This reads only the handful of columns the table renders, builds one
JSON document with shared vocabularies and integer indices, and serves it
pre-gzipped from memory.

Everything the gene cards showed is carried over (see
``docs/plans/landing_table.md``), computed by the same helpers the cards use —
:func:`data.cdlmps_flags`, :func:`data.flag_counts`, :func:`data.best_isoform_key`,
:func:`data.read_llm_json` / :func:`data.synthesis_tags_from` — so the two can never
disagree.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from swissisoform.contract import (
    TAG_LABELS_COLUMN,
    TAG_SHA256_COLUMN,
    TAG_STATES_COLUMN,
    TAG_VERSION_COLUMN,
)
from swissisoform_site.data import (
    CARD_GROUPS,
    ISOFORM_TAG_VOCAB,
    _criteria_dict,
    _maybe_int,
    _maybe_str,
    _none_if_nan,
    _split_terms,
    best_isoform_key,
    cdlmps_flags,
    data_dir,
    flag_counts,
    read_llm_json,
    synthesis_tags_from,
    tis_slug,
)

logger = logging.getLogger(__name__)

REGISTRY_FILE = "tag_registry.json"
# The only columns the table needs. Missing ones (an older run without the tag
# layer) are skipped, never an error.
COLUMNS = (
    "gene_name",
    "generef_uniprot_id",
    "generef_keywords",
    "tis_id",
    "orf_type",
    "aa_len",
    "isoform_scoring_existence_score",
    "isoform_scoring_criteria",
    TAG_STATES_COLUMN,
    TAG_LABELS_COLUMN,
    TAG_VERSION_COLUMN,
    TAG_SHA256_COLUMN,
)
# Display order of isoforms within a gene — the order the card dropdown used:
# these types first, then the rest. Shipped in the payload for the client.
ORF_ORDER = ("extended", "truncated")
_ORF_RANK = {o: i for i, o in enumerate(ORF_ORDER)}
# One character per CDLMPS flag state, so a lineup is a short string.
FLAG_CODES = {
    "met": "m",
    "unmet": "u",
    "pending": "p",
    "interesting": "i",
    "neutral": "n",
    "not_interesting": "x",
}
_CATEGORY_ORDER = {g["letter"]: i for i, g in enumerate(CARD_GROUPS)}


@dataclass(frozen=True)
class LandingIndex:
    """The built payload plus what the server-rendered shell needs."""

    body: bytes  # JSON
    body_gz: bytes  # gzip of ``body``
    etag: str
    n_genes: int
    n_isoforms: int
    examples: tuple[str, ...]
    gene_names: frozenset[str] = frozenset()  # every gene listed, for name lookups
    # LLM coverage, counted while the build reads each isoform's files anyway.
    n_categories: int = 0  # isoforms with a categories.json
    n_category_cells: int = 0  # category reads across them
    n_synthesis: int = 0  # isoforms with a synthesis.json


def _load_registry(root: Path) -> dict[str, Any] | None:
    """The staged tag registry export (``prepare_deploy.sh``), or None."""
    path = root / REGISTRY_FILE
    if not path.is_file():
        return None
    # Optional, and read at app import: anything wrong with it degrades to
    # uncoloured chips rather than an app that cannot start.
    try:
        reg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:  # ValueError covers bad JSON and bad UTF-8
        logger.warning("unreadable %s: %s — tag chips render uncategorised", path, e)
        return None
    tags = reg.get("tags") if isinstance(reg, dict) else None
    if not isinstance(tags, list) or not all(
        isinstance(t, dict) and isinstance(t.get("tag_id"), str) for t in tags
    ):
        logger.warning("malformed %s — tag chips render uncategorised", path)
        return None
    return reg


def _tag_vocab(registry: dict[str, Any] | None, fired_ids: set[str]) -> list[dict[str, Any]]:
    """Every tag that can appear, ordered C→S then registry order.

    From the registry when staged: every code-fired tag, plus any other registry
    tag the run was seen firing, so it keeps its category. Tags seen firing but
    absent from the registry are kept, uncategorised, rather than dropped.
    """
    vocab: list[dict[str, Any]] = []
    known: set[str] = set()
    for pos, t in enumerate((registry or {}).get("tags") or []):
        if not t.get("code_fired") and t["tag_id"] not in fired_ids:
            continue
        vocab.append(
            {
                "id": t["tag_id"],
                "category": t.get("category") or None,
                "kind": t.get("kind") or None,
                "criterion": t.get("criterion_id") or None,
                "label": t.get("label") or t["tag_id"],
                "_pos": pos,
            }
        )
        known.add(t["tag_id"])
    for tid in sorted(fired_ids - known):
        vocab.append(
            {
                "id": tid,
                "category": None,
                "kind": None,
                "criterion": None,
                "label": tid,
                "_pos": len(vocab),
            }
        )
    vocab.sort(key=lambda t: (_CATEGORY_ORDER.get(t["category"], 99), t["_pos"]))
    for t in vocab:
        del t["_pos"]
    return vocab


def _encode_flags(flags: dict[str, dict[str, Any]]) -> str:
    """A CDLMPS lineup as one string: per letter, state code + n_met + n_evaluable.

    The tooltip text (``category_flag``'s label) is a function of those three
    characters, so it ships once per distinct cell in ``flag_labels``.
    """
    out = []
    for g in CARD_GROUPS:
        f = flags[g["letter"]]
        # One digit each: a category has at most four member criteria.
        out.append(
            f"{FLAG_CODES[f['state']]}{min(int(f['n_met']), 9)}{min(int(f['n_evaluable']), 9)}"
        )
    return "".join(out)


def _flag_cell_labels(flags: dict[str, dict[str, Any]], into: dict[str, str]) -> str:
    """:func:`_encode_flags`, recording each cell's label in ``into`` as it goes."""
    code = _encode_flags(flags)
    for k, g in enumerate(CARD_GROUPS):
        into.setdefault(code[3 * k : 3 * k + 3], flags[g["letter"]]["label"])
    return code


def _struct(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def build_landing_index(root: Path | None = None) -> LandingIndex:
    """Build the landing payload from ``<root>/all_paired.parquet`` (projected read)."""
    root = root or data_dir()
    parquet = root / "all_paired.parquet"
    llm_dir = root / "llm"
    llm_dir = llm_dir if llm_dir.is_dir() else None
    if not parquet.is_file():
        logger.warning("parquet not found at %s — landing table will be empty.", parquet)
        frame = pd.DataFrame(columns=["gene_name", "tis_id"])
    else:
        present = set(pq.read_schema(parquet).names)
        frame = pd.read_parquet(parquet, columns=[c for c in COLUMNS if c in present])

    registry = _load_registry(root)
    # A row without a gene name has no gene to sit under; ``load_all``'s groupby
    # drops it the same way.
    rows = [r for r in frame.to_dict("records") if _none_if_nan(r.get("gene_name")) is not None]

    fired_ids: set[str] = set()
    for r in rows:
        fired_ids.update(k for k, v in _struct(r.get(TAG_STATES_COLUMN)).items() if v is True)
    tags = _tag_vocab(registry, fired_ids)
    tag_idx = {t["id"]: i for i, t in enumerate(tags)}
    llm_idx = {t: i for i, t in enumerate(ISOFORM_TAG_VOCAB)}

    # The parquet stamps each row with the registry build that fired it; flag a
    # staged export from a different build, since its categories may not match.
    run_shas = {r.get(TAG_SHA256_COLUMN) for r in rows} - {None}
    if registry and run_shas and run_shas != {registry.get("sha256")}:
        logger.warning(
            "staged %s (sha %s) is not the build that fired this run's tags (%s)",
            REGISTRY_FILE,
            registry.get("sha256"),
            sorted(run_shas),
        )

    by_gene: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_gene.setdefault(str(r["gene_name"]), []).append(r)

    keywords: dict[str, int] = {}
    genes_out: list[dict[str, Any]] = []
    n_isoforms = n_categories = n_category_cells = n_synthesis = 0
    label_drift: set[tuple[str, str]] = set()
    flag_labels: dict[str, str] = {}
    for gene_name in sorted(by_gene):
        grp = by_gene[gene_name]
        head = grp[0]
        isoforms = []
        for r in grp:
            tis_id = str(r["tis_id"])
            slug = tis_slug(tis_id)
            orf = _maybe_str(r.get("orf_type")) or ""
            aa = _maybe_int(r.get("aa_len")) or 0
            # Read once each; "present" here is what the cards render from.
            categories = read_llm_json(llm_dir, slug, "categories.json")
            synthesis = read_llm_json(llm_dir, slug, "synthesis.json")
            verdicts = categories or {}
            if categories is not None:
                n_categories += 1
                n_category_cells += len(categories)
            n_synthesis += synthesis is not None
            flags = cdlmps_flags(_criteria_dict(r.get("isoform_scoring_criteria")), verdicts)
            n_green, n_red, _ = flag_counts(flags)
            states = _struct(r.get(TAG_STATES_COLUMN))
            labels = _struct(r.get(TAG_LABELS_COLUMN))
            fired = sorted((tag_idx[k] for k, v in states.items() if v is True and k in tag_idx))
            iso: dict[str, Any] = {
                "t": tis_id,
                "s": slug,  # the page URL's slug, so the client never re-derives it
                "o": orf,
                "a": aa,
                "f": _flag_cell_labels(flags, flag_labels),
                "g": fired,
                "y": [llm_idx[t] for t in synthesis_tags_from(synthesis)],
                "ns": n_green - n_red,
                "ni": n_green,
            }
            # A fired label is a function of (tag, ORF type) (``Tag.label_for``), so
            # it is shipped once per pair. A row that disagreed would mean the tag
            # layer changed that rule: log it rather than ship per-row overrides.
            for k, v in states.items():
                lab = labels.get(k)
                if v is not True or k not in tag_idx or not lab:
                    continue
                per_orf = tags[tag_idx[k]].setdefault("labels", {})
                if per_orf.setdefault(orf, lab) != lab:
                    label_drift.add((k, orf))
            key = best_isoform_key(
                n_green - n_red,
                n_green,
                _maybe_int(r.get("isoform_scoring_existence_score")),
                aa,
            )
            iso["_key"] = key
            isoforms.append(iso)
        # Best over the parquet's row order (first max wins), as the cards' `max`
        # over gene.isoforms did; then the display order, as the dropdown listed.
        best_iso = max(isoforms, key=lambda i: i["_key"])
        isoforms.sort(key=lambda i: _ORF_RANK.get(i["o"], len(ORF_ORDER)))
        best = next(j for j, i in enumerate(isoforms) if i is best_iso)
        for i in isoforms:
            del i["_key"]
        kws = _split_terms(head.get("generef_keywords"))
        for kw in kws:
            keywords.setdefault(kw, len(keywords))
        genes_out.append(
            {
                "n": gene_name,
                "u": _maybe_str(head.get("generef_uniprot_id")),
                "k": [keywords[kw] for kw in kws],
                "b": best,  # the best isoform's ns / ni rank the gene
                "i": isoforms,
            }
        )
        n_isoforms += len(isoforms)

    payload = {
        "groups": [{"letter": g["letter"], "name": g["name"]} for g in CARD_GROUPS],
        "flag_codes": {code: state for state, code in FLAG_CODES.items()},
        "flag_labels": flag_labels,  # 3-char lineup cell -> its tooltip label
        "orf_order": list(ORF_ORDER),  # listed first, in this order; the rest after
        "tags": tags,
        "llm_tags": list(ISOFORM_TAG_VOCAB),
        "keywords": list(keywords),
        "registry": {
            "version": (registry or {}).get("version"),
            "sha256": (registry or {}).get("sha256"),
            "staged": registry is not None,
        },
        "n_genes": len(genes_out),
        "n_isoforms": n_isoforms,
        "genes": genes_out,
    }
    if label_drift:
        logger.warning(
            "tag labels differ between rows of one ORF type (showing the first): %s",
            sorted(label_drift)[:10],
        )
    if registry is None and fired_ids:
        logger.warning("no %s in %s — fired-tag chips render uncategorised", REGISTRY_FILE, root)
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return LandingIndex(
        body=body,
        body_gz=gzip.compress(body, compresslevel=6, mtime=0),
        etag=hashlib.sha1(body).hexdigest()[:20],
        n_genes=len(genes_out),
        n_isoforms=n_isoforms,
        examples=tuple(g["n"] for g in genes_out[:6]),
        gene_names=frozenset(g["n"] for g in genes_out),
        n_categories=n_categories,
        n_category_cells=n_category_cells,
        n_synthesis=n_synthesis,
    )


@lru_cache(maxsize=1)
def landing_index() -> LandingIndex:
    """The landing payload for the configured data dir, built once per process."""
    idx = build_landing_index()
    logger.info(
        "landing index: %d genes, %d isoforms, %.1f KB json / %.1f KB gzip",
        idx.n_genes,
        idx.n_isoforms,
        len(idx.body) / 1024,
        len(idx.body_gz) / 1024,
    )
    # One line so missing LLM JSONs are obvious before clicking through tiles.
    logging.getLogger("swissisoform_site.coverage").warning(
        "%d isoforms loaded; LLM coverage: categories %d/%d (%d/%d cells), synthesis %d/%d",
        idx.n_isoforms,
        idx.n_categories,
        idx.n_isoforms,
        idx.n_category_cells,
        idx.n_isoforms * len(CARD_GROUPS),
        idx.n_synthesis,
        idx.n_isoforms,
    )
    return idx
