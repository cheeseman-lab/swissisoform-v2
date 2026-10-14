r"""Export the tag registry a run was fired with as ``tag_registry.json`` for the site.

The website shows each isoform's fired tags (``isoform_tags_states`` /
``isoform_tags_labels``) but colours and groups them by registry metadata —
category, kind, the criterion a derived tag mirrors — which the parquet does not
carry. The deploy vendors only part of ``swissisoform`` and not ``tags/``, so the
site cannot load the registry itself; this writes the few fields it needs, plus
the registry's own content hash so the site can tell when the staged metadata and
the parquet's tags came from different builds.

The version is read from the parquet's ``isoform_tags_registry_version`` (one
column, projected), so the export always matches what the rows were fired with.

Usage:
    python scripts/export/export_site_tag_registry.py \
        --parquet website/data/all_paired.parquet --out website/data/tag_registry.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from swissisoform.tags import registry as tag_registry

logger = logging.getLogger("export_site_tag_registry")

VERSION_COLUMN = "isoform_tags_registry_version"


def run_version(parquet: Path) -> str | None:
    """The single registry version the run's tags were fired with, or None."""
    names = pq.read_schema(parquet).names
    if VERSION_COLUMN not in names:
        return None
    col = pq.read_table(parquet, columns=[VERSION_COLUMN]).column(VERSION_COLUMN)
    versions = [v for v in pc.unique(col).to_pylist() if v]
    if len(versions) > 1:
        raise SystemExit(f"{parquet} mixes tag registry versions {versions}")
    return versions[0] if versions else None


def site_payload(reg: tag_registry.TagRegistry) -> dict:
    """The fields the site renders, in registry-file order."""
    return {
        "version": reg.version,
        "sha256": reg.sha256,
        "provisional": reg.provisional,
        "tags": [
            {
                "tag_id": t.tag_id,
                "category": t.category,
                "kind": t.kind,
                "label": t.label,
                "criterion_id": t.criterion_id,
                "code_fired": t.code_fired,
            }
            for t in reg
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parquet", type=Path, required=True, help="the staged all_paired.parquet")
    ap.add_argument("--out", type=Path, required=True, help="where to write tag_registry.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    version = run_version(args.parquet)
    if version is None:
        # A pre-tag run: nothing to colour. Remove a stale export rather than leave
        # one describing some other run's tags.
        args.out.unlink(missing_ok=True)
        logger.info("no %s in %s — no tag registry staged", VERSION_COLUMN, args.parquet)
        return
    reg = tag_registry.load(version)
    args.out.write_text(json.dumps(site_payload(reg), indent=1), encoding="utf-8")
    logger.info(
        "wrote %s (registry %s, %d tags, sha %s…)", args.out, version, len(reg), reg.sha256[:12]
    )


if __name__ == "__main__":
    main()
