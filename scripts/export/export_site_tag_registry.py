r"""Export the tag registry a run was fired with as ``tag_registry.json`` for the site.

The website shows each isoform's fired tags (``isoform_tags_states`` /
``isoform_tags_labels``) but colours and groups them by registry metadata —
category, kind, the criterion a derived tag mirrors — which the parquet does not
carry. The deploy vendors only part of ``swissisoform`` and not ``tags/``, so the
site cannot load the registry itself; this writes the few fields it needs, plus
the registry's own content hash so the site can tell when the staged metadata and
the parquet's tags came from different builds.

The version is read from the parquet's ``isoform_tags_registry_version`` (one
column, projected), so the export always matches what the rows were fired with;
the registry's content hash must equal the parquet's
``isoform_tags_registry_sha256``, or nothing is written and it exits non-zero.
The registry is optional for the site (chips render uncoloured without it), so
a registry that is simply not provisioned on this machine is a warning, not a
failure: nothing is staged and the deploy goes on.

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

from swissisoform.modules.tags import SHA256_COLUMN, VERSION_COLUMN
from swissisoform.tags import registry as tag_registry

logger = logging.getLogger("export_site_tag_registry")


def _single(parquet: Path, column: str) -> str | None:
    """The one non-null value of ``column`` across the run, or None if absent."""
    if column not in pq.read_schema(parquet).names:
        return None
    col = pq.read_table(parquet, columns=[column]).column(column)
    values = [v for v in pc.unique(col).to_pylist() if v]
    if len(values) > 1:
        raise SystemExit(f"{parquet} mixes {column} values {values}")
    return values[0] if values else None


def run_version(parquet: Path) -> str | None:
    """The single registry version the run's tags were fired with, or None."""
    return _single(parquet, VERSION_COLUMN)


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
    reg_path = tag_registry.version_dir(version) / tag_registry.REGISTRY_FILE
    if not reg_path.exists():
        args.out.unlink(missing_ok=True)
        logger.warning(
            "WARNING: tag registry %s is not provisioned here (%s) — no %s staged; the "
            "site's fired-tag chips will render uncoloured",
            version,
            reg_path,
            args.out.name,
        )
        return
    reg = tag_registry.load(version)
    run_sha = _single(args.parquet, SHA256_COLUMN)
    if run_sha and run_sha != reg.sha256:
        # Same version name, different definitions: the categories exported here
        # would not be the ones the rows were fired with. A previous export at
        # --out is removed too, as on the other no-export paths.
        args.out.unlink(missing_ok=True)
        raise SystemExit(
            f"registry {version} on disk (sha {reg.sha256[:12]}…) is not the build that "
            f"fired {args.parquet} (sha {run_sha[:12]}…) — rebuild or restage the registry"
        )
    args.out.write_text(json.dumps(site_payload(reg), indent=1), encoding="utf-8")
    logger.info(
        "wrote %s (registry %s, %d tags, sha %s…)", args.out, version, len(reg), reg.sha256[:12]
    )


if __name__ == "__main__":
    main()
