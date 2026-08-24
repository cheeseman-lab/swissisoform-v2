"""Coverage accounting for the genome-wide campaign merge.

``scripts/slurm/full_run/merge.py`` is the last gate before a genome-wide catalog
becomes the substrate everything downstream calibrates on, so the cases that must
never pass silently — a failed shard, a stale re-split, a GPU precompute hole —
are pinned here. The script lives outside the package, so it is loaded by path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

MERGE_PY = Path(__file__).resolve().parents[1] / "scripts" / "slurm" / "full_run" / "merge.py"


def _load_merge():
    spec = importlib.util.spec_from_file_location("full_run_merge", MERGE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge = _load_merge()

CAMPAIGN = "testcamp"
SHARDS = {0: ["A", "B"], 1: ["C", "D"], 2: ["E", "F"]}


@pytest.fixture()
def campaign(tmp_path: Path) -> Path:
    """An output root holding a 3-shard campaign with no shard outputs yet."""
    root = tmp_path / "output"
    (root / CAMPAIGN / "shards").mkdir(parents=True)
    for k, genes in SHARDS.items():
        (root / CAMPAIGN / "shards" / f"shard_{k}.txt").write_text("\n".join(genes) + "\n")
    (root / CAMPAIGN / "split_manifest.txt").write_text(
        f"campaign\t{CAMPAIGN}\nn_proteins\t6\nn_chunks\t1\n"
        f"n_genes\t6\nn_shards\t{len(SHARDS)}\nshard_size\t2\n"
    )
    return root


def write_shard(
    root: Path,
    k: int,
    genes: list[str],
    *,
    fingerprint_of: int | None = None,
    statuses: dict[str, list[str]] | None = None,
) -> Path:
    """Write shard ``k``'s output. ``fingerprint_of`` stamps shard_meta.json."""
    d = root / f"{CAMPAIGN}_shard_{k}"
    d.mkdir(parents=True, exist_ok=True)
    frame = {"gene_name": genes, "v": range(len(genes))}
    frame.update(statuses or {})
    pd.DataFrame(frame).to_parquet(d / "all_paired.parquet")
    for g in genes:
        pd.DataFrame({"gene_name": [g]}).to_parquet(d / f"{g}_paired.parquet")
    if fingerprint_of is not None:
        listing = root / CAMPAIGN / "shards" / f"shard_{fingerprint_of}.txt"
        d.joinpath("shard_meta.json").write_text(
            json.dumps({
                "campaign": CAMPAIGN,
                "shard": k,
                "shard_list_sha1": hashlib.sha1(listing.read_bytes()).hexdigest(),
            })
        )
    return d


def write_all(root: Path, **kwargs) -> None:
    """Write every shard of the fixture campaign, fingerprinted."""
    for k, genes in SHARDS.items():
        write_shard(root, k, genes, fingerprint_of=k, **kwargs)


def run(root: Path, **kwargs):
    return merge.merge_campaign(CAMPAIGN, root, **kwargs)


def test_complete_campaign_publishes(campaign: Path) -> None:
    write_all(campaign)
    code, report = run(campaign)
    out = campaign / CAMPAIGN
    assert code == 0
    assert set(report["status"]) == {"ok"}
    assert len(pd.read_parquet(out / "all_paired.parquet")) == 6
    assert not (out / "all_paired.partial.parquet").exists()
    assert sorted(p.name for p in out.glob("?_paired.parquet")) == [
        f"{g}_paired.parquet" for g in "ABCDEF"
    ]


def test_missing_shard_withholds_the_catalog(campaign: Path) -> None:
    write_shard(campaign, 0, SHARDS[0], fingerprint_of=0)
    write_shard(campaign, 1, SHARDS[1], fingerprint_of=1)
    code, report = run(campaign)
    out = campaign / CAMPAIGN
    assert code == 1
    assert report.loc[report["shard"] == 2, "status"].iloc[0] == "missing"
    # The half-catalog exists for inspection but never at the published path.
    assert (out / "all_paired.partial.parquet").exists()
    assert not (out / "all_paired.parquet").exists()


def test_refilling_the_missing_shard_publishes(campaign: Path) -> None:
    write_shard(campaign, 0, SHARDS[0], fingerprint_of=0)
    write_shard(campaign, 1, SHARDS[1], fingerprint_of=1)
    assert run(campaign)[0] == 1
    write_shard(campaign, 2, SHARDS[2], fingerprint_of=2)
    code, _ = run(campaign)
    out = campaign / CAMPAIGN
    assert code == 0
    assert len(pd.read_parquet(out / "all_paired.parquet")) == 6
    assert not (out / "all_paired.partial.parquet").exists()  # superseded


def test_allow_partial_publishes_with_the_gap(campaign: Path) -> None:
    write_shard(campaign, 0, SHARDS[0], fingerprint_of=0)
    code, _ = run(campaign, allow_partial=True)
    assert code == 0
    assert len(pd.read_parquet(campaign / CAMPAIGN / "all_paired.parquet")) == 2


def test_legacy_shard_without_a_sidecar_is_unverified_not_fatal(campaign: Path) -> None:
    write_all(campaign)
    (campaign / f"{CAMPAIGN}_shard_1" / "shard_meta.json").unlink()
    code, report = run(campaign)
    assert code == 0
    assert report.loc[report["shard"] == 1, "status"].iloc[0] == "unverified"


def test_stale_membership_is_not_merged(campaign: Path) -> None:
    """A shard re-split under a gene set it never ran must not reach the catalog."""
    write_all(campaign)
    (campaign / f"{CAMPAIGN}_shard_1" / "shard_meta.json").write_text(
        json.dumps({"campaign": CAMPAIGN, "shard": 1, "shard_list_sha1": "deadbeef"})
    )
    code, report = run(campaign)
    assert code == 1
    assert report.loc[report["shard"] == 1, "status"].iloc[0] == "stale"
    merged = pd.read_parquet(campaign / CAMPAIGN / "all_paired.partial.parquet")
    assert set(merged["gene_name"]) == {"A", "B", "E", "F"}


def test_orphan_shard_dir_from_a_bigger_split_is_fatal(campaign: Path) -> None:
    write_all(campaign)
    write_shard(campaign, 3, ["Z"])  # beyond the manifest's 0..2
    code, _ = run(campaign)
    assert code == 1
    assert not (campaign / CAMPAIGN / "all_paired.parquet").exists()


def test_orphan_dir_contributes_no_rows(campaign: Path) -> None:
    """The merge is manifest-scoped, so an out-of-range dir is never read."""
    write_all(campaign)
    write_shard(campaign, 3, ["Z"])
    run(campaign)
    merged = pd.read_parquet(campaign / CAMPAIGN / "all_paired.partial.parquet")
    assert "Z" not in set(merged["gene_name"])


def test_gene_merged_from_two_shards_is_fatal(campaign: Path) -> None:
    write_all(campaign)
    write_shard(campaign, 1, ["C", "D", "A"], fingerprint_of=1)  # A also lives in shard 0
    code, _ = run(campaign)
    assert code == 1


def test_gene_outside_every_shard_list_is_fatal(campaign: Path) -> None:
    write_all(campaign)
    write_shard(campaign, 1, ["C", "D", "ZZZ"], fingerprint_of=1)
    code, report = run(campaign)
    assert code == 1
    assert report.loc[report["shard"] == 1, "n_genes_unexpected"].iloc[0] == 1


def test_genes_dropped_at_collapse_are_counted_not_fatal(campaign: Path) -> None:
    """A gene with no isoforms legitimately vanishes; that is not a failed shard."""
    write_shard(campaign, 0, ["A"], fingerprint_of=0)
    write_shard(campaign, 1, SHARDS[1], fingerprint_of=1)
    write_shard(campaign, 2, SHARDS[2], fingerprint_of=2)
    code, report = run(campaign)
    assert code == 0
    assert report.loc[report["shard"] == 0, "n_genes_dropped"].iloc[0] == 1


def test_unreadable_parquet_is_caught(campaign: Path) -> None:
    """A shard killed mid-write leaves a truncated file that must not pass."""
    write_all(campaign)
    (campaign / f"{CAMPAIGN}_shard_1" / "all_paired.parquet").write_bytes(b"PAR1 truncated")
    code, report = run(campaign)
    assert code == 1
    assert report.loc[report["shard"] == 1, "status"].iloc[0] == "unreadable"


def test_gpu_holes_are_reported(campaign: Path) -> None:
    """A dead GPU chunk shows up as no_cache rows in the shards it touched."""
    write_all(campaign)
    write_shard(
        campaign, 1, SHARDS[1], fingerprint_of=1,
        statuses={
            "isoform_structure_status": ["no_cache", "ok"],
            "isoform_plm_vep_status": ["no_cache", "no_cache"],
            "isoform_sae_status": ["ok", "ok"],
        },
    )
    code, report = run(campaign)
    row = report[report["shard"] == 1].iloc[0]
    assert row["frac_no_cache_structure"] == 0.5
    assert row["frac_no_cache_plm"] == 1.0
    assert row["frac_no_cache_sae"] == 0.0
    # Reported, but refillable by re-running Phase B — so not withholding.
    assert code == 0


def test_legitimate_non_ok_statuses_are_not_holes(campaign: Path) -> None:
    """too_long is a real structural verdict, not a missing precompute."""
    write_all(campaign)
    write_shard(
        campaign, 1, SHARDS[1], fingerprint_of=1,
        statuses={"isoform_structure_status": ["too_long", "no_diff_region"]},
    )
    _, report = run(campaign)
    assert report.loc[report["shard"] == 1, "frac_no_cache_structure"].iloc[0] == 0.0


def test_report_is_written_even_when_the_merge_is_withheld(campaign: Path) -> None:
    write_shard(campaign, 0, SHARDS[0], fingerprint_of=0)
    run(campaign)
    written = pd.read_csv(campaign / CAMPAIGN / "merge_report.tsv", sep="\t")
    assert len(written) == len(SHARDS)


def test_unknown_campaign_exits_two(campaign: Path) -> None:
    code, report = merge.merge_campaign("nope", campaign)
    assert code == 2
    assert report is None


def test_manifest_from_another_campaign_is_refused(campaign: Path) -> None:
    manifest = campaign / CAMPAIGN / "split_manifest.txt"
    manifest.write_text(manifest.read_text().replace(f"campaign\t{CAMPAIGN}", "campaign\tother"))
    code, _ = run(campaign)
    assert code == 2
