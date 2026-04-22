"""Tests for the Newick parser + MRCA-depth utilities."""

from __future__ import annotations

import pytest

from swissisoform.conservation_frame.tree import (
    depth_from_reference,
    find_leaf,
    mrca_depth,
    parse_newick,
)

# ((hg38, panTro6)hominid, (rheMac10, calJac4)simian)mammal;
SIMPLE_NEWICK = "((hg38:1,panTro6:1)hominid:3,(rheMac10:2,calJac4:5)simian:2)mammal;"

# Realistic-ish subset: primates deep, mouse as outgroup
MIXED_NEWICK = (
    "(((hg38,panTro6)hominid,"
    "(rheMac10,calJac4)simian)primate,"
    "mm10)euarchontoglires;"
)


class TestParseNewick:
    def test_simple(self):
        root = parse_newick(SIMPLE_NEWICK)
        assert root.name == "mammal"
        assert len(root.children) == 2

    def test_leaves(self):
        root = parse_newick(SIMPLE_NEWICK)
        assert find_leaf(root, "hg38") is not None
        assert find_leaf(root, "panTro6") is not None
        assert find_leaf(root, "rheMac10") is not None
        assert find_leaf(root, "calJac4") is not None

    def test_missing_leaf_returns_none(self):
        root = parse_newick(SIMPLE_NEWICK)
        assert find_leaf(root, "notApresent") is None

    def test_handles_comments(self):
        nwk = "((hg38[comment],panTro6),mm10);"
        root = parse_newick(nwk)
        assert find_leaf(root, "hg38") is not None

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_newick("   ;")


class TestMrcaDepth:
    def test_self(self):
        root = parse_newick(SIMPLE_NEWICK)
        assert mrca_depth(root, "hg38", "hg38") == 0

    def test_sister(self):
        # hg38 <-> panTro6: MRCA is the `hominid` node, one level up
        root = parse_newick(SIMPLE_NEWICK)
        assert mrca_depth(root, "hg38", "panTro6") == 1

    def test_deeper(self):
        # hg38 <-> rheMac10: MRCA is `mammal`, two levels up
        root = parse_newick(SIMPLE_NEWICK)
        assert mrca_depth(root, "hg38", "rheMac10") == 2

    def test_missing_species(self):
        root = parse_newick(SIMPLE_NEWICK)
        assert mrca_depth(root, "hg38", "missing") is None


class TestDepthFromReference:
    def test_all_leaves(self):
        depths = depth_from_reference(SIMPLE_NEWICK, "hg38")
        assert depths["hg38"] == 0
        assert depths["panTro6"] == 1
        assert depths["rheMac10"] == 2
        assert depths["calJac4"] == 2

    def test_ordering(self):
        # mm10 should be the outgroup → deepest
        depths = depth_from_reference(MIXED_NEWICK, "hg38")
        deepest = max(depths, key=depths.get)
        assert deepest == "mm10"
        assert depths["mm10"] > depths["rheMac10"]
        assert depths["rheMac10"] > depths["panTro6"]

    def test_reference_not_in_tree(self):
        depths = depth_from_reference(SIMPLE_NEWICK, "missing")
        assert depths == {}
