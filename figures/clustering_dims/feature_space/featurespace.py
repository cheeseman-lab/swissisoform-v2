#!/usr/bin/env python
"""Embed isoforms in a CDLMPS-balanced feature space via Multiple Factor Analysis.

The scoring layer thresholds ~19 numbers; ``feature_catalog.csv`` marks 480 of
the pipeline's 747 features usable as dimensions. This module turns those into
coordinates, so isoforms can be placed on a map of protein variation rather than
sorted by a 16-bit criterion vector.

Why MFA rather than plain PCA. The 480 features are grouped into the six CDLMPS
categories, and those groups are wildly unequal: S carries 212 columns, P carries
19. PCA maximizes projected variance and total inertia is the sum of block
inertias, so column count buys influence directly — PC1 would report how finely
biophysics was instrumented, not what varies biologically. MFA divides each block
by its own first singular value, setting every block's *leading* direction to
unit variance before a single global PCA runs. The normalization is deliberately
partial: after weighting, a block's total inertia is ``1 + λ₂/λ₁ + λ₃/λ₁ + …``,
which stays large for a genuinely multi-dimensional block and collapses toward 1
for a block that is one direction plus noise. Blocks get equal a-priori voice and
earn more only by actually varying.

The components are then linear combinations of all columns — free to mix
categories, which is where the cross-block structure lives. Per-block PCA followed
by concatenation (a different, tempting method) would destroy exactly that before
the global step ever saw it.

One matrix, all-ORF: 397 dims x 6,462 isoforms. Drops the shared-region
features, which do not exist for every ORF type, so every ORF type — including
the rare separate ones (uORF, uoORF, internal-OOF, 3'UTR-ORF) — sits in the same
space.

Consumed by ``plot_feature_space.py`` and ``principled_sampler.py`` so both work
from an identical embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from export_feature_catalog import list_lengths, read_flat, resolve_parquet
from scipy.spatial.distance import pdist
from scipy.stats import norm, rankdata

HERE = Path(__file__).resolve().parent
CATALOG_CSV = HERE / "feature_catalog.csv"

CATEGORIES = ("C", "D", "L", "M", "P", "S")

# The 12 hand-curated anchor genes of the cheeseman50 coreset — 9 carried over
# from cheeseman_test plus ASPM, SKA3 and FMR1. Held here rather than read from
# the preset: cheeseman50.toml now lists all 50 isoforms as [[isoforms]] picks
# and has no `genes` key, so the anchor subset is no longer recoverable from it.
# This list is an input to building the coreset, so it cannot be derived from the
# coreset's own output either.
ANCHOR_GENES = frozenset(
    {
        "CBX1",
        "CDC34",
        "EIF2B1",
        "MAD2L1",
        "SRSF2",
        "TRIP13",
        "TRNT1",
        "UBE2D2",
        "UBE2M",
        "ASPM",
        "SKA3",
        "FMR1",
    }
)

# Features the catalog flagged as unavailable (or heavily depleted) for separate
# ORFs. Dropped from the matrix so every ORF type sits in the same feature set.
SHARED_REGION_FLAGS = ("absent_for_separate_orfs", "depleted_for_separate_orfs")

# Carried alongside the scores for colouring and reporting — never dimensions.
# The scoring columns in particular are computed FROM these features, so using
# them as axes would double-count and leak the label.
META_COLUMNS = (
    "gene_name",
    "tis_id",
    "transcript_id",
    "orf_type",
    "diff_region_confidence",
    "start_codon",
    "aa_len",
    "isoform_scoring_existence_score",
    "isoform_scoring_functional_score",
)


# ---------------------------------------------------------------------------
# Column transforms
# ---------------------------------------------------------------------------


def apply_transform(values: np.ndarray, kind: str) -> np.ndarray:
    """Apply the catalog's prescribed transform to one column.

    Compresses heavy tails before normalization: the ``*_ratio`` and
    ``*_enrichment`` families span orders of magnitude and cross zero, and
    untransformed would make PC1 "the isoform with the extreme ratio".
    """
    if kind == "log1p":
        # Counts and lengths: non-negative, right-skewed.
        return np.log1p(np.clip(values, 0, None))
    if kind == "signed_log":
        # Ratios and deltas: heavy-tailed in BOTH directions, so compress
        # magnitude while preserving sign.
        return np.sign(values) * np.log1p(np.abs(values))
    if kind == "neglog10":
        # p-values: small is significant, so flip and expand the small end.
        return -np.log10(np.clip(values, 1e-300, None))
    return values


def rank_to_normal(values: np.ndarray) -> np.ndarray:
    """Map observed values to normal scores by rank; leave NaN as NaN.

    ``q = Φ⁻¹((rank − 0.5) / n_obs)`` over the non-null entries. Preferred to
    z-scoring because it is immune to the outliers the transforms above only
    partly tame, and because it puts every column on an identical scale
    regardless of its original distribution — which is what makes a distance
    across 480 heterogeneous features meaningful at all. The cost is magnitude
    information, which diversity sampling does not need.
    """
    out = np.full(values.shape, np.nan, dtype=float)
    observed = np.isfinite(values)
    n_obs = int(observed.sum())
    if n_obs < 2:
        out[observed] = 0.0
        return out
    ranks = rankdata(values[observed], method="average")
    out[observed] = norm.ppf((ranks - 0.5) / n_obs)
    return out


# ---------------------------------------------------------------------------
# Matrix assembly
# ---------------------------------------------------------------------------


@dataclass
class FeatureMatrix:
    """A normalized feature matrix with its block structure and metadata.

    Attributes:
        X: ``(n, p)`` normalized values, no NaN (gaps imputed to 0).
        features: Column names, in matrix order.
        blocks: CDLMPS letter per column, in matrix order.
        meta: Per-row metadata (gene, ORF type, scores, ...).
        observed: ``(n, p)`` mask — True where measured, False where imputed.
        name: Human label for the matrix (``"all-ORF"``).
    """

    X: np.ndarray
    features: list[str]
    blocks: np.ndarray
    meta: pd.DataFrame
    observed: np.ndarray
    name: str

    def block_observed_fraction(self) -> pd.DataFrame:
        """Per-row fraction of each block's columns that were actually measured.

        A row whose block is mostly imputed sits near that block's centre by
        construction, so anything reading position as evidence needs to know
        which rows those are.
        """
        return pd.DataFrame(
            {
                cat: self.observed[:, self.blocks == cat].mean(axis=1)
                for cat in CATEGORIES
                if (self.blocks == cat).any()
            },
            index=self.meta.index,
        )


def load_catalog() -> pd.DataFrame:
    """Load the feature catalog, restricted to features usable as dimensions."""
    if not CATALOG_CSV.exists():
        raise SystemExit(f"{CATALOG_CSV} not found — run export_feature_catalog.py first.")
    catalog = pd.read_csv(CATALOG_CSV).fillna(
        {"null_pattern": "", "derived_from": "", "transform": "linear"}
    )
    return catalog[catalog["include_in_plot"]].copy()


def _raw_frame(files: list[Path], wanted: list[str]) -> pd.DataFrame:
    """Read raw values for *wanted* features, flattening structs and lists.

    Struct leaves arrive via ``read_flat``; the handful of list-derived
    ``n_<col>`` counts come from list lengths. Only the list columns actually
    needed are decoded — the 876 MB clinical hit lists were caught as
    ``redundant`` by the catalog and never appear here.
    """
    frame, _types = read_flat(files)

    derived = [c for c in wanted if c.startswith("n_") and c not in frame.columns]
    if derived:
        parents = {c[2:] for c in derived}
        schema = pq.ParquetFile(files[0]).schema_arrow
        needed = pa.schema([f for f in schema if f.name in parents])
        for name, series in list_lengths(files, needed).items():
            frame[f"n_{name}"] = series.to_numpy()

    missing = [c for c in wanted if c not in frame.columns]
    if missing:
        raise SystemExit(f"{len(missing)} catalog features absent from the parquet: {missing[:5]}")
    return frame


def build_matrix(frame: pd.DataFrame, catalog: pd.DataFrame, *, name: str) -> FeatureMatrix:
    """Transform, rank-normalize, impute and standardize one feature matrix."""
    features = list(catalog["feature"])
    blocks = catalog["category"].to_numpy()
    transforms = dict(zip(catalog["feature"], catalog["transform"]))

    n, p = len(frame), len(features)
    X = np.empty((n, p), dtype=float)
    observed = np.empty((n, p), dtype=bool)

    for j, feat in enumerate(features):
        raw = pd.to_numeric(frame[feat], errors="coerce").to_numpy(dtype=float)
        col = rank_to_normal(apply_transform(raw, transforms[feat]))
        observed[:, j] = np.isfinite(col)
        # Impute gaps to 0 — the rank-normal median, i.e. the column's centre.
        X[:, j] = np.nan_to_num(col, nan=0.0)

    # Re-standardize exactly. Imputation deflates variance in proportion to
    # missingness, so without this a 60%-filled column would silently carry less
    # weight in the distance than a fully observed one.
    X -= X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    X /= sd

    meta = frame[[c for c in META_COLUMNS if c in frame.columns]].reset_index(drop=True)
    return FeatureMatrix(X, features, blocks, meta, observed, name)


def build_matrix_all_orf(parquet: str | None = None) -> FeatureMatrix:
    """Build the all-ORF matrix from a genome-wide run."""
    files = resolve_parquet(parquet)
    catalog = load_catalog()

    # Identity/coordinate features belong to no CDLMPS block, so MFA has nowhere
    # to put them; exon counts and lengths are dropped rather than given a
    # seventh block that would hand four columns a whole category's voice.
    catalog = catalog[catalog["category"].isin(CATEGORIES)]

    frame = _raw_frame(files, list(catalog["feature"]))

    all_orf_catalog = catalog[~catalog["null_pattern"].isin(SHARED_REGION_FLAGS)]
    return build_matrix(frame, all_orf_catalog, name="all-ORF")


# ---------------------------------------------------------------------------
# Multiple Factor Analysis
# ---------------------------------------------------------------------------


@dataclass
class MFAResult:
    """A fitted MFA embedding plus everything needed to interpret it.

    ``r`` throughout is the retained component count — unrelated to the cluster
    count ``k`` used downstream.

    Attributes:
        scores: ``(n, r)`` row coordinates ``F = UΣ``, unwhitened.
        loadings: ``(p, r)`` right singular vectors.
        eigenvalues: ``(r,)`` global eigenvalues.
        block_weights: Per-block σ₁ divisor.
        blocks: CDLMPS letter per column.
        matrix: The matrix this was fitted on.
    """

    scores: np.ndarray
    loadings: np.ndarray
    eigenvalues: np.ndarray
    block_weights: dict[str, float]
    blocks: np.ndarray
    matrix: FeatureMatrix = field(repr=False)

    def explained(self) -> np.ndarray:
        """Fraction of total inertia per component."""
        return self.eigenvalues / self.eigenvalues.sum()

    def block_contributions(self) -> pd.DataFrame:
        """``Σ_{f∈block} V_fj²`` per component — each column sums to 1.

        Turns a component into a readable category profile, so a cluster's
        coordinates can be described as "conserved, structurally unchanged,
        localization shifted" rather than as bare numbers.
        """
        r = self.loadings.shape[1]
        return pd.DataFrame(
            {
                cat: (self.loadings[self.blocks == cat] ** 2).sum(axis=0)
                for cat in CATEGORIES
                if (self.blocks == cat).any()
            },
            index=[f"PC{j + 1}" for j in range(r)],
        ).T

    def partial_scores(self, cat: str) -> np.ndarray:
        """Row coordinates using only block *cat*'s columns.

        Their mean over blocks recovers the global scores up to a factor, so an
        isoform whose partial scores disagree sharply across categories is one
        where the evidence tells conflicting stories.
        """
        mask = self.blocks == cat
        weighted = self.matrix.X[:, mask] / self.block_weights[cat]
        return weighted @ self.loadings[mask]


def _weight_blocks(X: np.ndarray, blocks: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Divide each block by its own first singular value — MFA's balancing step."""
    Z = X.copy()
    n = X.shape[0]
    weights: dict[str, float] = {}
    for cat in CATEGORIES:
        mask = blocks == cat
        if not mask.any():
            continue
        # σ₁ of the block on the 1/√n-weighted scale used for the global fit.
        sigma1 = float(np.linalg.svd(X[:, mask] / np.sqrt(n), compute_uv=False)[0])
        if sigma1 <= 0:
            sigma1 = 1.0
        weights[cat] = sigma1
        Z[:, mask] = X[:, mask] / sigma1
    return Z, weights


def fit_mfa(matrix: FeatureMatrix) -> MFAResult:
    """Fit MFA: per-block σ₁ weighting, then one global SVD.

    Every component is kept. Truncating would break the distance-preservation
    invariant — ``‖F_i − F_j‖`` equals ``‖Z_i − Z_j‖`` only at full rank — which
    is what licenses measuring coverage in score space rather than on the
    features themselves.
    """
    X, blocks = matrix.X, matrix.blocks
    n = X.shape[0]

    Z, weights = _weight_blocks(X, blocks)
    U, S, Vt = np.linalg.svd(Z / np.sqrt(n), full_matrices=False)

    # Undo the 1/√n used for the eigenvalues so ``‖F_i − F_j‖`` reproduces
    # ``‖Z_i − Z_j‖`` exactly.
    return MFAResult(
        scores=(U * S) * np.sqrt(n),
        loadings=Vt.T,
        eigenvalues=S**2,
        block_weights=weights,
        blocks=blocks,
        matrix=matrix,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(result: MFAResult, *, rtol: float = 1e-6) -> list[str]:
    """Check the invariants that license clustering on the scores."""
    problems: list[str] = []
    X, blocks = result.matrix.X, result.matrix.blocks
    n = X.shape[0]

    # Each weighted block's leading eigenvalue must be exactly 1 — that IS the
    # definition of the σ₁ weighting, and the guarantee no block dominates.
    for cat, w in result.block_weights.items():
        mask = blocks == cat
        lead = float(np.linalg.svd(X[:, mask] / (w * np.sqrt(n)), compute_uv=False)[0])
        if abs(lead**2 - 1.0) > 1e-8:
            problems.append(f"block {cat}: weighted leading eigenvalue {lead**2:.6f} != 1")

    contrib = result.block_contributions().sum(axis=0)
    if not np.allclose(contrib, 1.0, rtol=1e-6):
        dev = float(abs(contrib - 1).max())
        problems.append(f"block contributions do not sum to 1 (max dev {dev:.2e})")

    Z, _ = _weight_blocks(X, blocks)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(200, n), replace=False)
    d_full = pdist(Z[idx])
    d_scores = pdist(result.scores[idx])
    if not np.allclose(d_full, d_scores, rtol=rtol, atol=1e-8):
        worst = float(np.abs(d_full - d_scores).max())
        problems.append(f"scores do not reproduce feature-space distances (max dev {worst:.2e})")
    return problems


def imputation_bias(result: MFAResult) -> float:
    """Correlation of |score| with block observed fraction.

    A strong negative value means heavily-imputed rows are being pulled toward
    the origin, making their coordinates an artifact of missingness rather than
    a measurement.
    """
    radius = np.linalg.norm(result.scores, axis=1)
    observed = result.matrix.observed.mean(axis=1)
    return float(np.corrcoef(radius, observed)[0, 1])


def score_correlation(result: MFAResult, n_pcs: int = 3) -> pd.DataFrame:
    """Correlate the leading components with the evidence scores.

    Strong correlation would mean the embedding rediscovered the ~19 scored
    levers and the other ~461 features contributed nothing.
    """
    meta = result.matrix.meta
    cols = [c for c in meta.columns if c.startswith("isoform_scoring_")]
    if not cols:
        return pd.DataFrame()
    out = {}
    for col in cols:
        vals = pd.to_numeric(meta[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(vals)
        out[col.replace("isoform_scoring_", "")] = [
            float(np.corrcoef(result.scores[ok, j], vals[ok])[0, 1]) for j in range(n_pcs)
        ]
    return pd.DataFrame(out, index=[f"PC{j + 1}" for j in range(n_pcs)])


def summarize(result: MFAResult) -> str:
    """One-paragraph description of a fitted embedding."""
    exp = result.explained()
    blocks = ", ".join(
        f"{c}={int((result.blocks == c).sum())}" for c in CATEGORIES if (result.blocks == c).any()
    )
    sigmas = ", ".join(f"{c}={v:.2f}" for c, v in result.block_weights.items())
    return "\n".join(
        [
            f"{result.matrix.name}: {result.matrix.X.shape[0]:,} isoforms x "
            f"{result.matrix.X.shape[1]} features",
            f"  block sizes:   {blocks}",
            f"  block sigma1:  {sigmas}",
            f"  components: {len(exp)} (full rank, no truncation)",
            "  PC1-PC10 inertia: " + ", ".join(f"{v * 100:.1f}%" for v in exp[:10]),
            "  cumulative at PC5/PC10/PC20: "
            + ", ".join(f"{exp[:k].sum() * 100:.1f}%" for k in (5, 10, 20)),
        ]
    )


__all__ = [
    "ANCHOR_GENES",
    "CATEGORIES",
    "FeatureMatrix",
    "MFAResult",
    "build_matrix_all_orf",
    "fit_mfa",
    "imputation_bias",
    "score_correlation",
    "summarize",
    "verify",
]
