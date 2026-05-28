"""Plotly figure builders for the V2 per-isoform page."""

from .protein import build_protein_figure
from .transcript import build_transcript_figure

__all__ = ["build_transcript_figure", "build_protein_figure"]
