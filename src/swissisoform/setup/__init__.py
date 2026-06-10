"""Reference-database setup layer: idempotent fetch/build of external sources.

Logic that downloads and standardizes the reference databases the annotation
modules depend on, plus the UniProt generef fetch. Driven by thin CLIs in
``scripts/setup/``.
"""
