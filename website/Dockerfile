FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1

# Install ONLY third-party deps. We deliberately do NOT `pip install .` the
# local package — building the wheel went through a Railway build-cache layer
# that baked a corrupted swissisoform_site/__init__.py ("source code string
# cannot contain null bytes"). Instead the source is copied straight onto
# PYTHONPATH below, so there is no local-package build/cache step at all.
RUN pip install --no-cache-dir "flask>=3.0" "gunicorn>=21.2" "pandas>=2.0" "pyarrow>=15.0"

# Source onto the path directly (swissisoform_site from src/, scripts.site from
# scripts/). prepare_deploy.sh dereferences data/ symlinks and stages scripts/.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/
ENV PYTHONPATH=/app/src:/app
ENV SWISSISOFORM_DATA_DIR=/app/data

# Strip bytecode, then deterministically rewrite the trivial package __init__.py:
# Railway's upload/cache consistently delivers /app/src/swissisoform_site/__init__.py
# with null bytes even though the source is clean ASCII, which crashes import with
# "source code string cannot contain null bytes". Writing it in-image sidesteps that.
# Then fail the build loudly if any OTHER importable .py still has null bytes.
RUN find /app -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null; \
    find /app -name '*.pyc' -delete 2>/dev/null; \
    printf '"""SwissIsoform v2 Flask viewer."""\n__version__ = "0.1.0"\n' \
      > /app/src/swissisoform_site/__init__.py; \
    python - <<'PY'
import os, sys
roots = ["/app/src", "/app/scripts"]
bad = [os.path.join(d, f)
       for r in roots for d, _, fs in os.walk(r) for f in fs
       if f.endswith(".py") and b"\x00" in open(os.path.join(d, f), "rb").read()]
print("NULL-BYTE .py:", bad or "none")
sys.exit(1 if bad else 0)
PY

EXPOSE 8000
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
