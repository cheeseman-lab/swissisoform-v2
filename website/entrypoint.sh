#!/bin/bash
# Boot the Flask app under gunicorn. Railway sets $PORT; default to 8000 for
# local docker runs. Two workers is plenty for the read-only MVP; bump if the
# parquet load becomes a bottleneck on cold restarts.
set -e

exec gunicorn \
  --bind "0.0.0.0:${PORT:-8000}" \
  --workers 2 \
  --timeout 60 \
  --access-logfile - \
  --error-logfile - \
  swissisoform_site.app:app
