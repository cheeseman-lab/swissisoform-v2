# SwissIsoform v2 — Website

A small Flask viewer over the SwissIsoform v2 paired-evidence parquet. Renders
a 12-gene grid and per-gene pages showing each alternative TIS with its dual
evidence axis (E1–E6 / F1–F6), key metrics, pathogenic variants in the
isoform-unique region, an embedded Mol\* structure viewer, and an LLM-written
interpretation when available.

Mirrors the deploy pattern of `affinage/web/` — Docker on Railway, healthcheck
at `/healthz`, data baked into the image.

## Routes

| Route | Purpose |
|-------|---------|
| `GET /` | 12-gene grid (Existence / Functional bars per gene) |
| `GET /genes/<gene>` | Per-gene page with one section per alternative TIS |
| `GET /api/data.json` | Full parquet + LLM dump as JSON |
| `GET /structures/<file>` | Static `.cif` from `data/structures/` |
| `GET /healthz` | `{"ok": true}` — Railway healthcheck |

## Data layout

The app reads everything from a single directory whose path is set by the
`SWISSISOFORM_DATA_DIR` environment variable (default `./data`):

```
<DATA_DIR>/
├── all_paired.parquet     # required — one row per (gene, TIS)
├── structures/*.cif       # optional — baked AlphaFold/Boltz models
└── llm/<gene>.json        # optional — per-gene LLM interpretation
```

Missing structures, or missing LLM JSONs, degrade gracefully — the page
renders with a "not available" placeholder.

## Local development

```bash
# 1. Populate data/ with a symlink (or copy) of the cheeseman_12gene outputs.
cd /lab/barcheese01/mdiberna/swissisoform-v2/website
ln -s ../data/output/cheeseman_12gene/all_paired.parquet data/all_paired.parquet
ln -s ../data/output/cheeseman_12gene/structures           data/structures
mkdir -p data/llm  # empty is fine; the site will render placeholders

# 2. Install (use the project conda env, or a fresh venv).
eval "$(conda shell.bash hook)" && conda activate swissisoform-v2
pip install -e .

# 3. Run the dev server.
flask --app swissisoform_site.app run --port 5050
#   or:   python -m swissisoform_site.app
```

Then hit:

- http://127.0.0.1:5050/         → the 12-gene grid
- http://127.0.0.1:5050/genes/TRNT1  → per-gene page

## Docker

```bash
cd /lab/barcheese01/mdiberna/swissisoform-v2/website
docker build -t swissisoform-site .
docker run --rm -p 8000:8000 swissisoform-site
```

`data/` is COPYed into the image at build time. For larger data sets, mount a
Railway volume at `/data` and set `SWISSISOFORM_DATA_DIR=/data` instead of
baking.

## Railway deploy

```bash
cd website
railway login
railway init       # one-time
railway up         # builds via Dockerfile, deploys, runs entrypoint.sh
```

`railway.json` already wires the healthcheck at `/healthz`. No env vars are
required for the baked-in-data MVP.

## Updating the data set

1. Re-run the SwissIsoform v2 pipeline (`scripts/run.py` or the relevant
   driver).
2. Refresh the parquet + structures in `website/data/`.
3. (Optional) drop per-gene LLM JSONs into `website/data/llm/`.
4. Rebuild + redeploy (`railway up`).

## What this site does NOT do

- No `swissisoform` package import — the site is standalone Flask + pandas +
  jinja2, deliberately, so the image stays small and the dependency surface
  doesn't track the analysis pipeline.
- No DB. No write path. No on-the-fly LLM calls.
- No build step for assets. Tailwind + Mol\* are both loaded from CDN.
