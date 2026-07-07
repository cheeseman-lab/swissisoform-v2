# SwissIsoform v2 — Website

A small Flask viewer over the SwissIsoform v2 paired-evidence parquet. Renders
a 13-gene grid and per-isoform pages showing each alternative TIS with its dual
evidence axis (E1–E6 / F1–F6), key metrics, pathogenic variants in the
isoform-unique region, an embedded 3Dmol.js structure viewer, and an LLM-written
interpretation when available.

Mirrors the deploy pattern of `affinage/web/` — Docker on Railway, healthcheck
at `/healthz`, data baked into the image.

## Routes

| Route | Purpose |
|-------|---------|
| `GET /` | 13-gene grid (Existence / Functional bars per gene) |
| `GET /about` | Methodology / evidence-criteria explainer |
| `GET /genes/<gene>/isoforms/<slug>` | Per-isoform page (dual evidence axis, metrics, variants, 3Dmol.js viewer, LLM) |
| `GET /genes/<gene>` | Deprecated — 302-redirects to the gene's first isoform page |
| `GET /api/data.json` | Full parquet + LLM dump as JSON |
| `GET /structures/<file>` | Static `.cif` from `data/structures/` |
| `GET /structure-colors/<file>` | Per-residue `.colors.json` from `data/structures/colors/` |
| `GET /healthz` | `{"ok": true}` — Railway healthcheck |

## Data layout

The app reads everything from a single directory whose path is set by the
`SWISSISOFORM_DATA_DIR` environment variable (default `./data`):

```
<DATA_DIR>/
├── all_paired.parquet            # required — one row per (gene, TIS)
├── variants_long.parquet         # required — per-variant rows for the clinical panel
├── transcript_skeletons.parquet  # required — exon skeletons for the transcript diagram
├── structures/*.cif              # optional — baked AlphaFold/Boltz models
├── structures/colors/*.colors.json  # optional — per-residue 3Dmol colouring
└── llm/<slug>/                   # optional — per-isoform LLM interpretation
    ├── synthesis.json            #   the written narrative
    └── criteria.json             #   per-criterion notes
```

Missing structures, colours, or LLM JSONs degrade gracefully — the page renders
with a "not available" placeholder.

## Local development

```bash
# 1. Populate data/ with the cheeseman_13gene outputs (or just run
#    ./prepare_deploy.sh, which cp -L's them into place).
cd /path/to/swissisoform-v2/website
ln -s ../data/output/cheeseman_13gene/all_paired.parquet          data/all_paired.parquet
ln -s ../data/output/cheeseman_13gene/variants_long.parquet       data/variants_long.parquet
ln -s ../data/output/cheeseman_13gene/transcript_skeletons.parquet data/transcript_skeletons.parquet
ln -s ../data/output/cheeseman_13gene/structures                  data/structures
mkdir -p data/llm  # empty is fine; the site will render placeholders

# 2. Install (use the project conda env, or a fresh venv).
eval "$(conda shell.bash hook)" && conda activate swissisoform-v2
pip install -e .

# 3. Run the dev server.
flask --app swissisoform_site.app run --port 5050
#   or:   python -m swissisoform_site.app
```

Then hit:

- http://127.0.0.1:5050/         → the 13-gene grid
- http://127.0.0.1:5050/genes/TRNT1  → redirects to TRNT1's first isoform page

## Docker

```bash
cd /path/to/swissisoform-v2/website
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

- No heavy analysis dependencies — the site imports only the small
  `swissisoform.site.evidence` helper (numpy + pandas), which `prepare_deploy.sh`
  copies into the build context, not the full analysis pipeline. So the image
  stays small and the dependency surface doesn't track the pipeline.
- No DB. No write path. No on-the-fly LLM calls.
- No asset build step at request time. Tailwind is **precompiled** to a tracked
  `static/css/tailwind.css` (from `tailwind.input.css`); Plotly and the 3Dmol.js
  structure viewer are loaded from CDN.
