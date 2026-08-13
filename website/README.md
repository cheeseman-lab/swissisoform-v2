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
| `POST /api/variants/scan` | Upload a VCF; resolves it against `orf_index.parquet` and returns a scan token |
| `GET /api/variants/<token>.json` | That scan's digest (404 unknown, 410 expired) |

## Data layout

The app reads everything from a single directory whose path is set by the
`SWISSISOFORM_DATA_DIR` environment variable (default `./data`):

```
<DATA_DIR>/
├── all_paired.parquet            # required — one row per (gene, TIS)
├── variants_long.parquet         # required — per-variant rows for the clinical panel
├── transcript_skeletons.parquet  # required — exon skeletons for the transcript diagram
├── orf_index.parquet             # required for /api/variants/scan — whole-catalogue ORF coords
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
# 1. Populate data/ with a run's outputs (or just run ./prepare_deploy.sh,
#    which cp -L's them into place; its default RUN is cheeseman_test).
#    Use cheeseman_test or newer, NOT cheeseman_13gene — that run predates a
#    column rename (generef_uniprot_function, no generef_keywords) that data.py
#    no longer reads, so the gene function text and keyword facet come up empty.
cd /path/to/swissisoform-v2/website
ln -s ../data/output/cheeseman_test/all_paired.parquet          data/all_paired.parquet
ln -s ../data/output/cheeseman_test/variants_long.parquet       data/variants_long.parquet
ln -s ../data/output/cheeseman_test/transcript_skeletons.parquet data/transcript_skeletons.parquet
ln -s ../data/output/cheeseman_test/structures                  data/structures
mkdir -p data/llm  # empty is fine; the site will render placeholders

# The variant-scan endpoint additionally needs an ORF index (see below):
python ../scripts/export/build_orf_index.py --run full_catalog
ln -s ../data/output/full_catalog/orf_index.parquet             data/orf_index.parquet

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

### The variant-query endpoints need one replica

`POST /api/variants/scan` writes uploaded VCFs and their scan digests to the
container's **ephemeral writable layer** (`/tmp/swissisoform-scans` by default,
override with `SWISSISOFORM_SCAN_DIR`). That layer is shared by both gunicorn
workers because they live in one container — but **not across containers**.

So the service must stay at **one replica**. A second replica would take uploads
on one container and serve the follow-up requests from another, which sees
nothing and 404s roughly half of them. This — not durability — is the trigger to
move the scan store to object storage; all filesystem access goes through
`scanstore.py` precisely so that swap is one file.

Two consequences that are fine as designed:

- A redeploy or crash wipes in-flight scans. The read path treats "files gone"
  and "past its TTL" identically, so the user sees *this scan is no longer
  available, upload again* rather than an error.
- Retention is 24 h (`SWISSISOFORM_SCAN_TTL_HOURS`), enforced when a token is
  read and swept lazily at most hourly on the upload path. There is no background
  thread; a per-worker timer would race for no benefit.

Bounded so behaviour never depends on Railway's ephemeral-disk quota (which is
plan-dependent and not readable from inside the container): 100 MB per upload
(`MAX_CONTENT_LENGTH` → JSON 413), 20,000 hits per digest, and a total store
budget with oldest-first eviction (`SWISSISOFORM_SCAN_BUDGET_BYTES`, default
2 GiB).

### `orf_index.parquet` is staged separately from the displayed run

`prepare_deploy.sh` stages `data/orf_index.parquet` from `ORF_INDEX_RUN`
(default `full_catalog`), which is **independent of `RUN`**. The site displays
one small run but should resolve variants against the whole catalogue: the index
is ~1.4 MB for all 3,371 genes, whereas that run's `all_paired.parquet` is
2.09 GB and cannot be baked in or held in RAM.

Build it before deploying, or `prepare_deploy.sh` will refuse to stage:

```bash
python scripts/export/build_orf_index.py --run full_catalog
```

A hit can therefore land in a gene that has no detail page in this build. The
scan digest carries `provenance.catalog_genes` / `catalog_isoforms` so the UI can
say which catalogue was searched — without that, zero hits is indistinguishable
from a misconfigured index.

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
