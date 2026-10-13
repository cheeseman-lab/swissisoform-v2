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

### The variant query runs in the browser

Dropping a VCF on the landing page scans it **in the browser** (issue #33, step
2): `static/js/scan_worker.js` runs Pyodide in a module Web Worker (Pyodide 314
refuses classic workers) and calls the same Python the server would — `variantquery` + `variantclass` +
`scanview`, all stdlib-only — so the file never leaves the user's machine. It
fetches two content-keyed downloads, cached as immutable (a 404 on either means
the page predates a deploy, and the drop zone says to reload):

| URL | What | Built from |
|---|---|---|
| `/api/variants/scanner/<hash>.zip` | the scan package | `browserscan.SCANNER_MODULES`, zipped from the modules this process imports |
| `/api/variants/index/<fingerprint>/f<format>.bin.gz` | the ORF index, packed binary (~2.9 MB for `full_catalog`) | `variantquery.portable`, from the loaded `orf_index.parquet`; keyed by a hash of that file (`data.orf_index_fingerprint`), so any rebuild gets a new URL, and by the file format |

Both are built once per gunicorn worker on first request; there is no extra
deploy artifact for them.

Pyodide itself is loaded from jsDelivr at the pinned version
(`browserscan.PYODIDE_URL`). The result is the same response body the server route returns,
and the browser keeps it in `sessionStorage` (one scan per tab). Measured with the
real worker under Pyodide 314.0.7: ~5 s first-time start-up, then about 2.3×
CPython's scan time.

`POST /api/variants/scan` stays as a **fallback**, offered as a button only when
the browser cannot run the scanner (no Web Workers/WebAssembly, or Pyodide failed
to load) — never automatically, since it uploads the file. It stages the upload to
a private temp file for the length of the request, deletes it before responding,
and returns the result with `Cache-Control: no-store`. Nothing about a scan is kept
on the server either way: no token, no TTL, no store to sweep.

Bounded per request: 100 MB per upload (`MAX_CONTENT_LENGTH` → JSON 413) and
20,000 hits per scan. The only state left on disk is the throttle below, which is
why the service still runs as **one replica**: a second container would keep its
own throttle buckets and double every client's limit.

### The scan is bounded in time and size, not just on disk

`MAX_CONTENT_LENGTH` caps the **compressed** body. Gzip expands ~1000:1, so a
file that passes it can still be hundreds of gigabytes of parsing, and the scan
runs synchronously on two sync workers. Four bounds close that, and the first two
answer 4xx because they are the uploader's file to fix:

| Bound | Default | Override | On breach |
|---|---|---|---|
| Decompressed bytes | 2 GiB | `SWISSISOFORM_VCF_MAX_BYTES` | JSON 413 `expands_too_large` |
| Single line | 4 MiB | `SWISSISOFORM_VCF_MAX_LINE_BYTES` | JSON 422 `line_too_long` |
| Records scanned | 5,000,000 | — (`scan(max_records=…)`) | partial result, `counts.stopped` |
| Wall clock | 45 s (server route only) | — (`scan(max_seconds=…)`) | partial result, `counts.stopped` |

The browser scan keeps the byte, line and record caps but has no wall-clock
cap — a Web Worker has no request timeout. On the server, the last two deliberately sit under gunicorn's 60 s `--timeout`: past that the
worker is killed and the uploader gets a proxy error, where stopping early
returns a partial answer *that says on the page that it is partial*. `0` disables
any of them; the CLI (`python -m swissisoform.variantquery scan`) is unbounded by
default and takes `--max-records` / `--max-seconds`.

### Per-IP throttling — set `SWISSISOFORM_TRUST_PROXY=1` in production

The upload endpoint is unauthenticated on a public URL, so uploads are rate
limited per client: **5 per hour and 20 per day** by default
(`SWISSISOFORM_SCAN_RATE_HOURLY` / `SWISSISOFORM_SCAN_RATE_DAILY`; `0` disables
either window). Every upload is parsed and charged — with nothing stored there is
no cache hit to exempt. Over the limit is a JSON 429 with `Retry-After`.

State is one marker file per request under `SWISSISOFORM_THROTTLE_DIR` (default
`/tmp/swissisoform-throttle`), hashed by IP, counted by mtime, and purged at most
hourly from the upload path. Files rather than a counter because two workers
writing a counter lose updates; the one-replica caveat above applies. It **fails open** — a disk that refuses the write logs a warning
and lets the request through, since the per-request bounds above still hold.

**Railway fronts the container with its own proxy, so `remote_addr` is the proxy
hop and every client would share one bucket.** Set `SWISSISOFORM_TRUST_PROXY=1`
on the deployment to read the client IP from `X-Forwarded-For` — without it the
throttle is inert in production. It is off by default because trusting that
header with no proxy in front would let anyone spoof their way to a fresh bucket.

`SWISSISOFORM_SCAN_TOKEN` remains available and unset: it closes the endpoint to
anyone without a shared secret, at the cost of the public drop zone on the
landing page. Throttling is the configuration that keeps that form working.

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
