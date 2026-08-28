# Website data directory

Populated by the deploy recipe — see the project root `README.md`.

Expected layout:

```
data/
├── all_paired.parquet            # required — pipeline output ($RUN, default cheeseman_test)
├── variants_long.parquet         # required — per-variant rows for the clinical panel
├── transcript_skeletons.parquet  # required — exon skeletons for the transcript diagram
├── orf_index.parquet             # required for /api/variants/scan — see below
├── structures/*.cif              # optional — 3Dmol viewer disabled if absent
├── structures/colors/*.colors.json  # optional — per-residue 3Dmol colouring
└── llm/<slug>/                   # optional — per-isoform LLM (synthesis.json + criteria.json)
```

For local development, symlink a run's outputs into place (or run
`../prepare_deploy.sh`, which cp -L's them):

```bash
ln -s ../../data/output/cheeseman_test/all_paired.parquet          all_paired.parquet
ln -s ../../data/output/cheeseman_test/variants_long.parquet       variants_long.parquet
ln -s ../../data/output/cheeseman_test/transcript_skeletons.parquet transcript_skeletons.parquet
ln -s ../../data/output/cheeseman_test/structures                  structures
mkdir -p llm
```

**Use `cheeseman_test` or newer, not `cheeseman_13gene`.** That run predates a
column rename — it carries `generef_uniprot_function` and no `generef_keywords`,
while `data.py` reads `generef_function` / `generef_keywords` — so staging it
gives a site with no gene function text and an empty keyword facet.

`orf_index.parquet` is staged from a **different** run than the other three
(`ORF_INDEX_RUN`, default `full_catalog`): the site displays one small run but
resolves uploaded variants against the whole catalogue. Build it first, or
`prepare_deploy.sh` refuses to stage:

```bash
python ../../scripts/export/build_orf_index.py --run full_catalog
ln -s ../../data/output/full_catalog/orf_index.parquet             orf_index.parquet
```

For Docker / Railway, the directory is baked into the image at build time
(see `../Dockerfile`).
