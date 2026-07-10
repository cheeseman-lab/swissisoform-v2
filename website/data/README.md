# Website data directory

Populated by the deploy recipe — see the project root `README.md`.

Expected layout:

```
data/
├── all_paired.parquet            # required — pipeline output (cheeseman_13gene)
├── variants_long.parquet         # required — per-variant rows for the clinical panel
├── transcript_skeletons.parquet  # required — exon skeletons for the transcript diagram
├── structures/*.cif              # optional — 3Dmol viewer disabled if absent
├── structures/colors/*.colors.json  # optional — per-residue 3Dmol colouring
└── llm/<slug>/                   # optional — per-isoform LLM (synthesis.json + criteria.json)
```

For local development, symlink the cheeseman_13gene outputs into place (or run
`../prepare_deploy.sh`, which cp -L's them):

```bash
ln -s ../../data/output/cheeseman_13gene/all_paired.parquet          all_paired.parquet
ln -s ../../data/output/cheeseman_13gene/variants_long.parquet       variants_long.parquet
ln -s ../../data/output/cheeseman_13gene/transcript_skeletons.parquet transcript_skeletons.parquet
ln -s ../../data/output/cheeseman_13gene/structures                  structures
mkdir -p llm
```

For Docker / Railway, the directory is baked into the image at build time
(see `../Dockerfile`).
