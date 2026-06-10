# Website data directory

Populated by the deploy recipe — see the project root `README.md`.

Expected layout:

```
data/
├── all_paired.parquet     # required — pipeline output (cheeseman_12gene)
├── structures/*.cif       # optional — Mol* viewer disabled if absent
└── llm/<gene>.json        # optional — per-gene LLM interpretation
```

For local development, symlink the cheeseman_12gene outputs into place:

```bash
ln -s ../../data/output/cheeseman_12gene/all_paired.parquet all_paired.parquet
ln -s ../../data/output/cheeseman_12gene/structures           structures
mkdir -p llm
```

For Docker / Railway, the directory is baked into the image at build time
(see `../Dockerfile`).
