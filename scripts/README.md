# scripts

`package_server.py` builds the upload for the server that stores the simulation (ROOT) files. That
server only runs the extraction, so the package holds the standalone compactor and the study's
extraction recipes: no training code, models, data or credentials.

```bash
python scripts/package_server.py [--study studies/cg_bbc] [--outdir DIR]
```

It writes `hepml-server-extract.zip` and `hepml-server-extract.zip.sha256`, by default to
`<HEPML_OUTPUT_ROOT>/packages`, outside the repository. The ZIP unpacks into a versioned folder,
`hepml-compactor-<version>/`, so a new version never overwrites an old one:

| Content | Purpose |
| --- | --- |
| `hepml_compact/`, `pyproject.toml`, `run_extraction.sh` | The compactor and its batch runner |
| `constraints-py311.txt` | Pinned library versions for the server environment |
| `study/extraction*.yaml`, `study/samples.yaml` | The extraction recipes (each validated before packing) and the sample manifest |
| `MANIFEST.json` | Version, git commit, whether the packed files had uncommitted changes, SHA-256 of every file |

On the server:

```bash
sha256sum -c hepml-server-extract.zip.sha256 && unzip hepml-server-extract.zip
cd hepml-compactor-<version>
python -m hepml_compact.environment constraints-py311.txt   # the environment matches the pins
bash run_extraction.sh all --dry-run                        # plan every sample of the manifest
nohup bash run_extraction.sh all > extraction.log 2>&1 < /dev/null &
```

`study/samples.yaml` is a template. Keep the filled manifest, with the data paths and cross
sections, beside the versioned folders at `../manifests/samples.ready.yaml`. That is the runner's
default, so every version reads the same manifest.

No installation is needed: `run_extraction.sh` uses the bundled source. It checks the environment
against the pins and resumes an interrupted export. See [compactor/README.md](../compactor/README.md).

Only the compact Parquet exports come back for training; the ROOT files stay on the server.
