# Federated Data Catalog

Discover dataset metadata across two simulated sites using `SimEnv`. Like the
federated statistics and Hello PyTorch examples, this example separates data
preparation from a recipe-based `job.py`. It runs on CPU, uses deterministic
synthetic data, and downloads no datasets.

## Install

From this directory in an activated Python environment:

```bash
python -m pip install -r requirements.txt
```

## Prepare data

```bash
python prepare_data.py
```

Each site gets a tiny clinical cohort and an internal dataset under
`/tmp/nvflare/federated-catalog/data/site-N/`. The preparation script also writes
site-owned configuration:

| File under each site | Purpose |
| --- | --- |
| `datasets/clinical/patients.parquet` | Synthetic patient IDs, ages and BMI values |
| `datasets/clinical/catalog.json` | Site-authored descriptions and supported operations |
| `datasets/internal/records.parquet` | Dataset withheld from discovery |
| `local/study_runtime.yaml` | Absolute paths to this site's datasets |
| `local/fdn/catalog_site.json` | Connector configuration and installed operations |
| `local/fdn/default.json` | Publication policy for the `default` study |

Both sites share the clinical schema, exclude the `patient_id` column and owner
metadata, and withhold the internal dataset entirely. All identifiers and records
in this example are synthetic. Re-running preparation restores the initial data
and policies.

## Run

```bash
python job.py
```

The recipe sends `catalog.read` tasks to `site-1` and `site-2` through the simulator.
The Parquet connector reads schemas from file footers. Each executor applies its
site's policy before returning metadata; data rows stay in the site's dataset.
The recipe reconciles results into `CatalogStore`, reopens the store, and searches
it to demonstrate persistent discovery.

The run checks the expected site/dataset set and column/owner exclusions. The
printed JSON and `/tmp/nvflare/federated-catalog/store/discovery.json` contain two
`clinical` entries, each with a `patients` table containing only `age` and `bmi`.
The `internal` dataset does not appear. Catalog/schema version counters start at
1 in a fresh store. Reading unchanged metadata again preserves those counters.
Simulator logs and intermediate metadata manifests are under
`/tmp/nvflare/federated-catalog/simulation/catalog_read/`.

## Change one site's policy

Acting as the local owner of site-2, withdraw its clinical metadata:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path('/tmp/nvflare/federated-catalog/data/site-2/local/fdn/default.json')
policy = json.loads(path.read_text())
policy['datasets']['clinical']['share'] = False
path.write_text(json.dumps(policy, indent=2) + '\n')
PY
python job.py
```

Discovery now returns only site-1's clinical metadata. The second simulation loads
previous names from the persistent store, so site-2 can explicitly withdraw the
metadata published by the first run. To restore sharing, run `prepare_data.py`
and then `job.py` again.

## Customize paths

```bash
python prepare_data.py --data-root /tmp/my-catalog/data
python job.py --data-root /tmp/my-catalog/data \
  --workspace /tmp/my-catalog/simulation --store-root /tmp/my-catalog/store
```

Use a workspace dedicated to this example: `SimEnv` replaces the simulation job
workspace on each run. The data and persistent store live outside that workspace.
`NVFLARE_SIMULATOR_WORKSPACE_ROOT`, when set, overrides `--workspace`.

## How simulation maps to deployment

`client.py` defines `ParquetCatalogReader`, the site-local counterpart of a
federated statistics generator. Implement `CatalogReader.read_catalog()` to
supply another source. The reader returns metadata; the executor applies the
site-owned publication policy and validates it before transmission.

`job.py` uses the public recipe directly:

```python
recipe = CatalogRecipe(
    sites=["site-1", "site-2"],
    catalog_reader=ParquetCatalogReader(),
    site_config_root=data_root,
    output_dir=store_root,
)
recipe.execute(SimEnv(clients=["site-1", "site-2"]))
print(recipe.catalog_results)
```

`CatalogRecipe` owns component wiring, task dispatch, simulation result
collection, reconciliation and persistent output. `site_config_root` is a
SimEnv-only setting locating site-owned configuration on this shared host.
Simulation support is provided by the framework, without custom example
controllers or executors.

This demonstrates connector reads, site policy enforcement, federated task/result
exchange, reconciliation, versioning and persistent search. It does not exercise
admin login, the `nvflare catalog` commands, scheduling, or production artifact
protection. Intermediate manifests remain unsigned in the local simulation
workspace. In a provisioned federation omit `site_config_root` and use
`CatalogRecipe` with the server's `CatalogManager` and site-owned policies and
runtime bindings. The parent manager handles authenticated private result
handoff, persistence and archive/download redaction.
