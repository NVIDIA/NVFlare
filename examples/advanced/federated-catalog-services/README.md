# Federated Catalog: Unity Catalog and PostgreSQL in a two-site POC

This second catalog example runs a provisioned FLARE server, two sites and an
admin connection as separate local processes. Each site uses a different real
catalog source hosted in Docker. It complements the [first SimEnv example](../federated-catalog),
which continues to use Parquet/CSV at site-1 and SQLite through `SQLConnector`
at site-2.

| Site | Source | Dataset / table | Synthetic rows |
| --- | --- | --- | ---: |
| site-1 | Unity Catalog OSS, registered Parquet | `sanctuary` / `pets` | 50 |
| site-2 | PostgreSQL, SQLAlchemy reflection | `sanctuary` / `pets` | 70 |

The fixtures use the existing [Unity Catalog Playground sanctuary generator](https://github.com/open-lakehouse/unitycatalog-playground/blob/3310a1f61b345951adc29cdb8f12f9e1d982f88c/marimo-playground/notebooks/sanctuary.py)
and the same `uuid`, `name`, `age`, `adopted` projection as its Delta notebook.
Setup downloads that pinned source and its Apache-2.0 license into ignored
`data/playground/`; preparation calls its standard-library-only generator.
Site seeds differ, so these are separate synthetic populations. This catalog-only
example writes Parquet with PyArrow and registers it through the Unity Catalog
API; Spark, Delta processing and the Playground notebook stack are unnecessary.

## Install and run

Prerequisites: Docker with Compose v2 running, Python 3.10+, and a Python
virtual environment with this FLARE checkout installed. Activate the environment
before running these commands so provisioned POC processes use the same Python.
Allow outbound HTTPS for the pinned Playground assets and Docker image pulls.
From this directory:

```bash
python -m pip install -r requirements.txt
python setup.py
python prepare_data.py
python job.py
python -m json.tool discovery.json
python setup.py stop
```

1. **Setup** creates a private local `.env`, starts Unity Catalog 0.7.0 and
   PostgreSQL 17 in Docker, downloads preparation assets and provisions an
   isolated two-site POC under `workspace/`. It installs `CatalogManager` in
   the server's site-owned resources. Existing workspaces are refused rather
   than overwritten. Ports 18080 and 55432 are reserved for the two source
   services; FLARE's local POC uses 8002/8003. Stop other POCs before running.
2. **Prepare data** creates the Parquet fixture and registers it as
   `flare_demo.sanctuary.pets` in Unity Catalog. It creates/populates PostgreSQL's
   `sanctuary.pets`, creates a metadata reader, and installs the checked-in
   configurations/policies into each provisioned site's `local/fdn/`.
   Re-running preparation replaces only these demo table fixtures.
3. **Run** starts the provisioned POC, submits an ordinary `CatalogRecipe`
   through `ProdEnv`, waits for the job and server reconciliation, queries the
   authenticated catalog admin API and saves `discovery.json`. `ProdEnv` here
   addresses a local POC; it does not use simulation-only catalog overrides.
   POC processes are stopped in `finally`; Docker services stay up until step 5.
4. **Inspect** expects two dataset catalogs containing three approved columns:
   `uuid`, `age`, `adopted`. Both policies exclude pet names. Physical types
   remain source-specific. PostgreSQL reports the `uuid` primary key; UC OSS
   does not expose primary/foreign-key constraints in its table metadata.
5. **Stop** stops any remaining POC services and runs `docker compose down`.
   PostgreSQL's named volume and local preparation assets remain. To reset
   completely, first stop the example, then remove its `workspace/`, `.env`,
   generated assets and Compose volume (`docker compose down --volumes`).

To run again after stopping, keep the prepared POC workspace and run
`docker compose up -d --wait`, `python prepare_data.py`, and `python job.py`.
Preparation re-registers Unity Catalog metadata after its container is recreated.

## Source and publication configuration

The checked-in `config/site-1/catalog_site.json` binds `sanctuary` to the
built-in `unity-catalog` reader with catalog/schema selection. Site-2's file
names `SQLConnector`, references `CATALOG_DATABASE_URL`, and selects the
PostgreSQL `sanctuary` schema. No endpoint credentials or source column schemas
are passed in the federated job. Both sites have their own `catalog_policy.json`.

The source files and rows remain at their respective services. Unity Catalog's
container has a read-only mount of site-1's generated Parquet file. The catalog
connector reads only REST metadata, never that file. PostgreSQL's catalog reader
has schema visibility but no `SELECT` grant on the application table.

The local demo starts both FLARE sites from one environment, so their parent
processes inherit the example's demo connection settings. A deployed federation
must supply separate, site-local identities and secrets to each site's runtime.
The PostgreSQL bootstrap password is randomly generated in ignored `.env`;
the fixed `catalog_reader` password is a local-demo credential only. Unity
Catalog uses its default local unauthenticated endpoint. These settings are for
loopback demonstration and do not qualify production authentication, tenant
isolation, cancellation or Kubernetes deployment.

Logs and server catalog state remain under `workspace/`. Data preparation and
service lifecycle are separate from `job.py`; no training, analytics or data
processing workflow is submitted. Preparation runs against sources only to
construct the example fixtures.
