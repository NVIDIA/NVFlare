# Federated Catalog: Unity Catalog and PostgreSQL sites

Read site-approved metadata from two containerized sites. This example uses Docker to simulate a production
deployment: each site's data lives in its own server, a Unity Catalog server at site-1 and a PostgreSQL database at
site-2, which the site reaches over the network. Both hold synthetic patients in the shape of the OMOP Common Data
Model. Source rows stay at those sites. Each site publishes the schema of its `omop` tables as visible to its configured source identity, including the
source-value column names in this synthetic example. The catalog publishes no row values. A
[notebook](catalog_explorer.ipynb) shows the schemas as pandas DataFrames.

The [first example](../federated-catalog) needs none of this. It runs on the host in a simulation, with Parquet, CSV,
SQLite and DuckDB sources.

## Quick start

Prerequisites: Docker Compose **2.30 or later** and Python. Install NVFlare first, then download the matching
example. If NVFlare is already installed, keep that installation and skip the first command; add catalog dependencies
to the same release, nightly wheel or editable installation if needed.

```bash
python -m pip install "nvflare[catalog]"
nvflare examples get federated-catalog-services
cd federated-catalog-services
python -m pip install -r requirements.txt
```

The container build uses this example directory and the same SDK source revision as the installed NVFlare. Setup
records that revision automatically; a repository checkout is not required.

For a contributor SDK with local changes or an unpublished commit, build a wheel from that SDK checkout with
`python -m pip wheel --no-deps . --wheel-dir /tmp/nvflare-sdk-wheels`. Then select its actual host path before
running setup:

```bash
export NVFLARE_PACKAGE="/tmp/nvflare-sdk-wheels/<wheel file>.whl[catalog]"
python example_setup.py --workspace /tmp/nvflare/catalog
```

Setup copies the selected wheel into the ignored `.sdk/` build directory and writes its container path as
`CATALOG_SDK_PACKAGE` in `.env`. The host input `NVFLARE_PACKAGE` can remain exported without overriding that
container path. Keep the wheel path on one line. Clean editable installations whose commits are available on GitHub
can use the default archive build. Dirty or unpublished SDKs need the wheel route because GitHub cannot supply their
local changes.

Then, from this example directory:

```bash
python example_setup.py --workspace /tmp/nvflare/catalog   # create the startup kits and select the admin
docker compose up -d --build --wait                        # start the server and both sites
nvflare system status                                     # confirm both sites are registered
python job.py                                              # read the catalog and save catalog_results.json
```

Before running `python job.py`, confirm `nvflare system status` shows both `site-1` and `site-2` registered. If
startup is still in progress, wait a few seconds and rerun the status command.

Compose prepares both sites automatically during startup. To prepare or refresh just the demo data from the host
after running `example_setup.py`, run:

```bash
python prepare_data.py
```

This starts the source services as needed and runs the Unity Catalog and PostgreSQL seeders in order. It also works
before the FLARE server and clients are started. To prepare a single site, run `docker compose run --rm site-1-seed`
or `docker compose run --rm site-2-seed`.

Setup is safe to rerun, and `--workspace` can be any path (the default is `./workspace`). Kits are tied
to the SDK that created them: if you change the SDK, setup refuses the old kits and asks for `--reprovision`. Source
rows stay at the sites.

Stop with `docker compose down`. Kits and source data are kept for the next `docker compose up -d --wait`. For a
clean run after changing demo credentials:

```bash
docker compose down --volumes --remove-orphans
docker compose up -d --build --wait
```

This reset deletes the current project's demo data and job results. Startup kits are kept. If something goes wrong,
see [Troubleshooting](#troubleshooting).

## The result

`job.py` calls `get_catalog` with a `ProdEnv` for the admin startup kit that setup registered. It submits one catalog
read job to both sites, waits for it, prints a summary and saves the result:

```text
Catalog of Study default, job 3f1c...
  site-1: OK
  site-2: OK
  site-1 / clinical_records [ODCS]: condition_occurrence (16 columns), drug_exposure (23 columns), ...
  site-2 / clinical_records [ODCS]: condition_occurrence (16 columns), drug_exposure (23 columns), ...
Full metadata: catalog_results.json
```

`catalog_results.json` holds each site's status and every published dataset with its tables, columns, types,
nullability, keys and relationships. Each read returns the complete answer; the latest read is the current catalog.
The job's result is also kept in the job store, like any job, so it can be downloaded again with the normal job
commands. If the read fails, `job.py` prints an `nvflare job logs` command with the job's ID.

These standard commands are available when needed:

```bash
docker compose logs -f nvflare-server nvflare-site-1 nvflare-site-2
nvflare system status
nvflare job list
```

## Explore in a notebook

Use [get_catalog.ipynb](get_catalog.ipynb) for the complete workflow after starting the federation:

```bash
jupyter lab get_catalog.ipynb
```

It runs `job.py`, then executes `catalog_explorer.ipynb` on the saved `catalog_results.json`. Its visualization cells
show the fresh results from your run.

[catalog_explorer.ipynb](catalog_explorer.ipynb) follows the federated statistics visualization pattern: load a
result, create `CatalogVisualization`, and call its display functions. It runs offline with the checked-in
[sample result](demo/catalog_results.json), captured from the upstream Unity and PostgreSQL importers in this two-site federation (6 tables / 102 columns and 5 tables / 79 columns respectively). No startup kit, login or
running Docker services are needed. It can be opened independently for an existing result; the runner supplies
`CATALOG_FILE` when invoking it with `%run -i`.

```python
from nvflare.app_opt.fdn.catalog.visualization import CatalogVisualization

vis = CatalogVisualization("catalog_results.json")
vis.show_sites()
vis.show_datasets()
vis.show_schema("person", site="site-1", dataset="clinical_records")
vis.show_table_comparison()
vis.show_column_comparison("person", dataset="clinical_records")
```

Other views list tables and relationships or compare nullability and primary keys. Column headers identify both site
and dataset, so multiple databases at one site remain distinct. Red marks metadata that was not published, gray marks
a dataset that was not read, and yellow marks different reported column definitions. The `get_*` counterparts return
DataFrames; the `show_*` functions return styled tables that Jupyter renders, each with a title and explanation that
also appear when invoked by the runner notebook.

## What setup does

[example_setup.py](example_setup.py) runs these standard steps and prints each command and its output as it goes, so
you can repeat them by hand or reuse them in your own deployment. It invokes the `nvflare` executable from your active
environment. Rerunning it reuses existing kits and updates the three setup-managed keys in the Compose `.env` file.
Your other settings, including `POSTGRES_PASSWORD`, are preserved.

| Step | Command or file | What it sets up |
| --- | --- | --- |
| 1 | `nvflare provision -p project.yml -w <workspace>` | A startup kit (certificates, configuration, start scripts) for the server, each site and the admin, as described in [project.yml](project.yml). |
| 2 | `nvflare config add catalog-admin@nvidia.com <kit> --force`, then `nvflare config use catalog-admin@nvidia.com` | Registers the admin kit as the active identity for `get_catalog`, `nvflare job` and `nvflare system`. The admin kit stays on your computer and is never mounted in a container. |
| 3 | `CATALOG_KITS`, `CATALOG_SDK_PACKAGE` and `CATALOG_SDK_BASE_VERSION` in `.env` | Selects kits, the matching SDK source/wheel and a version fallback derived from that SDK. Local wheels are staged in `.sdk/`. |

Setup finishes with `nvflare config list`, showing the registered admin kits. The server uses the provisioned
default resources; a catalog read needs no server component of its own.

When switching from an older manual setup, unset a previously exported `CATALOG_KITS` if it points elsewhere: shell
variables override Compose's `.env`. Changing the workspace does not reset Docker volumes; use `down --volumes` when
you also want fresh demo data.

## Deployment configuration

| Configuration | Supplied by | Purpose |
| --- | --- | --- |
| [project.yml](project.yml) | `nvflare provision` | Defines participants, endpoints, certificate host names and startup kits. |
| Each site's dataset configuration | `example_setup.py` copies `config/<site>/datasets.yaml` into the site's startup kit as `local/datasets.yaml` | Declares the site's datasets for each Study: where each one is, how to connect, and the source identity used to read metadata. |
| site-2's `CATALOG_DB_USER` and `CATALOG_DB_PASSWORD` | Compose environment | The database credentials its dataset configuration names as secrets. |

That is where a deployed site keeps it, and on a deployed site an edit applies from the next job. In this example each site container copies its read-only kit into a temporary workspace when it starts, so after editing a `config/<site>/datasets.yaml`, rerun `python example_setup.py` and restart the site (`docker compose up -d --force-recreate nvflare-<site>`); the next catalog read uses it.

### Network and workspace

`project.yml` provisions two distinct routes to the same server identity:

- Sites connect to `nvflare-server:8002` through Compose DNS.
- The host admin connects to `localhost:8003` through the published port.

The server's certificate covers both `nvflare-server` and `localhost`. The admin has an explicit `connect_to`
override; clients use the Compose hostname. Source services have no published host ports. Remote deployments must
substitute their own routable host names and provision matching certificates.

Setup stores the selected host kit directory as `CATALOG_KITS` in `.env`. To run from another directory, pass an
absolute Compose path with `docker compose -f /path/to/compose.yaml ...` and an absolute path to `job.py`. Setup
locates `project.yml` relative to its own file.

Each container mounts its own kit read-only at `/startup-kit`. The generated startup script uses its native
`NVFL_WORKSPACE` support to create a working copy in `/workspace`, a container-local tmpfs. Runtime PIDs and copied
credentials therefore do not survive container stop or recreation. Job and snapshot volumes use the standard
provisioned resources paths, so completed jobs and their results remain available through the normal job commands.

### How a read works

A catalog read is an ordinary job. `CatalogController` runs in the job on the server and sends each site one task.
`CatalogExecutor` runs at each site, reads the site's dataset configuration, reads each shared dataset's metadata
with the ODCS reader and returns the source-visible metadata. The controller validates every entry and
writes `catalog/index.json` and standalone ODCS contracts in the job workspace, which is saved to the job store with the job; `get_catalog`
downloads it. The server keeps no catalog of its own.

Rerun `python job.py` to read again after source changes. Use an external scheduler such as cron for regular reads.

### Site dataset configurations

| Site configuration | Dataset type | Source |
| --- | --- | --- |
| [site-1/datasets.yaml](config/site-1/datasets.yaml) | `unity-catalog` | Catalog `flare_demo`, schema `omop` on the site's Unity Catalog server, with its registered tables (`person`, `observation_period`, `visit_occurrence`, `condition_occurrence`, `drug_exposure`, `measurement`). |
| [site-2/datasets.yaml](config/site-2/datasets.yaml) | `sql` | Schema `omop` in the site's `clinical_warehouse` PostgreSQL database (the same tables except `measurement`). |

site-1's configuration:

```yaml
format_version: 1
studies:
  default:
    datasets:
      clinical_records:
        type: unity-catalog
        uri: http://unity-catalog:8080/api/2.1/unity-catalog   # API root; authorization is off in this demo
        options: {catalog: flare_demo, schema: omop}
```

site-2's dataset has `type: sql`, `uri: postgresql://postgres:5432/clinical_warehouse`, the option `schema: omop`, and `secrets: {username: CATALOG_DB_USER, password: CATALOG_DB_PASSWORD}`.

The dataset is named `clinical_records` in the `default` Study at both sites; the same name at two sites identifies two separate
site catalogs. Data Contract CLI imports the Unity Catalog tables through the Databricks SDK and the PostgreSQL schema through psycopg. The job reads no
business rows, and Spark is not required. There is no catalog publication policy; source identities and metadata permissions control visibility. An invalid configuration fails that site's read with `CONFIG_UNAVAILABLE`; a missing one means the site offers the Study no datasets. Jobs supply neither the
configuration nor credentials.

### Source credentials and assumptions

- The upstream PostgreSQL importer receives site-2’s `CATALOG_DB_USER` and `CATALOG_DB_PASSWORD`. The seeder grants the reader `USAGE` on the schema and `REFERENCES` on the demo tables so they appear in `information_schema`. The reader has no table `SELECT` privilege. The seeder uses pg8000. `POSTGRES_PASSWORD` is the separate bootstrap
  password used by PostgreSQL and demo seeding; override it in the shell or `.env`. Set it before the first start:
  PostgreSQL initializes credentials only for an empty volume.
- site-1's Unity Catalog server has authorization disabled in this local demo, so its dataset has no secrets. A
  `token` secret is sent as a bearer token and requires HTTPS, except to a loopback address or localhost.
- The one-shot `site-1-seed` service prepares the Parquet files and Unity Catalog registrations before the FLARE
  client starts. Only this seeder mounts `site-1-data` read-write; the client and Unity Catalog mount it read-only.
- Preparation seeds 150 synthetic patients at site-1 (Parquet files registered in Unity Catalog) and 200 at site-2
  (PostgreSQL tables with primary and foreign keys). Complete fixtures are preserved on restart. PostgreSQL setup
  validates every expected table and row count and repairs incomplete demo data in one transaction; a failed seed
  rolls back. Parquet files are validated before reuse and published with an atomic rename, so an interrupted write is
  retried. [omop_data.py](omop_data.py) generates them with OMOP column names and types and simple clinical patterns;
  the concept ids are a small hand-picked set, not validated against the OHDSI vocabulary. These are demonstration
  data, not a deployment configuration step.
- Matching Unity Catalog registrations are preserved. If columns change, preparation recreates only external Parquet
  registrations pointing to the expected demo file locations; the source files stay in place. A conflicting table
  type, format or location stops preparation rather than replacing it.
- The admin runs on the same Docker host. Admin credentials are not mounted in FLARE containers. Production
  authentication, network policy and secret distribution remain site responsibilities.
- The normal process job launchers run each site's jobs inside its container. No Docker socket is mounted in a FLARE
  container.

The image installs the matching FLARE revision and its catalog dependencies. No Spark, PyArrow or database driver
installation is needed on the admin computer.

### Databricks Unity Catalog

The same Unity Catalog reader reads schema metadata from a Databricks workspace. For example, change site-1's
dataset to:

```yaml
      clinical_records:
        type: unity-catalog
        uri: https://<workspace>/api/2.1/unity-catalog
        options: {catalog: your_catalog, schema: your_schema}
        secrets: {token: DATABRICKS_TOKEN}
```

Supply `DATABRICKS_TOKEN` only to that site's container through its secret configuration. Do not run the demo's
seeder against the production workspace. Grant the token identity permission to discover the intended tables
and verify that its metadata visibility is appropriate for the Study. The catalog does not redact columns. `job.py` and the recipe stay the same. Secrets belong at the site,
never in the job or the published catalog.

The upstream Unity importer uses the Databricks SDK and the [Unity Catalog tables REST API](https://docs.databricks.com/api/uc-tables/v1/list-tables) for
table names, columns and types, with pagination. It does not collect Databricks lineage or table constraints. The
tests exercise the API contract with controlled responses; this example runs Unity Catalog OSS, not a live Databricks
workspace. No Spark server or client is needed for this metadata-only path.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `docker compose` cannot find the kits, or uses an old workspace | Rerun `python example_setup.py --workspace <path>`. A shell variable `CATALOG_KITS` overrides `.env`; `unset CATALOG_KITS`. |
| Changed `project.yml`, or setup says the kits were created for a different NVFlare SDK | `python example_setup.py --workspace <path> --reprovision`, then `docker compose up -d --build --wait` |
| A job ends with `FAILED_TO_RUN` | The server refused to start it. Run `docker compose logs nvflare-server \| grep -E "ERROR\|WARNING"` for the reason. |
| `job.py` cannot connect | The server is still starting. Wait a few seconds, or check `nvflare system status`. |
| A site reports `NO_RESULT` | The site did not answer in time or is not registered. Check `nvflare system status` and `docker compose logs nvflare-site-1` (or `-site-2`). |
| A site reports `FAILED` with `CONFIG_UNAVAILABLE` | Its `config/<site>/datasets.yaml` is invalid. The site's log names the problem. |
| A dataset is listed as unavailable | The site could not read its source; for site-2, check `docker compose logs postgres`. |
| Start over with fresh demo data | `docker compose down --volumes --remove-orphans`, then `docker compose up -d --build --wait`. This deletes this project's current demo data and job results. |
| PostgreSQL is unhealthy, or site-2’s seeder fails | Check `docker compose logs postgres site-2-seed`. Existing volumes retain their initialized database and password; changing `POSTGRES_DB` or `POSTGRES_PASSWORD` does not update them. For disposable demo data, use the clean reset above. |

The site images install the optional `nvflare[catalog]` dependencies (datacontract-cli, PyArrow, DuckDB, psycopg and the Databricks SDK). Their license and notice files are included from the repository's
`3rdParty` directory.
