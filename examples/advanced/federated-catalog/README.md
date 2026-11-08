# Federated Catalog: Parquet, CSV and SQL sites

Discover site-approved metadata across different backends with one
`CatalogRecipe`. Site-1 exposes synthetic OMOP-shaped Parquet tables and two
CSV datasets; site-2 exposes OMOP-shaped SQLite tables through the general
`SQLConnector`. Each site filters its publication locally. Patient records and
source credentials stay at the sites.

## Quick start

Prerequisites: Docker Compose **2.30 or later** and a Python environment with
this FLARE checkout installed. From this directory:

```bash
python example_setup.py --workspace /tmp/nvflare/catalog-files
docker compose up -d --build --wait
nvflare system status
python job.py
nvflare catalog show
```

Before running `python job.py`, confirm `nvflare system status` shows both
`site-1` and `site-2` registered. If startup is still in progress, wait a few
seconds and rerun the status command.

Setup uses `nvflare provision` to create server, client and admin startup kits,
then selects the host admin identity with `nvflare config add` and `config use`.
The workspace can be any path; the default is `./workspace`. Compose starts the
server and both sites with the generated startup scripts. Startup hooks prepare
demo data automatically, preserving it on restart. `job.py` submits the catalog
recipe and monitors progress with the normal CLI.

Stop with `docker compose down`. Kits, source data and catalog state persist.
`docker compose down --volumes` also removes the demo data and stored results.

## Progress and results

```bash
nvflare system status
docker compose logs -f nvflare-files-server nvflare-files-site-1 nvflare-files-site-2
nvflare catalog show --site site-1 --dataset omop
nvflare catalog show --site site-2 --dataset omop
```

The catalog commands above display human-readable results. For JSON export,
add `--format json` and redirect the output to a file.

`job.py` prints a ready-to-run `nvflare job logs` command containing the actual
submitted job ID. Copy that command to inspect the job logs.

Use the standard job CLI to download job archives. Catalog queries return the
current permitted metadata; raw publications are removed from job archives.
Rerun `python job.py` to refresh the catalog after a source or policy change.
Wait for `nvflare system status` to report both clients before submission if the
server is still initializing.

## Deployment configuration

| Configuration | Purpose |
| --- | --- |
| [project.yml](project.yml) | Defines the server, sites, admin, endpoints and certificate host names. Sites use `nvflare-files-server:8102`; the host admin uses `localhost:8103`. |
| Server kit `local/resources.json` | Setup installs [config/server/resources.json](config/server/resources.json), which declares `CatalogManager` and the standard server services. It applies results, stores catalog versions and health, and handles catalog queries. |
| Site kit `local/study_runtime.yaml` | Setup copies the authored [site-1](config/site-1/study_runtime.yaml) and [site-2](config/site-2/study_runtime.yaml) runtime mappings into new kits, preserving existing operator files. These map logical dataset names to site-local paths. |
| `config/<site>/catalog_site.json` | Selects the site's installed connectors. Compose mounts it read-only at `/fdn/policy/catalog_site.json`. |
| `config/<site>/catalog_policy.json` | Defines what the site permits publishing for Study `default`. Compose mounts it read-only at `/fdn/policy/default/catalog_policy.json`. |

[config/server/resources.json](config/server/resources.json) is the server deployment
configuration. CatalogManager is declared directly in its `components` list;
setup simply loads and installs the file. Edit this source file to customize the
example deployment. Rerunning setup replaces the kit's `local/resources.json`
with this configuration. The built-in default component allowlist remains enforced.

Step 4 writes a one-line `.env` file containing `CATALOG_KITS`, just as you would
create for Docker Compose after provisioning. Setup replaces this example-owned
file on each run; keep additional environment overrides in your shell or add them
to `.env` after setup.

The same standard `CatalogController` and `CatalogExecutor` serve every source.
The recipe selects sites and optional dataset names; backend selection belongs
to site configuration, not the job. `CatalogManager` is enabled through normal
server resources configuration, not through another controller.

| Site / dataset | Connector | Local source |
| --- | --- | --- |
| site-1 / `omop` | `ParquetConnector` | `/data/omop`, schema from Parquet footers with declared keys/relationships in `catalog.json` |
| site-1 / `lab_results`, `visits` | `CSVConnector` | `/data/lab_results.csv`, `/data/visits.csv`, headers and bounded type inference |
| site-2 / `omop` | `SQLConnector` | `/data/omop.sqlite`, reflected table schema, primary keys and foreign keys |

Both OMOP publications exclude `person_source_value` and
`condition_source_value`. The two sites have deliberately different condition
columns to illustrate heterogeneous schemas. OMOP-shaped fixtures demonstrate
metadata discovery, not clinical vocabulary validation or OMOP conformance.

`prepare_data.py` only creates synthetic source data. Deployment configuration
is supplied separately by setup and Compose. The standard process launcher runs
jobs inside each site container; `mode: ro` in its runtime mapping is advisory
and does not enforce filesystem permissions. Production sites must enforce
read-only access with filesystem, database or launcher controls.

Each container mounts only its own startup kit read-only, with an ephemeral
working copy in `/workspace` using the startup script's native `NVFL_WORKSPACE`
support. Separate volumes retain source data, server catalog, jobs and
snapshots. The admin kit stays on the host; no Docker socket is mounted.

[example_setup.py](example_setup.py) runs and displays ordinary `nvflare provision`,
`nvflare config add`, `nvflare config use` and `nvflare config list` commands.
Their output is shown directly so the script also serves as a deployment
walkthrough. It is standalone; the service example is not required.
Setup writes the selected kit location to
`.env`; it does not hard-code the workspace. Unset an old exported
`CATALOG_KITS` if it overrides that file. Use `--reprovision` after changing
`project.yml`.

For PostgreSQL and Unity Catalog OSS, see the
[service-backed example](../federated-catalog-services). It also describes
Databricks Unity Catalog configuration and the REST metadata coverage limits.
SQL databases use SQLAlchemy plus the relevant site's database driver; the
recipe and workflow stay the same.
