# Federated Catalog: Unity Catalog and PostgreSQL sites

Discover site-approved metadata from two containerized sites: site-1 uses Unity
Catalog OSS with a Parquet table; site-2 uses PostgreSQL. Source rows stay at
those sites. Each publishes the schema of its `sanctuary.pets` table, excluding
`name` through its local catalog policy.

## Prepare and start

Prerequisites: Docker Compose **2.30 or later** (for automatic demo-data startup
hooks), and a Python environment with this FLARE checkout installed for admin
commands. The commands below run from this example directory; the POC workspace
can be anywhere accessible to Docker.

Generate the standard server, client and admin startup kits:

```bash
# Choose an absolute path; this is an example, not a required location.
export NVFLARE_POC_WORKSPACE=/tmp/nvflare/poc
nvflare poc config --pw "$NVFLARE_POC_WORKSPACE"
nvflare poc prepare -i project.yml
```

The admin kit is generated at
`$NVFLARE_POC_WORKSPACE/catalog-demo/prod_00/catalog-admin@nvidia.com`,
registered as `catalog-admin@nvidia.com` and selected as the active CLI identity.
It stays on your computer. Subsequent admin commands do not need `--startup-kit`.
The example uses its own admin id so it does not collide with the default
`admin@nvidia.com` that other POCs register.

If `catalog-admin@nvidia.com` is already registered to a different workspace that
still exists, `nvflare poc prepare` generates the kits, keeps that registration
and stops with `startup kit id 'catalog-admin@nvidia.com' already exists outside
POC workspace` (exit code 4). Select the new kit explicitly:

```bash
nvflare config add catalog-admin@nvidia.com "$NVFLARE_POC_WORKSPACE/catalog-demo/prod_00/catalog-admin@nvidia.com" --force
nvflare config use catalog-admin@nvidia.com
```

Register the catalog service in the server's normal deployment configuration:
edit `$NVFLARE_POC_WORKSPACE/catalog-demo/prod_00/nvflare-poc-server/local/resources.json`
and append this entry to its existing `components` array:

```json
{
  "id": "catalog_manager",
  "path": "nvflare.app_common.fdn.catalog.server.manager.CatalogManager"
}
```

Keep the other generated settings and components. Compose mounts the server's
workspace, including this `local/resources.json`, into its container.

Start the containers. Each site automatically prepares its own synthetic demo
data during container startup; this is demo initialization, not a federated job.
In a real deployment, the data already exists at each site.

```bash
docker compose up -d --build --wait
docker compose ps
```

Wait for Compose startup to finish successfully and confirm all five services
are running. Then start each FLARE participant with the standard POC CLI:

```bash
docker compose exec nvflare-poc-server nvflare poc start -p nvflare-poc-server --no-wait
docker compose exec nvflare-poc-site-1 nvflare poc start -p site-1 --no-wait
docker compose exec nvflare-poc-site-2 nvflare poc start -p site-2 --no-wait
```

If `docker compose exec` reports `service "nvflare-poc-server" is not running`,
inspect the container state and logs:

```bash
docker compose ps -a
docker compose logs --tail 100 nvflare-poc-server
```

Run `docker compose up -d --build --wait` successfully before retrying the POC
start commands. If no containers are listed, they have not been created in the
current Docker context or were removed. Keep `NVFLARE_POC_WORKSPACE` exported
in this shell; it must point to the workspace where you prepared the kits.

`--no-wait` starts the participant without an admin login inside its container.
Each container mounts only its own participant kit; the admin credentials stay
on the host. The generated startup scripts manage the FLARE processes. There is
no custom process launcher. The server is reachable from your computer on
loopback ports 8002/8003; clients use its Compose hostname from `project.yml`.

## Run, monitor and retrieve

Run these commands on the admin user's computer:

```bash
nvflare config use catalog-admin@nvidia.com
nvflare system status
python job.py
nvflare catalog show --format json > catalog-summary.json
```

Select the demo identity even if another POC previously selected a different
admin. An error that the active kit points to a missing path means the CLI is
still using an old registration. If the demo identity itself points to an old
workspace, use the explicit `nvflare config add ... --force` command above to
update its registration, then select it again.

`job.py` submits `CatalogRecipe`, enables standard client log streaming, and
uses `nvflare job monitor` to display progress through completion. It does not
prepare data, start services or export results. Startup is asynchronous; use
`nvflare system status` to check registration. The recipe's admin login waits
up to 60 seconds for the server.

Inspect job logs and retrieve the published schema for each site:

```bash
nvflare job logs JOB_ID --site all --tail 100
nvflare catalog show --site site-1 --dataset sanctuary --format json > site-1-catalog.json
nvflare catalog show --site site-2 --dataset sanctuary --format json > site-2-catalog.json
```

The standard POC CLI writes participant console logs into the mounted workspace.
For startup and registration progress:

```bash
tail -f "$NVFLARE_POC_WORKSPACE"/catalog-demo/prod_00/{nvflare-poc-server,site-1,site-2}/poc_console.log
```

Use `docker compose logs` for source-service errors. The normal `nvflare job`
CLI also supports job archive downloads; the catalog CLI retrieves the server's
reconciled catalog, which is separate from a job archive.

## Stop and restart

Stop FLARE through the standard CLI on the host, then stop the containers:

```bash
nvflare poc stop
docker compose down
```

Source data, startup kits and catalog state persist. To restart, run Compose up
and the three `nvflare poc start` commands again; do not prepare the POC again.
For a full demo reset, stop FLARE, run `docker compose down --volumes`, then
`nvflare poc prepare -i project.yml --force`. This replaces the configured POC
workspace and identities. The next Compose startup recreates the source fixtures.
Reapply the `CatalogManager` entry to the regenerated server `local/resources.json`
before starting FLARE.

## Deployment configuration

Configure the server and each client before submitting a catalog job. The
settings belong to the deployment; `job.py` submits the workflow and
`prepare_data.py` only creates synthetic source data.

| Configuration | Supplied by | Purpose |
| --- | --- | --- |
| [project.yml](project.yml) | `nvflare poc prepare` | Defines participants and generates their identities, certificates and startup kits. |
| Server `local/resources.json` | Edit the generated file after preparation | Registers the long-lived `CatalogManager` alongside the existing server components. |
| Client `local/fdn/catalog_site.json` | Read-only Compose mount from `config/<site>/catalog_site.json` | Maps Study and dataset names to that site's metadata connector and source. |
| Client `local/fdn/default/catalog_policy.json` | Read-only Compose mount from `config/<site>/catalog_policy.json` | Authorizes publication and filters the metadata that the site shares. |

### Workspace selection

Compose requires `NVFLARE_POC_WORKSPACE`; it has no fallback to a directory in
this example. The variable is also honored by the POC CLI, and `poc config --pw`
persists the selection for future CLI sessions. Keep the variable exported for
Compose commands. To reuse an already configured workspace in a new terminal:

```bash
export NVFLARE_POC_WORKSPACE="$(nvflare poc config --format json | python -c 'import json, sys; print(json.load(sys.stdin)["data"]["poc_workspace_dir"])')"
```

To change the selection, export the new absolute path and run
`nvflare poc config --pw "$NVFLARE_POC_WORKSPACE"` before preparing it. The
container-local `/poc` paths are mount points inside the image; the host kits and
storage come from the selected workspace. Missing kit directories fail startup.

### Server: CatalogManager

Register `nvflare.app_common.fdn.catalog.server.manager.CatalogManager` in the
server's existing `local/resources.json` as shown in the preparation steps.
Keep the generated scheduler, job manager, launcher, logging and component
authorization settings.

`CatalogController` runs inside each catalog job and collects site-approved
metadata. `CatalogManager` runs in the server parent process: it reconciles those
results into a persistent catalog, tracks versions and site health, and registers
the commands used by `nvflare catalog show`. It also coordinates catalog reads
per Study and handles private catalog artifacts during job archival/download.
Without this component, the deployed job's results are not applied to the
persistent catalog and the catalog query commands are unavailable.

The catalog state is stored under `fdn/catalog/` in the server workspace and
survives job completion and ordinary container restarts. Compose mounts the
server's kit/workspace and the job/snapshot storage paths generated by POC
preparation. Regenerating the POC workspace requires registering the manager again.

Refresh is job-driven: rerun `python job.py` to collect source changes. This
example does not schedule periodic discovery. The manager's default health
threshold marks a site's last successful read as stale after two days.

### Clients: catalog_site.json

Each site owns its source bindings. The `studies.default.sanctuary` entry means
that the logical dataset `sanctuary` is available to catalog jobs in the `default`
Study. The same logical dataset name at two sites identifies two separate site
catalogs.

| Site configuration | Connector | Source selected by the binding |
| --- | --- | --- |
| [site-1/catalog_site.json](config/site-1/catalog_site.json) | Built-in `unity-catalog` | Catalog `flare_demo`, schema `sanctuary`, containing the registered `pets` table. |
| [site-2/catalog_site.json](config/site-2/catalog_site.json) | `clinical-db`, an instance of `SQLConnector` | Schema `sanctuary` in the PostgreSQL database selected by its connection URL. |

Site-2 declares its named connector in the `connectors` section. Its `url_env`
argument names `CATALOG_DATABASE_URL`; the URL value and credentials are supplied
at the site through Compose environment settings. Its `connect_args` sets a
five-second connection timeout. Site-1's built-in connector reads
`UNITY_CATALOG_URL` from its own environment. SQLAlchemy inspects the database
schema and Unity Catalog returns REST metadata; the catalog job does not read
business rows. Spark is not required.

### Clients: publication policy

The [site-1 policy](config/site-1/catalog_policy.json) and
[site-2 policy](config/site-2/catalog_policy.json) each set `share: true` for
`sanctuary` and exclude the `name` column from table `pets`. The resulting
published columns are `uuid`, `age` and `adopted`. Dataset names in the policy
must match the site bindings. Missing or invalid policy fails closed.

The policy path is Study-specific: this example uses
`local/fdn/default/catalog_policy.json`. Publication policy controls shared
metadata; source credentials and database permissions are configured separately.
Neither the binding nor the policy is delivered by the federated job. Compose
mounts both files read-only, and the data preparation script does not modify them.

### Credentials and example assumptions

- The admin and Compose use the same Docker host. The host workspace is selected
  through `NVFLARE_POC_WORKSPACE`; no mount depends on the shell's current
  directory. Project `catalog-demo`, initial kit directory `prod_00` and participant
  names still match this example's `project.yml`.
- To run from another directory, pass the absolute Compose file path with
  `docker compose -f /path/to/federated-catalog-services/compose.yaml ...`, and
  absolute paths to `project.yml` and `job.py` for preparation/submission.
  Configuration files and build context resolve relative to the Compose file.
  Keep `NVFLARE_POC_WORKSPACE` exported in that shell.
- The demo uses the `default` Study. `ProdEnv`, the site bindings and the policy
  directory all use that same Study.
- Each FLARE container receives its own participant kit. The admin kit remains
  on the host, selected by `poc prepare`; participant startup uses `--no-wait`
  because admin credentials are not mounted into the containers.
- Host admin connections use `localhost:8003`; sites connect through the Compose
  server hostname on port 8002. `project.yml` supplies these connection settings
  and certificate host names. Source services have no published host ports.
- `CATALOG_DATABASE_URL` is supplied only to site-2. The demo URL uses
  `catalog_reader:local-demo-reader` at `postgres:5432/sanctuary`. This role has
  schema visibility without table `SELECT` permission. `POSTGRES_PASSWORD` is
  the separate bootstrap password used by PostgreSQL and site-2's demo seeding;
  it can be overridden through the shell or `.env`. The server and site-1 do not
  receive these database credentials.
- `UNITY_CATALOG_URL` points site-1 at the local Unity Catalog REST endpoint.
  Authentication is disabled in this Unity Catalog demo. Real deployments must
  provide the appropriate source authentication, network policy and secrets.
  A connector configured with `token_env` requires HTTPS, with HTTP allowed only
  for literal loopback addresses or localhost during local testing.
- Compose startup hooks seed 50 synthetic pets at site-1 and 70 at site-2, then
  preserve existing fixtures on restart. In a real deployment, data already
  exists; supply the bindings, credentials and publication policy for that data.
- The normal POC process launchers run each FLARE parent and its job subprocesses
  inside its participant container. No Docker socket is mounted into a FLARE
  container. Compose starts the containers; the documented POC CLI commands
  start and stop FLARE.

The image builds FLARE from this checkout and includes the pinned
[Unity Catalog Playground sanctuary generator](https://github.com/open-lakehouse/unitycatalog-playground/blob/3310a1f61b345951adc29cdb8f12f9e1d982f88c/marimo-playground/notebooks/sanctuary.py)
and its Apache-2.0 license. No Spark, PyArrow or database driver installation is
needed on the admin's computer. The [first example](../federated-catalog) uses
SimEnv with Parquet/CSV and SQLite.
