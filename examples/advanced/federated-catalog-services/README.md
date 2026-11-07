# Federated Catalog: Unity Catalog and PostgreSQL sites

Discover site-approved metadata from two containerized sites: site-1 uses Unity
Catalog OSS with a Parquet table; site-2 uses PostgreSQL. Source rows stay at
those sites. Each publishes the schema of its `sanctuary.pets` table, excluding
`name` through its local catalog policy.

## Provision startup kits

Prerequisites: Docker Compose **2.30 or later** (for automatic demo-data startup
hooks), and a Python environment with this FLARE checkout installed for admin
commands. Run the following commands from this example directory.

Use the standard provision command to generate server, site and admin kits:

```bash
# Example location; choose any absolute path accessible to Docker.
export CATALOG_WORKSPACE=/tmp/nvflare/catalog
nvflare provision -p project.yml -w "$CATALOG_WORKSPACE"
export CATALOG_KITS="$CATALOG_WORKSPACE/catalog-demo/prod_00"
```

Select the generation printed by provisioning. The first is `prod_00`; later
provisioning runs can generate `prod_01`, etc. Compose requires `CATALOG_KITS`
and mounts only each participant's own kit. No POC commands or POC workspace
settings are used.

Register the generated admin kit on your computer and select it:

```bash
nvflare config add catalog-admin@nvidia.com "$CATALOG_KITS/catalog-admin@nvidia.com" --force
nvflare config use catalog-admin@nvidia.com
```

The admin kit stays on the host. Subsequent commands do not need `--startup-kit`.
The explicit registration also replaces any registration left by an older run
of this example.

Activate the server's normal resources configuration:

```bash
cp "$CATALOG_KITS/nvflare-server/local/resources.json.default" \
   "$CATALOG_KITS/nvflare-server/local/resources.json"
```

Edit that `resources.json` and append this entry to its existing `components`
array, retaining the other generated components and authorization settings:

```json
{
  "id": "catalog_manager",
  "path": "nvflare.app_common.fdn.catalog.server.manager.CatalogManager"
}
```

Do this once per newly provisioned server kit; keep an existing customized
`resources.json` rather than overwriting it with the default again.

## Start and monitor

```bash
docker compose up -d --build --wait
docker compose ps
docker compose logs -f nvflare-server nvflare-site-1 nvflare-site-2
```

Compose starts FLARE automatically using the generated
`startup/sub_start.sh` in the foreground. This is the same script invoked by
`startup/start.sh`; running it in the foreground lets Docker track the service.
There is no custom process launcher and no separate `poc start` step.

Each site automatically prepares its own synthetic demo data during container
startup. In a real deployment the data already exists. Preparation only seeds
the sources; deployment configuration is supplied separately by Compose.

`up --wait` waits for container/source-service startup, not FLARE registration.
After inspecting startup logs, check readiness from the host:

```bash
nvflare system status
```

Wait until both sites are registered before submitting a job. If connection
fails, inspect `docker compose logs --tail 100 nvflare-server` and check that
`nvflare config use catalog-admin@nvidia.com` selects the kit provisioned above.

## Run and retrieve

Run these commands on the admin user's computer:

```bash
python job.py
nvflare catalog show --format json > catalog-summary.json
nvflare catalog show --site site-1 --dataset sanctuary --format json > site-1-catalog.json
nvflare catalog show --site site-2 --dataset sanctuary --format json > site-2-catalog.json
nvflare job logs JOB_ID --site all --tail 100
```

`job.py` submits `CatalogRecipe`, enables standard client log streaming, and
uses `nvflare job monitor` to display progress through completion. It does not
prepare data, start services or export results. Use the normal job CLI for job
archive downloads. Catalog commands retrieve the reconciled catalog separately.

## Stop and restart

Finish the demo job, then request shutdown through the generated kit scripts:

```bash
docker compose exec -T nvflare-site-1 bash -c 'printf "y\n" | /startup-kit/startup/stop_fl.sh'
docker compose exec -T nvflare-site-2 bash -c 'printf "y\n" | /startup-kit/startup/stop_fl.sh'
docker compose exec -T nvflare-server bash -c 'printf "y\n" | /startup-kit/startup/stop_fl.sh'
docker compose wait nvflare-site-1 nvflare-site-2 nvflare-server
docker compose down
```

The confirmation is for the generated local process-stop script. Its daemon
observes the shutdown request, stops its worker, and exits. Compose waits for
these foreground startup scripts to finish before removing the containers.

To restart, run `docker compose up -d --wait`; existing kits and source data are
reused, and FLARE starts automatically. No provisioning step is needed.

Named volumes retain source fixtures, server catalog state, archived jobs and
snapshots. `docker compose down --volumes` removes that demo state. Provision a
new generation only when changing participant identities or certificates, select
its directory with `CATALOG_KITS`, register its admin kit, and configure its new
server resources before starting it.

## Deployment configuration

| Configuration | Supplied by | Purpose |
| --- | --- | --- |
| [project.yml](project.yml) | `nvflare provision` | Defines participants, endpoints, certificate host names and startup kits. |
| Server `local/resources.json` | Edit the generated kit | Registers the long-lived `CatalogManager` alongside the existing server components. |
| Client `/fdn/policy/catalog_site.json` | Read-only Compose mount from `config/<site>/catalog_site.json` | Maps Study and dataset names to the site's connector and source. |
| Client `/fdn/policy/default/catalog_policy.json` | Read-only Compose mount from `config/<site>/catalog_policy.json` | Authorizes publication and filters the shared metadata. |

### Network and workspace

`project.yml` provisions two distinct routes to the same server identity:

- Sites connect to `nvflare-server:8002` through Compose DNS.
- The host admin connects to `localhost:8003` through the published port.

The server's certificate covers both `nvflare-server` and `localhost`. The admin
has an explicit `connect_to` override; clients use the Compose hostname. Source
services have no published host ports. Remote deployments must substitute their
own routable host names and provision matching certificates.

`CATALOG_KITS` is the selected host kit directory, with no hard-coded workspace
fallback. Keep it exported for Compose commands. To run from another directory,
pass an absolute Compose path with `docker compose -f /path/to/compose.yaml ...`
and absolute paths to `project.yml` and `job.py`.

Each container mounts its own kit read-only at `/startup-kit`. The generated
startup script uses its native `NVFL_WORKSPACE` support to create a working copy
in `/workspace`, a container-local tmpfs. Runtime PIDs and copied credentials
therefore do not survive container stop/recreation. Server catalog state is
persisted separately at `/workspace/fdn`; job and snapshot volumes use the
standard provisioned resources paths. Parent process logs stream to Compose;
completed job archives remain available through the normal CLI.

### Server: CatalogManager

`CatalogManager` runs in the server parent process. It reconciles site results,
tracks schema versions and site health, registers catalog query commands, and
coordinates reads and private artifacts during archival/download.
`CatalogController` runs inside each submitted job and collects the site-approved
metadata. Without the server manager, job results are not applied to the
persistent catalog and catalog query commands are unavailable.

Refresh is job-driven: rerun `python job.py` to collect source changes. This
example does not schedule periodic discovery. The default health threshold marks
a site's last successful read stale after two days.

### Clients: bindings and publication policy

The container executor uses the existing `/fdn/policy` configuration location.
Compose mounts these files read-only, independently of the ephemeral working
copy of the startup kit. Policy updates are visible to subsequent reads.

| Site configuration | Connector | Source selected by the binding |
| --- | --- | --- |
| [site-1/catalog_site.json](config/site-1/catalog_site.json) | Built-in `unity-catalog` | Catalog `flare_demo`, schema `sanctuary`, registered `pets` table. |
| [site-2/catalog_site.json](config/site-2/catalog_site.json) | `clinical-db`, an instance of `SQLConnector` | Schema `sanctuary` in the configured PostgreSQL database. |

The `studies.default.sanctuary` binding selects the logical dataset `sanctuary`
in the `default` Study. The same name at the two sites identifies separate site
catalogs. SQLAlchemy reflects database metadata; Unity Catalog returns REST
metadata. The job does not read business rows and Spark is not required.

The [site-1 policy](config/site-1/catalog_policy.json) and
[site-2 policy](config/site-2/catalog_policy.json) share `sanctuary` while excluding
column `name` from `pets`. Published columns are `uuid`, `age` and `adopted`.
Missing or invalid policy fails closed. Credentials and database permissions are
configured separately; jobs supply neither bindings nor publication policy.

### Source credentials and assumptions

- `CATALOG_DATABASE_URL` is supplied only to site-2. Its demo value uses
  `catalog_reader:local-demo-reader` at `postgres:5432/sanctuary`. The reader has
  schema visibility without table `SELECT` permission. Its connector sets a
  five-second connection timeout. `POSTGRES_PASSWORD` is the separate bootstrap
  password used by PostgreSQL and demo seeding; override it in the shell or `.env`.
- `UNITY_CATALOG_URL` is supplied only to site-1. Unity Catalog authentication is
  disabled in this local demo. A configured bearer token requires HTTPS except
  literal loopback addresses or localhost during local testing.
- Startup hooks seed 50 synthetic pets at site-1 and 70 at site-2, preserving
  existing fixtures on restart. These are demonstration data, not a deployment
  configuration step.
- The admin runs on the same Docker host. Admin credentials are not mounted in
  FLARE containers. Production authentication, network policy and secret
  distribution remain site responsibilities.
- The normal process job launchers run each site's jobs inside its container.
  No Docker socket is mounted in a FLARE container.

The image builds FLARE from this checkout and includes the pinned
[Unity Catalog Playground sanctuary generator](https://github.com/open-lakehouse/unitycatalog-playground/blob/3310a1f61b345951adc29cdb8f12f9e1d982f88c/marimo-playground/notebooks/sanctuary.py)
and its Apache-2.0 license. No Spark, PyArrow or database driver installation is
needed on the admin computer. The [first example](../federated-catalog) uses
SimEnv with Parquet/CSV and SQLite.
