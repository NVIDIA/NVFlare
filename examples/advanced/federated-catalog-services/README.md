# Federated Catalog: Unity Catalog and PostgreSQL sites

Discover site-approved metadata from two containerized sites: site-1 uses Unity
Catalog OSS with a Parquet table; site-2 uses PostgreSQL. Source rows stay at
those sites. Each publishes the schema of its `sanctuary.pets` table, excluding
`name` through its local catalog policy.

## Quick start

Prerequisites: Docker Compose **2.30 or later**, and a Python environment with
this FLARE checkout installed. Run from this example directory:

```bash
python setup.py --workspace /tmp/nvflare/catalog
docker compose up -d --build --wait
python job.py
nvflare catalog show
```

`--workspace` accepts any absolute or relative location. Omitting it uses this
example's `workspace` directory. Setup provisions kits when needed, registers
and selects the admin identity, adds `CatalogManager` to the server resources,
and writes Compose's `.env`. Existing kits, custom server components and other
`.env` settings are preserved when setup is rerun. Use `--reprovision` after
changing `project.yml` to generate new kits.

Compose starts FLARE through the generated startup scripts and prepares demo
data at each site. `job.py` submits and monitors the catalog read from your
computer using the selected admin kit. Source rows stay at the sites.

Stop after the job completes:

```bash
docker compose down
```

Compose's stop hook calls each generated `stop_fl.sh` before Docker stops the
container. Kits, source data, accepted catalogs and archived jobs remain for
the next `docker compose up -d --wait`. `down --volumes` removes demo data/state.

## Progress and results

These standard commands are available when needed:

```bash
docker compose logs -f nvflare-server nvflare-site-1 nvflare-site-2
nvflare system status
nvflare catalog show --site site-1 --dataset sanctuary --format json > site-1-catalog.json
nvflare catalog show --site site-2 --dataset sanctuary --format json > site-2-catalog.json
nvflare job logs JOB_ID --site all --tail 100
```

`job.py` streams client logs and uses `nvflare job monitor` through completion.
Use the normal job CLI for archive downloads. `docker compose up --wait` waits
for container/source startup; `nvflare system status` checks FLARE registration.

## What setup configures

[setup.py](setup.py) calls the standard `nvflare provision` and `nvflare config
add/use` commands. It selects the latest generated kit directory, creates the
server's `local/resources.json` from its default when needed, and appends the
`catalog_manager` component once. It records `CATALOG_KITS` in `.env` so users
need no exports or manual kit-generation selection. The admin kit remains on
your computer; it is never mounted in a service container.

When switching from an older manual setup, unset a previously exported
`CATALOG_KITS` if it points elsewhere: shell variables override Compose's `.env`.
Changing the workspace does not reset Docker volumes; use `down --volumes` when
you also want fresh demo data/state.

## Deployment configuration

| Configuration | Supplied by | Purpose |
| --- | --- | --- |
| [project.yml](project.yml) | `nvflare provision` | Defines participants, endpoints, certificate host names and startup kits. |
| Server `local/resources.json` | Setup updates the generated kit | Registers the long-lived `CatalogManager` alongside the existing server components. |
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

Setup stores the selected host kit directory as `CATALOG_KITS` in `.env`.
To run from another directory, pass an absolute Compose path with
`docker compose -f /path/to/compose.yaml ...` and an absolute path to `job.py`.
Setup locates `project.yml` relative to its own file.

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
