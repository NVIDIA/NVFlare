# Federated Catalog: containerized Unity Catalog and PostgreSQL sites

This example discovers approved metadata from two sites. Docker Compose runs the
FLARE server, each client site, and each source service in separate containers.
Data preparation is automatic demo initialization, not an admin operation or a
federated task. In a real deployment, the data already exists at each site.

| Service | Role |
| --- | --- |
| `nvflare-poc-server` | FLARE server and aggregate catalog manager |
| `nvflare-poc-site-1` | FLARE site-1; prepares its Unity Catalog demo source once |
| `nvflare-poc-site-2` | FLARE site-2; prepares its PostgreSQL demo source once |
| `unity-catalog` | Unity Catalog OSS metadata; registered Parquet table |
| `postgres` | PostgreSQL source database |

Each site owns the `sanctuary` dataset with a `pets` table. Site-1 has 50 synthetic
pets and site-2 has 70. Their policies exclude `name` from published schemas.
The first [SimEnv example](../federated-catalog) uses Parquet/CSV and SQLite
through the same catalog interfaces.

## Run

Prerequisites: Docker with Compose v2, and an activated Python environment with
this FLARE checkout installed for the admin commands. No database drivers,
PyArrow, or Spark are needed on the admin's computer. Run from this directory:

```bash
# Once, on the admin user's computer: select the workspace and generate all kits.
nvflare poc config --pw ./workspace
nvflare poc prepare -i project.yml

docker compose up -d --build --wait

# On the same computer, using the admin kit selected by poc prepare:
python job.py
nvflare catalog show --format json > catalog-summary.json

docker compose down
```

`nvflare poc config` sets the host's POC workspace. `nvflare poc prepare` generates
the server, site-1, site-2 and **admin** startup kits from `project.yml` and
registers the Project Admin kit as the active CLI identity. The admin kit is at
`workspace/catalog-demo/prod_00/admin@nvidia.com`; it stays on the host and is
not mounted into any container. No separate admin-kit generation or copy is needed,
and subsequent admin commands do not require `--startup-kit`.

The image builds FLARE from this checkout. Compose mounts each FLARE participant's
own prepared kit from `workspace/catalog-demo/prod_00/` as that container's
`/workspace`. Missing kit directories fail startup; run the two POC preparation
commands first. The project explicitly directs clients to the Compose server
hostname while the host admin connects to `localhost`.

`job.py` only submits a `CatalogRecipe` through `ProdEnv` and invokes the standard
`nvflare job monitor` command to show progress until completion; it does
not start services, prepare data, or export results. The normal catalog CLI
retrieves the server's aggregated view. To retrieve full schema details:

```bash
nvflare catalog show --site site-1 --dataset sanctuary --format json > site-1-catalog.json
nvflare catalog show --site site-2 --dataset sanctuary --format json > site-2-catalog.json
```

The normal `nvflare job` CLI remains available for job status, logs and job
archive downloads. The catalog CLI is the interface for the server's reconciled
catalog; a generic job archive is not the catalog store.

`up --wait` waits for source preparation and fresh site registration. The server
is reachable from the host on loopback ports 8002/8003. Source services and sites are
reachable inside the Compose network and have no published host ports.
`down` sends stop signals to the FLARE containers and stops the source services.
Named volumes retain source data; the host POC workspace retains identities and
catalog state. Repeating `up --wait` reuses them without preparing the POC again.
To reset source fixtures, run `docker compose down --volumes`. To also recreate
identities and catalog state, run `nvflare poc prepare -i project.yml --force`
after stopping the stack, then start Compose again. The force option replaces
the configured POC workspace. Compose owns the running services in this example;
use `docker compose up/down` for lifecycle operations.

## Monitor progress and logs

`job.py` enables standard client log streaming, prints the submitted job ID, and
uses `nvflare job monitor` for status and progress. In another terminal, inspect
its server/site logs using the normal CLI:

```bash
nvflare job logs JOB_ID --site all --tail 100
```

For Compose startup, automatic demo preparation, registration and service errors:

```bash
docker compose logs -f nvflare-poc-server nvflare-poc-site-1 nvflare-poc-site-2
```

These commands monitor the existing services and jobs; they do not prepare data
or export results. Result retrieval uses the catalog CLI shown above.

## Data and configuration

The image includes the pinned [Unity Catalog Playground sanctuary generator](https://github.com/open-lakehouse/unitycatalog-playground/blob/3310a1f61b345951adc29cdb8f12f9e1d982f88c/marimo-playground/notebooks/sanctuary.py)
and its Apache-2.0 license, fetched during the build. Each site uses its own seed
and the Playground's `uuid`, `name`, `age`, `adopted` projection. Site-1 writes
Parquet into its data volume and registers `flare_demo.sanctuary.pets` through
the Unity Catalog REST API. Site-2 creates and populates `sanctuary.pets` in
PostgreSQL. Existing tables and Parquet fixtures are preserved on restart.

Each site's checked-in `config/<site>/catalog_site.json` binds its dataset to
its reader. The adjacent `catalog_policy.json` controls publication. Startup
installs these into that site's own `local/fdn/` directory. Each FLARE role
mounts only its own startup kit/runtime volume; the server and site-1 do not
receive the PostgreSQL bootstrap password. The SQL connector's metadata reader
has schema visibility without an application-table `SELECT` grant. Neither
connector reads business rows during a catalog job.

This is a local POC: Unity Catalog's demo endpoint has authorization disabled,
and Compose supplies fixed demonstration PostgreSQL credentials (the bootstrap
password can be overridden with `POSTGRES_PASSWORD`). These are not production
credentials. Production authentication, network policy and secret distribution
remain site deployment responsibilities. No Docker socket is mounted into any
FLARE container. Logs are available with `docker compose logs` and in each role's
runtime volume. Initial image builds need Internet access; site restarts do not
need to download the generator or reinstall dependencies.
