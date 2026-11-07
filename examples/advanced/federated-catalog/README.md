# Federated Catalog: two tables at two sites

This CPU-only example uses `CatalogRecipe` and `SimEnv` to discover metadata
from two sites. Both sites have an `omop` dataset containing exactly two tables:
`person` and `condition_occurrence`.

| Site | Storage | Tables | Synthetic persons |
| --- | --- | --- | ---: |
| site-1 | Parquet | `person`, `condition_occurrence` | 8 |
| site-2 | SQLite | `person`, `condition_occurrence` | 12 |

There are two condition rows per person. The data is a small synthetic subset
shaped after [OMOP CDM 5.4](https://ohdsi.github.io/CommonDataModel/cdm54.html).
Concept IDs are 0 (unmapped); this is not a complete or clinically validated CDM.

Site-1 additionally shares two header-bearing CSV datasets, checked into
`data/site-1/`: `lab_results` and `visits`, each containing one table.
Site-1 therefore publishes three dataset catalogs; site-2 publishes one.

| CSV dataset | Columns and inferred types |
| --- | --- |
| `lab_results` | `person_id`: int64, `test_name`: string, `result_value`: double, `collected_date`: date32[day], `abnormal`: bool |
| `visits` | `visit_id`: int64, `person_id`: int64, `visit_date`: date32[day], `visit_type`: string, `duration_minutes`: int64 |

The CSV connector delegates parsing and type inference to PyArrow. It infers
from the first 64 KiB block, with at most 1 MiB of local source input for
record boundaries and read-ahead, subject to site call/time/schema budgets.
Only schema metadata is published; Arrow physical type names are retained.
All-empty columns have Arrow type `null`. Nullability remains unknown, and no
keys/relationships are inferred. Arrow may infer numeric identifiers with
leading zeros as integers. Inferred types describe the initial block, not a
validated full-file schema; later rows can differ. Preparation copies the
checked-in CSV files.

For a provisioned two-site POC using Docker-hosted Unity Catalog and PostgreSQL,
see the [catalog services example](../federated-catalog-services). Both examples
use `SQLConnector`; this example retains SQLite as its local SQL backend.

## Install, prepare and run

From this directory in an activated Python environment:

```bash
python -m pip install -r requirements.txt
python prepare_data.py
python job.py
```

`prepare_data.py` creates/populates the source tables, installs the checked-in
site configurations and writes runtime bindings for their generated paths.
Preparation regenerates this example's source files, including removing tables
left by earlier versions. `job.py` only configures/runs the catalog recipe and
prints the resulting metadata.

## Site configurations

Both sites explicitly declare their reader in separate checked-in files.

[config/site-1/catalog_site.json](config/site-1/catalog_site.json):

```json
{
  "format_version": 1,
  "default_connector": "parquet",
  "studies": {
    "default": {
      "lab_results": {
        "connector": "csv"
      },
      "visits": {
        "connector": "csv"
      }
    }
  }
}
```

[config/site-2/catalog_site.json](config/site-2/catalog_site.json):

```json
{"format_version": 1, "default_connector": "sql"}
```

Each site also has its own checked-in `catalog_policy.json`:
[site-1](config/site-1/catalog_policy.json), [site-2](config/site-2/catalog_policy.json).
The policies share `omop` and exclude `person_source_value` and
`condition_source_value` from published schemas. Source rows stay local.

Preparation copies the JSON configurations unchanged. Source paths go in the
site's `study_runtime.yaml`. Optional OMOP and Parquet key/relationship annotations
are included with the example's preparation assets under `data/` and copied
beside the generated sources.

The [design](../../../docs/design/federated_data_cataloging_design.md#catalog_sitejson-contract-v1)
defines the reader configuration and
[policy contract](../../../docs/design/federated_data_cataloging_design.md#catalog_policyjson-contract-v1).

## What schemas are published?

Both sites publish these `person` columns:

| Column | site-1: Parquet | site-2: SQLite |
| --- | --- | --- |
| `person_id` (primary key) | `int64` | `INTEGER` |
| `gender_concept_id` | `int64` | `INTEGER` |
| `year_of_birth` | `int64` | `INTEGER` |

The `condition_occurrence` schemas overlap but have one different column:

| Column | site-1: Parquet | site-2: SQLite |
| --- | --- | --- |
| `condition_occurrence_id` (primary key) | `int64` | `INTEGER` |
| `person_id` (references `person.person_id`) | `int64` | `INTEGER` |
| `condition_concept_id` | `int64` | `INTEGER` |
| `condition_start_date` | `date32[day]` | `TEXT` |
| `condition_type_concept_id` | `int64` | `INTEGER` |
| `condition_end_date` | `date32[day]` | absent |
| `condition_status_concept_id` | absent | `INTEGER` |

The Parquet reader observes schemas from footers; optional annotations declare
keys and relationships. The SQLAlchemy-based SQL reader opens this SQLite database read-only and
inspects schema metadata and declared constraints. It does not select source
rows. Published physical types retain their source representation; the catalog
does not infer clinical equivalence or perform conversions.

## Inspect results

For workflow progress on the console, run `python job.py --log_config progressive`
(`progress` is also supported). Invalid logging selectors are rejected before
the simulator starts, with a usage error listing supported modes.

A fresh run returns four catalog entries: three at site-1 and one at site-2.
The OMOP datasets each contain two tables; each CSV dataset contains one. The JSON is
printed and saved to `/tmp/nvflare/federated-catalog/store/discovery.json`.
Use a JSON viewer or standard tools:

```bash
python -m json.tool /tmp/nvflare/federated-catalog/store/discovery.json
```

With `jq`, display the table schemas at both sites:

```bash
jq '.entries[] | {site: .client, dataset, tables: .metadata.tables}' \
  /tmp/nvflare/federated-catalog/store/discovery.json
```

Catalog/schema versions start at 1 in a fresh store. Unchanged reads retain
their versions. Simulator logs are under
`/tmp/nvflare/federated-catalog/simulation/catalog_read/`.

## Change a publication policy

Edit the installed site-2 policy at
`/tmp/nvflare/federated-catalog/data/site-2/local/fdn/default/catalog_policy.json`.
To withhold its whole dataset, the complete policy can simply be:

```json
{
  "format_version": 1,
  "datasets": {
    "omop": {"share": false}
  }
}
```

Run `python job.py` again. Discovery now contains only site-1's dataset.
Previous published names come from the persistent catalog, so withdrawing
site-2 works across simulation runs. Running preparation again restores the
checked-in policies.

## Custom paths and deployment

```bash
python prepare_data.py --data-root /tmp/my-catalog/data
python job.py --data-root /tmp/my-catalog/data \
  --workspace /tmp/my-catalog/simulation --store-root /tmp/my-catalog/store
```

Use a dedicated simulation workspace; `SimEnv` replaces its job contents.
Data and catalog state live outside that workspace. The environment variable
`NVFLARE_SIMULATOR_WORKSPACE_ROOT`, when set, overrides `--workspace`.

When upgrading a reused simulation workspace, an explicit `class_allow_list`
without a SimEnv generation marker is preserved because it may be an operator's
security policy. If it predates catalog support, use a fresh `--workspace` or
add the following entries to its existing `local/resources.json` or
`local/resources.json.default` allowlist, retaining the other approved entries:

```text
nvflare.app_common.fdn.catalog.server.controller.CatalogController
nvflare.app_common.fdn.catalog.site.executor.CatalogExecutor
nvflare.app_common.fdn.catalog.simulation.SimulationCatalogController
nvflare.app_common.fdn.catalog.simulation.SimulationCatalogExecutor
```

Unchanged policies generated by current SimEnv versions refresh automatically.

For a different source combination, edit the relevant site's existing
`default_connector` and rerun preparation; no alternative configuration file is
needed. In production, omit `site_config_root` and configure `CatalogManager`
and the site-owned runtime bindings and policies. Simulation keeps unsigned
manifests locally and does not exercise production admin login, scheduling or
archive/download protections.
