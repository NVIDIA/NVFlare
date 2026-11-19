# Federated Catalog: Parquet, CSV, SQLite and DuckDB sites

This CPU-only example uses `get_catalog` and `SimEnv` to read the metadata of
three sites on one host. It needs no Docker and starts no services. Every
site has an `omop` dataset containing exactly two tables: `person` and
`condition_occurrence`. Each site stores them differently.

| Site | Storage | Connector | Tables | Synthetic persons |
| --- | --- | --- | --- | ---: |
| site-1 | Parquet | `parquet` (built in) | `person`, `condition_occurrence` | 8 |
| site-2 | SQLite | `sql` (built in, `SQLConnector`) | `person`, `condition_occurrence` | 12 |
| site-3 | DuckDB | `duckdb`, declared by the site | `person`, `condition_occurrence` | 10 |

There are two condition rows per person. The data is a small synthetic subset
shaped after [OMOP CDM 5.4](https://ohdsi.github.io/CommonDataModel/cdm54.html).
Concept IDs are 0 (unmapped); this is not a complete or clinically validated CDM.

Site-1 additionally shares two header-bearing CSV datasets, checked into
`data/site-1/`: `lab_results` and `visits`, each containing one table.
Site-1 therefore publishes three dataset catalogs; site-2 and site-3 publish one
each.

| CSV dataset | Columns and inferred types |
| --- | --- |
| `lab_results` | `person_id`: int64, `test_name`: string, `result_value`: double, `collected_date`: date32[day], `abnormal`: bool |
| `visits` | `visit_id`: int64, `person_id`: int64, `visit_date`: date32[day], `visit_type`: string, `duration_minutes`: int64 |

The CSV connector delegates parsing and type inference to PyArrow. It infers
from the first 64 KiB block, with at most 1 MiB of local source input for
record boundaries and read-ahead, subject to site call/time/schema budgets.
Only schema metadata is published; Arrow physical type names are retained.
All-empty columns have Arrow type `null`. Nullability remains unknown, and no
keys/relationships are inferred. Inferred types describe the initial block, not
a validated full-file schema; later rows can differ.

For a deployment with Unity Catalog and PostgreSQL servers running in Docker, see
the [catalog services example](../federated-catalog-services). It runs a real
server and provisioned startup kits.

## Install, prepare and run

From this directory in an activated Python environment with FLARE installed:

```bash
python -m pip install -r requirements.txt
python prepare_data.py
python job.py
```

`prepare_data.py` creates one folder per site under `/tmp/nvflare/federated-catalog/data`
and fills it like a deployed site would be:

```text
site-1/
  data/omop/                  Parquet files
  data/lab_results.csv        CSV
  data/visits.csv             CSV
  local/datasets.yaml         the site's dataset configuration
site-2/
  data/omop.sqlite            SQLite database
  local/datasets.yaml
site-3/
  data/omop.duckdb            DuckDB database
  local/datasets.yaml
```

Preparation regenerates this example's files, removing files left by earlier
runs. It marks each site folder it creates and refuses to replace a site folder
it did not create, such as a real site's workspace; add `--force` to replace
it anyway. `job.py` calls `get_catalog`, prints a summary and saves the full
result to `/tmp/nvflare/federated-catalog/catalog_results.json`.

## Visualize the results

For the complete prepare → read → visualize workflow, open
[get_catalog.ipynb](get_catalog.ipynb):

```bash
jupyter lab get_catalog.ipynb
```

It prepares the sites, runs `job.py`, and executes `catalog_explorer.ipynb` on
the resulting `catalog_results.json` in the same kernel.

Open [catalog_explorer.ipynb](catalog_explorer.ipynb) after running the job:

```bash
jupyter lab catalog_explorer.ipynb
```

Following the federated statistics visualization pattern, it loads
`/tmp/nvflare/federated-catalog/catalog_results.json` and calls the reusable
`CatalogVisualization` functions. Set the notebook's `CATALOG_FILE` if you used
a different output path. No federation connection is needed to visualize
the saved result.

```python
from nvflare.app_opt.fdn.catalog.visualization import CatalogVisualization

vis = CatalogVisualization("/tmp/nvflare/federated-catalog/catalog_results.json")
vis.show_sites()
vis.show_datasets()
vis.show_tables(dataset="omop")
vis.show_schema("person", site="site-1", dataset="omop")
vis.show_column_comparison("condition_occurrence", dataset="omop")
vis.show_table_comparison()  # compare every dataset, including CSV datasets
```

Red marks metadata not published by a source, gray marks a dataset that was not read,
and yellow marks different reported column definitions. The `get_*` counterparts
return DataFrames for further analysis; the `show_*` functions return styled
tables that Jupyter renders. Each table includes a title and explanation, so
these remain visible when the runner invokes the explorer notebook.

## Site configurations

Each site describes its datasets in one checked-in file, its dataset
configuration. `prepare_data.py` installs it as `<site>/local/datasets.yaml`,
replacing `${SITE_DATA}` with the site's absolute data folder, because a
`file://` uri must be absolute.

[config/site-1/datasets.yaml](config/site-1/datasets.yaml):

```yaml
format_version: 1
studies:
  default:                        # SimEnv runs in the default Study
    datasets:
      omop:                       # Parquet directory: one table per .parquet file
        type: parquet
        uri: file://${SITE_DATA}/omop
        usage:
          catalog:
            share: true
            exclude_columns:
              person: [person_source_value]
              condition_occurrence: [condition_source_value]
      lab_results:                # one CSV file
        type: csv
        uri: file://${SITE_DATA}/lab_results.csv
        usage:
          catalog: {share: true}
      visits:
        type: csv
        uri: file://${SITE_DATA}/visits.csv
        usage:
          catalog: {share: true}
```

[config/site-2/datasets.yaml](config/site-2/datasets.yaml) has one `omop`
dataset of type `sql` whose uri names its SQLite file.
[config/site-3/datasets.yaml](config/site-3/datasets.yaml) has one of type
`duckdb`, and declares the DuckDB connector by its class, because it is not
built in:

```yaml
usage:
  catalog:
    connectors:
      duckdb: {class: nvflare.app_opt.fdn.catalog.duckdb_connector.DuckDBConnector}
```

Each dataset's `usage.catalog` rules share it and exclude `person_source_value`
and `condition_source_value` from published schemas. Source rows stay local.

## What schemas are published?

All three sites publish these `person` columns:

| Column | site-1: Parquet | site-2: SQLite | site-3: DuckDB |
| --- | --- | --- | --- |
| `person_id` | `int64` | `INTEGER` | `INTEGER` |
| `gender_concept_id` | `int64` | `INTEGER` | `INTEGER` |
| `year_of_birth` | `int64` | `INTEGER` | `INTEGER` |

The `condition_occurrence` schemas overlap but each site has one column of its own:

| Column | site-1: Parquet | site-2: SQLite | site-3: DuckDB |
| --- | --- | --- | --- |
| `condition_occurrence_id` | `int64` | `INTEGER` | `INTEGER` |
| `person_id` | `int64` | `INTEGER` | `INTEGER` |
| `condition_concept_id` | `int64` | `INTEGER` | `INTEGER` |
| `condition_start_date` | `date32[day]` | `TEXT` | `DATE` |
| `condition_type_concept_id` | `int64` | `INTEGER` | `INTEGER` |
| `condition_end_date` | `date32[day]` | absent | absent |
| `condition_status_concept_id` | absent | `INTEGER` | absent |
| `stop_reason` | absent | absent | `VARCHAR` |

The SQLite and DuckDB tables declare `person_id` and `condition_occurrence_id`
as primary keys and `condition_occurrence.person_id` as a foreign key to
`person`; their readers report these constraints. Parquet footers carry no
keys, so site-1 publishes none. The Parquet reader observes schemas from
footers; the SQL and DuckDB readers open their databases read-only and inspect
schema metadata and constraints. None of them
selects source rows. Published physical types retain their source
representation; the catalog does not infer clinical equivalence or perform
conversions.

## Inspect results

For workflow progress on the console, run `python job.py --log_config progress`.
`--log_config` is a shared Recipe option, handled by NVFlare before the
example's data and output path arguments are parsed.

A fresh run reports each site as `OK` and returns five datasets: three at
site-1, one at site-2 and one at site-3. The OMOP datasets each contain two
tables; each CSV dataset contains one. The summary is printed, and the full
result is saved to `/tmp/nvflare/federated-catalog/catalog_results.json`:

```bash
python -m json.tool /tmp/nvflare/federated-catalog/catalog_results.json
```

With `jq`, display the table schemas at every site:

```bash
jq '.datasets[] | {site, dataset, tables: .metadata.tables}' \
  /tmp/nvflare/federated-catalog/catalog_results.json
```

Each read returns the complete answer; the latest read is the current catalog.
After a completed read, only `catalog_results.json` is kept in the simulation workspace. When a read fails, the
whole workspace is kept, with the simulator logs under `/tmp/nvflare/federated-catalog/simulation/get_catalog/`.

## Change what a site shares

Edit the installed site-2 configuration at
`/tmp/nvflare/federated-catalog/data/site-2/local/datasets.yaml` and set the
dataset's rule to `share: false`:

```yaml
        usage:
          catalog: {share: false}
```

Run `python job.py` again. The result now contains only the datasets of site-1
and site-3, and does not name site-2's `omop`. Running preparation again
restores the checked-in configuration.

## Custom paths and limits

```bash
python prepare_data.py --data-root /tmp/my-catalog/data
python job.py --data-root /tmp/my-catalog/data \
  --workspace /tmp/my-catalog/simulation --output /tmp/my-catalog/catalog_results.json
```

Use a dedicated simulation workspace; `SimEnv` replaces its job contents. Data
and results live outside that workspace. The environment variable
`NVFLARE_SIMULATOR_WORKSPACE_ROOT`, when set, overrides `--workspace`.

A reused simulation workspace keeps an explicit `class_allow_list` in its
`local/resources.json`, because it may be an operator's security policy. If that
list predates catalog support, use a fresh `--workspace`, or add these entries to
it, keeping the others:

```text
nvflare.app_common.fdn.catalog.server.controller.CatalogController
nvflare.app_common.fdn.catalog.site.executor.CatalogExecutor
nvflare.app_common.fdn.catalog.simulation.SimulationCatalogExecutor
```

A site's read budget is part of its dataset configuration: add
`usage: {catalog: {limits: {max_records: 50}}}` to a site's `datasets.yaml` to
see datasets reported as `skipped`. In a deployment, omit `site_config_root`:
each site keeps its own `local/datasets.yaml`, and `get_catalog` runs against a
`ProdEnv`; see the [catalog services example](../federated-catalog-services).
