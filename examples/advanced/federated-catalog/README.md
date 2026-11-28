# Federated Catalog: Parquet, CSV, SQLite and DuckDB sites

This CPU-only example uses `get_catalog` and `SimEnv` to read the metadata of
three sites on one host. It needs no Docker and starts no services. Every
site has a `clinical_records` dataset containing exactly two tables: `person` and
`condition_occurrence`. Each site stores them differently.

| Site | Storage | Dataset type | Tables | Synthetic persons |
| --- | --- | --- | --- | ---: |
| site-1 | Parquet | `parquet` | `person`, `condition_occurrence` | 8 |
| site-2 | SQLite | `sql` | `person`, `condition_occurrence` | 12 |
| site-3 | DuckDB | `duckdb` | `person`, `condition_occurrence` | 10 |

There are two condition rows per person. The data is a small synthetic subset
shaped after [OMOP CDM 5.4](https://ohdsi.github.io/CommonDataModel/cdm54.html).
Concept IDs are 0 (unmapped); this is not a complete or clinically validated CDM.

Site-1 additionally shares two header-bearing CSV datasets, checked into
`data/site-1/`: `lab_results` and `visits`, each containing one table.
Site-1 therefore publishes three dataset catalogs; site-2 and site-3 publish one
each.

On each catalog read the site generates one ODCS document per CSV dataset from its header. For both `lab_results` and `visits`, it publishes the column names in their original order.

For a deployment with Unity Catalog and PostgreSQL servers running in Docker, see
the [catalog services example](../federated-catalog-services). It runs a real
server and provisioned startup kits.

## Install, prepare and run

From this directory in an activated Python environment with FLARE installed:

```bash
python -m pip install "nvflare[catalog]>=2.10.0" -r requirements.txt
python prepare_data.py
python job.py
```

`prepare_data.py` creates one folder per site under `/tmp/nvflare/federated-catalog/data`
and fills it like a deployed site would be:

```text
site-1/
  data/clinical_records/                  Parquet files
  data/lab_results.csv        CSV
  data/visits.csv             CSV
  local/datasets.yaml         the site's dataset configuration
site-2/
  data/clinical_warehouse.sqlite            SQLite database
  local/datasets.yaml
site-3/
  data/clinical_warehouse.duckdb            DuckDB database
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
vis.show_tables(dataset="clinical_records")
vis.show_schema("person", site="site-1", dataset="clinical_records")
vis.show_column_comparison("condition_occurrence", dataset="clinical_records")
vis.show_table_comparison()  # compare every dataset, including CSV datasets
```

Red marks metadata not published by a source, gray marks a dataset that was not read,
and yellow marks different reported column definitions. The `get_*` counterparts
return DataFrames for further analysis. The explorer also uses `show_contract` to open
site-1's published `lab_results` contract in Jupyter. Each comparison table includes a title and explanation, so
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
      clinical_records:            # Parquet directory: one table per .parquet file
        type: parquet
        uri: file://${SITE_DATA}/clinical_records
      lab_results:                # one CSV file
        type: csv
        uri: file://${SITE_DATA}/lab_results.csv
      visits:
        type: csv
        uri: file://${SITE_DATA}/visits.csv
```

[config/site-2/datasets.yaml](config/site-2/datasets.yaml) has one `clinical_records`
dataset of type `sql` whose uri names its SQLite file.
[config/site-3/datasets.yaml](config/site-3/datasets.yaml) has one of type
`duckdb`. The dataset type selects the reader directly.

Every dataset configured for the Study is read. The source identity and readable files determine metadata visibility; the catalog does not exclude columns.

## What schemas are published?

All three sites publish these `person` columns:

| Column | site-1: Parquet | site-2: SQLite | site-3: DuckDB |
| --- | --- | --- | --- |
| `person_id` | `INT64` | `INTEGER` | `INT` |
| `gender_concept_id` | `INT64` | `INTEGER` | `INT` |
| `year_of_birth` | `INT64` | `INTEGER` | `INT` |
| `person_source_value` | `STRING` | `TEXT` | `TEXT` |

The `condition_occurrence` schemas overlap but each site has one column of its own:

| Column | site-1: Parquet | site-2: SQLite | site-3: DuckDB |
| --- | --- | --- | --- |
| `condition_occurrence_id` | `INT64` | `INTEGER` | `INT` |
| `person_id` | `INT64` | `INTEGER` | `INT` |
| `condition_concept_id` | `INT64` | `INTEGER` | `INT` |
| `condition_start_date` | `DATE` | `TEXT` | `DATE` |
| `condition_type_concept_id` | `INT64` | `INTEGER` | `INT` |
| `condition_source_value` | `STRING` | `TEXT` | `TEXT` |
| `condition_end_date` | `DATE` | absent | absent |
| `condition_status_concept_id` | absent | `INTEGER` | absent |
| `stop_reason` | absent | absent | `TEXT` |

The SQLite and DuckDB tables declare `person_id` and `condition_occurrence_id`
as primary keys and `condition_occurrence.person_id` as a foreign key to
`person`; their readers report these constraints. Parquet footers carry no
keys, so site-1 publishes none. Data Contract CLI imports Parquet footers and the DDL extracted from read-only SQLite and DuckDB files. The displayed type spellings are those produced by the upstream importers.

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
jq '.datasets[] | {site, dataset, tables: .contract.schema}' \
  /tmp/nvflare/federated-catalog/catalog_results.json
```

Each read returns the complete answer; the latest read is the current catalog.
The simulation workspace contains the standalone ODCS files in the server run’s `catalog/contracts/` directory. The simulator logs are under `/tmp/nvflare/federated-catalog/simulation/get_catalog/`.

## Control source metadata visibility

The catalog has no publication policy. Each configured Study dataset is read using the site's source identity and
file access. Use a database namespace or selected table projection whose metadata may be published; verify the
provider's metadata permissions independently of row access. For these local file examples, expose only suitable
files/directories to the job. Remove a dataset's binding from the Study to stop cataloging that source on later jobs.

Run `python job.py` again after changing the source or configuration. Each result is a new snapshot; previously
saved results are not retracted. Running preparation again restores the checked-in configuration.

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
`usage: {catalog: {limits: {max_records: 20}}}` to site-1's `datasets.yaml` to
see datasets reported as `skipped`. In a deployment, omit `site_config_root`:
each site keeps its own `local/datasets.yaml`, and `get_catalog` runs against a
`ProdEnv`; see the [catalog services example](../federated-catalog-services).
