# Catalog connectors

These connectors read a dataset's metadata for the federated data catalog. They do not select application rows,
execute arbitrary SQL, or process data. The CSV connector reads a bounded sample of one file only to infer column
types, and publishes schema metadata, never values.

A connector never reads configuration files or environment variables of its own. The site describes each dataset in
its dataset configuration (`local/datasets.yaml`), and the catalog executor reads it through the dataset
configuration API. Each call to a connector receives:

- `dataset`: the dataset's `DatasetConfig`: its `name`, `type`, `uri` (and `path` for a `file://` uri) and its
  non-secret `options`;
- `secret`: a function that returns one of that dataset's secrets by key, read from the environment variable the
  configuration names;
- `limits`: the remaining time, call and record budgets.

The dataset's catalog rules (`usage.catalog`) decide what is published; the connector only reads.

## The connectors

| Connector | Dataset `type` | `uri` | Options | Secrets |
| --- | --- | --- | --- | --- |
| `ParquetConnector` | `parquet` (built in) | `file://` Parquet file or directory | none | none |
| `CSVConnector` | `csv` (built in) | `file://` CSV file | none | none |
| `SQLConnector` | `sql` (built in) | `file://` SQLite file, or a SQLAlchemy URL | `schema`, `tables`, `connect_args` | `username`, `password` |
| `DuckDBConnector` | `duckdb` (declared by the site) | `file://` DuckDB file | `schema`, `tables` | none |
| `UnityCatalogConnector` | `unity-catalog` (built in) | The REST API root | `catalog`, `schema` (required), `tables` | `token` |

A file connector given a network URL fails the dataset with a safe error. A site declares an additional connector,
such as DuckDB, under `usage.catalog.connectors`, and a dataset's rules may name another connector with `connector`.

## ParquetConnector

`nvflare.app_opt.fdn.catalog.parquet_connector.ParquetConnector` reads Parquet file footers and never rows. Install
`pyarrow>=14`, which `nvflare[catalog]` includes.

```yaml
      omop:
        type: parquet
        uri: file:///data/oncology/omop
        usage:
          catalog: {share: true}
```

The dataset is either one `.parquet` file or a directory:

- A single file is one table, named after the file.
- In a directory, each `.parquet` file directly inside it is one table, named after the file.
- Each subdirectory is also one table, named after the directory, made of every `.parquet` part file below it.
  Symlinked directories are not followed.

Tables are listed in name order. Column names, physical types and nullability come from the footer, written as Arrow
type names. Every part file of a partitioned table must have the same schema as the first, or the dataset fails.
Partition-value types are not inferred.

Parquet footers carry no keys or relationships. A site that wants them published declares them in an optional
`catalog.json` inside the dataset directory (a single-file dataset has no sidecar):

```json
{
  "display_name": "OMOP CDM",
  "semantic_profile": {"standard": "OMOP", "version": "5.4"},
  "tables": {
    "person": {"primary_key": ["person_id"]},
    "condition_occurrence": {
      "relationships": [
        {"columns": ["person_id"], "target_table": "person", "target_columns": ["person_id"]}
      ]
    }
  }
}
```

The sidecar may also carry `description`, `owner`, `modalities`, `operations`, `source_version`, `lineage`,
`observed_at` and the provenance references `declaration_ref` and `validation_ref`. It is limited to 1 MiB, unknown
keys are rejected, and a table entry may only hold `primary_key` and `relationships`. Observed columns always come
from the footers.

## CSVConnector

`nvflare.app_opt.fdn.catalog.csv_connector.CSVConnector` reads one UTF-8 CSV file per dataset. The file needs a
header row with unique, nonempty column names. Install `pyarrow>=14`.

```yaml
      lab_results:
        type: csv
        uri: file:///data/oncology/lab_results/lab_results.csv
        usage:
          catalog: {share: true}
```

PyArrow infers column types from the first 64 KiB of a sample of at most 1 MiB of the file. These are sampled types,
not a validation of the whole file, written as Arrow type names such as `int64`, `double`, `string` or `date32[day]`.
The dataset is one table named after the file, marked tabular. Nullability stays unknown, and no keys are inferred.
The sample is read only to infer types; no values are published.

## SQLConnector

`nvflare.app_opt.fdn.catalog.sql_connector.SQLConnector` uses SQLAlchemy 2.x inspection for table names, columns,
physical types, nullability, primary keys and foreign keys. Install `SQLAlchemy>=2.0,<3` and the database's
SQLAlchemy driver; for example, PostgreSQL uses `psycopg[binary]>=3,<4` with a `postgresql+psycopg://` URL. Dialect
support does not establish that every database and version has been qualified. See
[SQLAlchemy reflection](https://docs.sqlalchemy.org/en/20/core/reflection.html) and
[database URLs](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls).

A SQLite database file:

```yaml
      registry:
        type: sql
        uri: file:///data/oncology/registry/registry.sqlite
        usage:
          catalog: {share: true}
```

The file is opened read-only, and a missing database is never created. Optional descriptive and semantic
declarations stay in `<database>.catalog.json` beside it; observed columns and constraints come from SQLAlchemy.

A network database:

```yaml
      ehr:
        type: sql
        uri: postgresql+psycopg://ehr-db.oncology.example:5432/clinical
        options:
          schema: public
          tables: [person, condition_occurrence]
          connect_args: {connect_timeout: 5}
        secrets: {username: EHR_DB_USER, password: EHR_DB_PASSWORD}
        usage:
          catalog: {share: true}
```

The `uri` holds no credentials; the `username` and `password` secrets complete it when configured. The site delivers
those environment variables to the job through its Study runtime `secret_env`. `connect_args` holds non-secret,
driver-specific settings; the example is PostgreSQL's connect timeout. Use a database identity with only metadata
visibility.

`schema` and `tables` are optional. Without `schema`, the database's default schema is used; without `tables`, the
connector discovers ordinary tables in that schema. Views are not discovered. Relationships name only selected tables
in the same schema. Physical types are rendered by the database dialect, without normalization.

## UnityCatalogConnector

`nvflare.app_opt.fdn.catalog.unity_catalog_connector.UnityCatalogConnector` reads the Unity Catalog REST API using
`requests`, already a FLARE dependency. Spark and storage access are not needed.

```yaml
      warehouse:
        type: unity-catalog
        uri: https://uc.oncology.example/api/2.1/unity-catalog
        options: {catalog: unity, schema: clinical}
        secrets: {token: UC_TOKEN}
        usage:
          catalog: {share: true}
```

The `uri` is the API root, for example `http://localhost:8080/api/2.1/unity-catalog` for a local demo with
authorization disabled, where `secrets` is left out. When a `token` secret is configured it is sent as a bearer token
and must be set; there is no anonymous fallback. A bearer token requires HTTPS, except to a loopback address or
localhost. TLS verification stays enabled, redirects are rejected, and no storage credentials are requested.

An optional `tables` list restricts the read; otherwise every page of the schema's table inventory is read. Table and
column names, `type_text` and reported nullability are published. Unity Catalog's `TableInfo` supplies no
primary-key or foreign-key constraints, so those stay empty. Storage locations, properties, view SQL and ownership
identities are never published. `request_timeout_s`, 10 seconds per request by default, is a connector argument a
site can set where it declares the connector under `usage.catalog.connectors`.

See the [Unity Catalog API definition](https://github.com/unitycatalog/unitycatalog/blob/main/api/all.yaml) and
[local server setup](https://docs.unitycatalog.io/how-to/run-local-server).

## DuckDBConnector

`nvflare.app_opt.fdn.catalog.duckdb_connector.DuckDBConnector` reads a DuckDB database file. Install `duckdb>=1.1`,
which `nvflare[catalog]` includes. SQLAlchemy is not used: reflection through `duckdb-engine` fails on DuckDB's
catalog, so the connector runs four fixed queries against DuckDB's own `information_schema` and
`duckdb_constraints()`. Column names, physical types, nullability, primary keys and foreign keys are published; no
application rows are read.

The file is opened read-only with external file access disabled. Optional declarations stay in
`<database>.catalog.json`, as for SQLite. The connector is not built in, so the site declares it:

```yaml
studies:
  oncology:
    datasets:
      claims:
        type: duckdb
        uri: file:///data/oncology/claims/claims.duckdb
        options: {schema: main}
        usage:
          catalog: {share: true}
usage:
  catalog:
    connectors:
      duckdb: {class: nvflare.app_opt.fdn.catalog.duckdb_connector.DuckDBConnector}
```

`schema` (default `main`) and `tables` are the only options. Without `tables`, every ordinary table in the schema is
published; views are not. A relationship is published only when its target is one of the selected tables. DuckDB
does not report the target's schema, so keep related tables in one schema. A file that another process has open for
writing cannot be read and is reported as unavailable.

## Limits and failures

All five connectors return a complete dataset catalog or raise a safe `ConnectorError`. They enforce record budgets
and check the duration budget before and after provider operations; the executor gives them the time left before the
task's deadline. Parquet counts one call for each footer it reads and each part file it lists; CSV counts its one
bounded sample read; SQL counts reflection statements emitted through the SQLAlchemy connection; DuckDB counts its
four metadata queries; Unity Catalog counts HTTP requests, including pages and detail lookups. A repeated page token
or inconsistent inventory fails the read. Connections and HTTP sessions are closed on success and failure; raw
provider errors, SQL text, endpoints and credentials are not returned.

Duration checks are cooperative, not a universal cancellation mechanism. Configure database connection and statement
timeouts for the chosen driver; HTTP timeouts limit connect and read inactivity. A connector that blocks past the time
it is given can still lose its site's result. A catalog read does not establish a transactionally consistent snapshot
across concurrent schema changes or prove that declared constraints are enforced.
