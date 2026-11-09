# Catalog connectors

These connectors publish `CatalogInfo` metadata through the existing site
publication policy. They do not select application rows, execute arbitrary SQL,
or process data. The CSV connector reads a bounded sample of one file only to
infer column types, and publishes schema metadata, never values. Connector
selection, source bindings and credentials are site-owned; a federated catalog
task cannot provide them.

## The connectors

| Connector | Name in `catalog_site.json` | What it reads | Binding options |
| --- | --- | --- | --- |
| `ParquetConnector` | `parquet` (built in, the default) | Parquet file footers of a mounted dataset | none |
| `CSVConnector` | `csv` (built in) | The header and a bounded sample of one CSV file | none |
| `SQLConnector` | `sql` (built in) | A SQL database through SQLAlchemy reflection | `schema`, `tables` |
| `DuckDBConnector` | declared by class path under `connectors` | A DuckDB file through its own catalog views | `schema`, `tables` |
| `UnityCatalogConnector` | `unity-catalog` (built in) | Unity Catalog or Databricks through the REST API | `catalog`, `schema`, `tables` |

A site names the connector for each dataset in `catalog_site.json`, or uses a
default connector for mounted datasets that have no binding. Different datasets,
and different sites, can use different connectors.

## ParquetConnector

`nvflare.app_opt.fdn.catalog.parquet_connector.ParquetConnector` is the default
connector, named `parquet`. It reads Parquet file footers and never rows. Install
`pyarrow>=14`, which `nvflare[catalog]` includes. A site whose datasets are all
mounted Parquet needs only `{"format_version": 1}` in `catalog_site.json`, with no
connector entries.

A mounted dataset is either one `.parquet` file or a directory:

- A single file is one table, named after the file.
- In a directory, each `.parquet` file directly inside it is one table, named
  after the file.
- Each subdirectory is also one table, named after the directory. Its table is
  made of every `.parquet` part file below it. Symlinked directories are not
  followed.

Tables are listed in name order. Column names, physical types and nullability
come from the footer, with types written as Arrow type names. For a partitioned
table every part file must have the same schema as the first. A difference fails
the dataset. Partition-value types are not inferred.

Parquet footers carry no keys or relationships. A site that wants them published
declares them in an optional `catalog.json` inside the dataset directory (a
single-file mount has no sidecar):

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

The sidecar may also carry `description`, `owner`, `modalities`, `operations`,
`source_version`, `lineage`, `observed_at` and the provenance references
`declaration_ref` and `validation_ref`. It is limited to 1 MiB, unknown keys are
rejected, and a table entry may only hold `primary_key` and `relationships`.
Observed columns always come from the footers. The built-in reader accepts no
binding options.

## CSVConnector

`nvflare.app_opt.fdn.catalog.csv_connector.CSVConnector`, named `csv`, reads one
UTF-8 CSV file per dataset. The file needs a header row with unique, nonempty
column names. Install `pyarrow>=14`. Bind a dataset to it in `catalog_site.json`:

```json
{
  "format_version": 1,
  "studies": {"default": {"lab_results": {"connector": "csv"}}}
}
```

PyArrow infers column types from the first 64 KiB of a sample of at most 1 MiB
of the file. These are sampled types, not a validation of the whole file, and are
written as Arrow type names such as `int64`, `double`, `string` or `date32[day]`.
The dataset is one table named after the file, marked tabular. Nullability stays
unknown, and no keys are inferred. The connector reads the sample only to infer
types and publishes the schema, never any values. It has no sidecar and accepts
no binding options.

## SQLConnector

`nvflare.app_opt.fdn.catalog.sql_connector.SQLConnector` uses SQLAlchemy 2.x
inspection for table names, columns, physical types, nullability, primary keys
and foreign keys. Install `SQLAlchemy>=2.0,<3` and the database's SQLAlchemy driver.
For example, PostgreSQL uses `psycopg[binary]>=3,<4` with a
`postgresql+psycopg://` URL. MySQL (`mysql+pymysql://`) and Oracle
(`oracle+oracledb://`) use their respective installed drivers. Dialect support
does not establish that every database/version has been qualified.

See [SQLAlchemy reflection](https://docs.sqlalchemy.org/en/20/core/reflection.html)
and [database URLs](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls).

For a mounted SQLite database, the site only needs:

```json
{"format_version": 1, "default_connector": "sql"}
```

The existing Study runtime dataset binding supplies the file path. The connector
opens it read-only and never creates a missing database. Optional descriptive
and semantic declarations remain in `<database>.catalog.json`; observed columns
and constraints come from SQLAlchemy. The old SQLite-specific connector is
replaced by `SQLConnector`, including in the first federated-catalog example.

For a network database, supply its URL through a site environment variable:

```json
{
  "format_version": 1,
  "connectors": {
    "clinical-db": {
      "class": "nvflare.app_opt.fdn.catalog.sql_connector.SQLConnector",
      "args": {
        "url_env": "CLINICAL_DATABASE_URL",
        "connect_args": {"connect_timeout": 5}
      }
    }
  },
  "studies": {
    "default": {
      "omop": {
        "connector": "clinical-db",
        "options": {"schema": "public", "tables": ["person", "condition_occurrence"]}
      }
    }
  }
}
```

The site supplies `CLINICAL_DATABASE_URL` to the job environment using its
existing runtime/secret mechanism. The URL, passwords and tokens never belong
in this JSON or in job arguments. `connect_args` is for non-secret,
driver-specific settings. The example above is PostgreSQL-specific; other
drivers have different timeout settings. Use a source identity with only the
required metadata visibility.

`schema` and `tables` are optional. Without `schema`, the database's default
schema is used; without `tables`, the connector discovers ordinary tables in
that schema. Views are not included in automatic discovery. Relationships name
only selected tables in the same schema. Physical types are rendered by the
database dialect, with no clinical equivalence or type normalization inferred.

## UnityCatalogConnector

`nvflare.app_opt.fdn.catalog.unity_catalog_connector.UnityCatalogConnector`
reads the Unity Catalog REST API using `requests`, already a FLARE dependency.
It needs a running Unity Catalog server; Spark and storage access are unnecessary
for this metadata read.

The built-in `unity-catalog` reader uses `UNITY_CATALOG_URL`, containing the API
root, for example `http://localhost:8080/api/2.1/unity-catalog` for a local demo.
Each dataset specifies its catalog and schema:

```json
{
  "format_version": 1,
  "studies": {
    "default": {
      "omop": {
        "connector": "unity-catalog",
        "options": {"catalog": "unity", "schema": "clinical"}
      }
    }
  }
}
```

An optional `tables` list restricts the read to those tables; otherwise all
pages of the schema's table inventory are read. Table and column names, physical
`type_text` and reported nullability are mapped to the shared metadata model.
UC OSS's `TableInfo` does not supply primary/foreign-key constraints, so these
are left empty. Storage locations, arbitrary properties, view SQL and ownership
identities are never copied into published catalogs.

For authenticated endpoints, define a named connector with non-secret arguments
`{"url_env": "STUDY_UC_URL", "token_env": "STUDY_UC_TOKEN"}` and bind datasets
to that name. The site provides the values. A configured token variable must be
present; there is no anonymous fallback. TLS certificate verification stays
enabled for HTTPS, redirects are rejected, and no storage credentials are requested.
`request_timeout_s` defaults to 10 seconds per request.

See the [Unity Catalog API definition](https://github.com/unitycatalog/unitycatalog/blob/main/api/all.yaml)
and [local server setup](https://docs.unitycatalog.io/how-to/run-local-server).

## DuckDBConnector

`nvflare.app_opt.fdn.catalog.duckdb_connector.DuckDBConnector` reads a mounted
DuckDB database file. Install `duckdb>=1.1`, which `nvflare[catalog]` includes.
SQLAlchemy is not used: reflection through `duckdb-engine` fails on DuckDB's catalog, so the connector runs four fixed
queries against DuckDB's own `information_schema` and `duckdb_constraints()`.
Column names, physical types, nullability, primary keys and foreign keys are
published; no application rows are read.

The file is opened read-only with external file access disabled. Optional
descriptive and semantic declarations stay in `<database>.catalog.json`, as for
SQLite. The connector is not built in, so the site names it:

```json
{
  "format_version": 1,
  "connectors": {
    "duckdb": {"class": "nvflare.app_opt.fdn.catalog.duckdb_connector.DuckDBConnector"}
  },
  "default_connector": "duckdb"
}
```

`schema` (default `main`) and `tables` are the only options. Without `tables`,
every ordinary table in the schema is published; views are not. A relationship
is published only when its target is one of the selected tables. DuckDB does not
report the target's schema, so keep related tables in one schema. A DuckDB file
that another process has open for writing cannot be read and is reported as
unavailable.

## Limits and failures

All five connectors return a complete dataset catalog or raise a safe `ConnectorError`.
They enforce record budgets and check the duration budget before/after provider
operations. Parquet counts one call for each footer it reads and for each part
file it lists; CSV counts its one bounded sample read. SQL call accounting includes reflection statements emitted through
the SQLAlchemy connection; DuckDB counts its four metadata queries; Unity Catalog counts HTTP requests, including pages
and detail lookups. A repeated page token or inconsistent inventory fails the read.
Connections and HTTP sessions are closed on success and failure; raw provider
errors, SQL text, endpoints and credentials are not returned.

Duration checks are cooperative, not a universal driver cancellation mechanism.
Configure database connection and statement timeouts for the chosen driver;
HTTP timeouts limit connect/read inactivity. The normal job/process lifecycle
remains responsible for interrupting a provider that never returns. A catalog
read does not establish a transactionally consistent snapshot across concurrent
schema changes or prove that declared constraints are enforced.
