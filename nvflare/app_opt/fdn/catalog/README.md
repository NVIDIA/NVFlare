# Site catalog reader

Install `nvflare[catalog]` at sites running the federated catalog job. The site configures datasets in its Study’s `local/datasets.yaml`. The executor calls `run_catalog_discovery` once per selected dataset.

| Site configuration | Metadata published |
| --- | --- |
| `type: csv` | Only original header names, in order. The importer receives a header projection. Data rows, types, source paths and statistics stay out of the contract. |
| `type: parquet` | Import one representative footer per table after checking that all partition footers have the same Arrow schema. Decimal type names include precision and scale. |
| `type: sql` | Live PostgreSQL importer for a `postgresql://` URI and site credentials, or upstream SQL DDL import for table definitions in a read-only SQLite file. Scope PostgreSQL with `options.schema` and `options.tables`. |
| `type: duckdb` | Upstream SQL importer on table definitions from the mounted file opened read-only. |
| `type: unity-catalog` | Upstream Unity importer using the Databricks SDK in the configured `catalog` and `schema`, optionally limited by `tables`. |

Site credentials use dataset-scoped secret bindings. SQL accepts `username` and `password`; Unity Catalog accepts a bearer `token`. Site-wide `usage.catalog.limits` sets a shared deadline, call allowance and record allowance. Schema discovery runs in a site-local worker. The executor stops and reaps the worker at the deadline, then returns completed datasets. Site replies have a 16 MiB JSON limit. Source permissions and the configured metadata selection determine what the Study can discover.

The validator uses [Bitol’s released ODCS 3.2.0 schema](https://github.com/bitol-io/open-data-contract-standard/blob/v3.2.0/schema/odcs-json-schema-v3.2.0.json). The CSV, PostgreSQL, Unity, Parquet and SQL importers ship with the optional MIT-licensed `datacontract-cli` package.

For discovery, read limits apply to importer calls and returned records. Upstream calls can execute multiple database statements or API requests internally. PostgreSQL uses its information schema, which requires privileges on the selected tables. Local database file import accepts table DDL; it does not reflect views.

A PostgreSQL dataset can require certificate and hostname validation with
`postgresql://db.example/study?sslmode=verify-full`. Configure the trusted CA through
`PGSSLROOTCERT` in the site runtime. Without a dataset override, libpq uses its normal site settings,
including `PGSSLMODE`. The deadline also covers connection setup, retries and catalog queries.
