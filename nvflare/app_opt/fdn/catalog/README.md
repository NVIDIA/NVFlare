# Site catalog reader

Install `nvflare[catalog]` at sites running the federated catalog job. The site configures datasets in its Study’s `local/datasets.yaml`. The executor calls `read_odcs` once per selected dataset.

| Site configuration | Metadata published |
| --- | --- |
| `usage.catalog.contract_uri: file:///…` on any dataset | The approved ODCS v3.2.0 contract. An unreadable or invalid contract marks the dataset unavailable. |
| `type: csv` | Only original header names, in order. The importer receives a header projection. Data rows, types, source paths and statistics stay out of the contract. |
| `type: parquet` | Import one representative footer per table after checking that all partition footers have the same Arrow schema. Decimal type names include precision and scale. |
| `type: sql` | Live PostgreSQL importer for a `postgresql://` URI and site credentials, or upstream SQL DDL import for table definitions in a read-only SQLite file. Scope PostgreSQL with `options.schema` and `options.tables`. |
| `type: duckdb` | Upstream SQL importer on table definitions from the mounted file opened read-only. |
| `type: unity-catalog` | Upstream Unity importer using the Databricks SDK in the configured `catalog` and `schema`, optionally limited by `tables`. |

Site credentials use dataset-scoped secret bindings. SQL accepts `username` and `password`; Unity Catalog accepts a bearer `token`. Site-wide `usage.catalog.limits` bounds cooperative read time, importer calls and published records. Source permissions and the approved contract file determine what the Study can discover.

The validator uses [Bitol’s released ODCS 3.2.0 schema](https://github.com/bitol-io/open-data-contract-standard/blob/v3.2.0/schema/odcs-json-schema-v3.2.0.json). The CSV, PostgreSQL, Unity, Parquet and SQL importers ship with the optional MIT-licensed `datacontract-cli` package.

For discovery, read limits apply to importer calls and returned records. Upstream calls can execute multiple database statements or API requests internally. PostgreSQL uses its information schema, which requires privileges on the selected tables. Local database file import accepts table DDL; it does not reflect views.
