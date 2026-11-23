# Site catalog reader

Install `nvflare[catalog]` at sites running the federated catalog job. The site configures datasets in its Study’s `local/datasets.yaml`. The executor calls `read_odcs` once per selected dataset.

| Site configuration | Metadata published |
| --- | --- |
| `usage.catalog.contract_uri: file:///…` on any dataset | The approved ODCS v3.2.0 contract. An unreadable or invalid contract marks the dataset unavailable. |
| `type: csv` | Only original header names, in order. The importer receives a header projection. Data rows, types, source paths and statistics stay out of the contract. |
| `type: parquet` | Physical types and nullability from local file or directory footers. |
| `type: sql` | Tables, columns and available constraints from SQLAlchemy reflection. Use a local SQLite file or install the dialect driver for a network database such as PostgreSQL. Scope discovery with `options.schema` and `options.tables`. |
| `type: duckdb` | Metadata and constraints from the mounted database file opened read-only. |
| `type: unity-catalog` | Paginated Unity Catalog REST metadata in the configured `catalog` and `schema`, optionally limited by `tables`. |

Site credentials use dataset-scoped secret bindings. SQL accepts `username` and `password`; Unity Catalog accepts a bearer `token`. Site-wide `usage.catalog.limits` bounds time, calls and records. Source permissions and the approved contract file determine what the Study can discover.

The validator uses [Bitol’s released ODCS 3.2.0 schema](https://github.com/bitol-io/open-data-contract-standard/blob/v3.2.0/schema/odcs-json-schema-v3.2.0.json). The CSV importer ships with the optional MIT-licensed `datacontract-cli` package.
