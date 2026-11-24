# Site catalog reader

Install `nvflare[catalog]` at sites running the federated catalog job. The site configures datasets in its Study’s `local/datasets.yaml`. The executor calls `run_catalog_discovery` once per selected dataset.

| Site configuration | Metadata published |
| --- | --- |
| `type: csv` | Only original header names, in order. The importer receives a header projection. Data rows, types, source paths and statistics stay out of the contract. |
| `type: parquet` | Import one representative footer per table after checking that all partition footers have the same Arrow schema. Decimal type names include precision and scale. |
| `type: sql` | Live PostgreSQL importer for a `postgresql://` URI and site credentials, or upstream SQL DDL import for table definitions in a read-only SQLite file. Scope PostgreSQL with `options.schema` and `options.tables`. |
| `type: duckdb` | Upstream SQL importer on table definitions from the mounted file opened read-only. |
| `type: unity-catalog` | Upstream Unity importer using the Databricks SDK in the configured `catalog` and `schema`, optionally limited by `tables`. |
| Other upstream metadata importers | Set `type` to the registered importer name and `options.metadata_source` to `host` or `uri`. Install the importer's optional driver at the site. |

For example, a site with `datacontract-cli[snowflake]==1.2.11` can add this dataset under its Study in `datasets.yaml`:

```yaml
clinical_records:
  type: snowflake
  uri: snowflake://acme-west
  options:
    metadata_source: host
    database: CLINICAL_WAREHOUSE
    schema: EHR
    config: {DATACONTRACT_SNOWFLAKE_WAREHOUSE: CATALOG_WH}
  secrets:
    DATACONTRACT_SNOWFLAKE_USERNAME: SITE_SNOWFLAKE_USER
    DATACONTRACT_SNOWFLAKE_PASSWORD: SITE_SNOWFLAKE_PASSWORD
```

`host` passes the bare host or account identifier as the upstream `source`; put database, schema or port in the importer's named options. `uri` passes the complete network URI to importers that expect a URL. Other options go to `DataContract.import_from_source()` as keyword arguments. Nonsecret upstream settings go in `options.config`. Keys under `secrets` are upstream setting names and their values name site environment variables.

An upstream credential setting in `options.config` makes the dataset configuration invalid. For example, bind `DATACONTRACT_SNOWFLAKE_PASSWORD` under `secrets`, as shown above. The validation error directs the operator to `secrets` and omits the value.

Use this form for an importer and source identity verified to read metadata only. The upstream registry also contains importers that sample data. FLARE's structural projection controls what is published after a read. File inputs such as CSV continue through their dedicated metadata-only read paths.

Site credentials use dataset-scoped secret bindings. SQL accepts `username` and `password`; Unity Catalog accepts a bearer `token`. Site-wide `usage.catalog.limits` sets a shared deadline, call allowance and record allowance. Schema discovery runs in a site-local worker. The executor stops and reaps the worker at the deadline, then returns completed datasets. Site replies have a 16 MiB JSON limit. Source permissions and the configured metadata selection determine what the Study can discover.

The validator uses [Bitol’s released ODCS 3.2.0 schema](https://github.com/bitol-io/open-data-contract-standard/blob/v3.2.0/schema/odcs-json-schema-v3.2.0.json). The CSV, PostgreSQL, Unity, Parquet and SQL importers ship with the optional MIT-licensed `datacontract-cli` package.

For discovery, read limits apply to importer calls and returned records. Upstream calls can execute multiple database statements or API requests internally. PostgreSQL uses its information schema, which requires privileges on the selected tables. Local database file import accepts table DDL; it does not reflect views.

A PostgreSQL dataset can require certificate and hostname validation with
`postgresql://db.example/study?sslmode=verify-full`. Configure the trusted CA through
`PGSSLROOTCERT` in the site runtime. Without a dataset override, libpq uses its normal site settings,
including `PGSSLMODE`. The deadline also covers connection setup, retries and catalog queries.
