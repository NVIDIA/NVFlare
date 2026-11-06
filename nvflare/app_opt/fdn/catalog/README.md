# Catalog connectors

These connectors publish `CatalogInfo` metadata through the existing site
publication policy. They do not select application rows, execute arbitrary SQL,
or process data. Connector selection, source bindings and credentials are
site-owned; a federated catalog task cannot provide them.

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

## Limits and failures

Both readers return a complete dataset catalog or raise a safe `ConnectorError`.
They enforce record budgets and check the duration budget before/after provider
operations. SQL call accounting includes reflection statements emitted through
the SQLAlchemy connection; Unity Catalog counts HTTP requests, including pages
and detail lookups. A repeated page token or inconsistent inventory fails the read.
Connections and HTTP sessions are closed on success and failure; raw provider
errors, SQL text, endpoints and credentials are not returned.

Duration checks are cooperative, not a universal driver cancellation mechanism.
Configure database connection and statement timeouts for the chosen driver;
HTTP timeouts limit connect/read inactivity. The normal job/process lifecycle
remains responsible for interrupting a provider that never returns. A catalog
read does not establish a transactionally consistent snapshot across concurrent
schema changes or prove that declared constraints are enforced.
