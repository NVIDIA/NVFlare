# Federated data catalog

Every `get_catalog(env)` request reads current metadata directly from the datasets configured for the Study. For the first request, the site needs only its source configuration in `local/datasets.yaml` and access to the source. The dataset URI points to the database, catalog service or data file itself. No pre-existing catalog document is required.

The site generates one Open Data Contract Standard (ODCS) document in memory for each dataset it successfully reads. The server saves those documents as job output and reports each site's outcome. A later request reads the sources again so it can discover schema changes. [ODCS v3.2.0](https://github.com/bitol-io/open-data-contract-standard/releases/tag/v3.2.0)

## Site execution

For each selected dataset, `CatalogExecutor` calls `run_catalog_discovery()` in `catalog_discovery_worker.py`. The child process calls `build_odcs_contract()` in `odcs_contract_builder.py` to generate and validate the document.

The site administrator configures the source address in `datasets.yaml`. FLARE translates that address and the dataset options into arguments for the pinned Data Contract CLI importer. The importer returns metadata in memory. FLARE selects the structural fields, builds the ODCS document and validates it before sending it to the server. [Upstream Python entry point](https://github.com/datacontract/datacontract-cli/blob/fc6452e457fc449c4bbd948db15d9cd4eccb73f2/datacontract/data_contract.py#L402)

| Configured source | How the site obtains metadata |
| --- | --- |
| PostgreSQL | FLARE parses the PostgreSQL URI and calls `DataContract.import_from_source(format="postgres", ...)` with the hostname, port, database, schema, optional table selection and resolved credentials. The upstream importer queries `information_schema` and `pg_catalog` for the configured schema, then applies the table selection to the returned metadata. |
| Unity Catalog | FLARE takes the service host from the HTTPS URI. When the site omits a table list, FLARE calls the Databricks SDK to list tables in the configured catalog and schema. It calls the upstream `unity` importer for each fully qualified table name. The importer retrieves the table metadata with `tables.get()`. |
| SQLite or DuckDB | FLARE opens the database file read-only, extracts table definitions and passes them to the upstream `sql` importer as DDL. |
| Parquet | FLARE compares the footer schemas of partition files and passes one representative file per table to the `parquet` importer. Published decimal types include precision and scale. |
| CSV | FLARE reads the header and passes only temporary column identifiers to the CSV importer. It publishes the original header names in order. |

These are the sources exercised by the two runnable examples. For additional network sources a site can select a compatible metadata importer by its registered Data Contract CLI name. For example, a Snowflake dataset sets `type: snowflake`, `uri: snowflake://acme-west` and `options: {metadata_source: host, database: CLINICAL_WAREHOUSE, schema: EHR}`. The site installs the upstream Snowflake driver and binds settings such as `DATACONTRACT_SNOWFLAKE_PASSWORD` to its own environment variables under `secrets`.

For this configuration-driven path, `metadata_source: host` passes the account or host to the importer. Use `metadata_source: uri` for an importer that expects a complete network URI. FLARE passes the remaining importer arguments, nonsecret `options.config` settings and resolved secret bindings to the upstream registry. The site qualifies the importer's metadata access using a scoped source identity. Adding another compatible importer requires configuration and its driver. The site worker applies the same deadline, structural projection and ODCS validation.

At the per-dataset call site, the executor supplies the authenticated site name for generated contract IDs and a secret lookup scoped to that dataset. All datasets in the task share one deadline, call allowance and published-record allowance. The site runs each read in a local child process. At the deadline it can stop a blocked source read, import or validation and return the contracts already completed. The sender reserves space for status entries within a 16 MiB site reply. The controller enforces the same reply limit. The reader chooses the source from site configuration.

```python
contract = run_catalog_discovery(dataset, secret, budget, site=site_name)
catalogs.append({"dataset": dataset.name, "contract": contract})
```

The site publishes structural ODCS fields, including nested properties and declared keys. Importer-generated server descriptions, comments, storage locations and examples stay at the site. The wheel includes Bitol’s tagged schema; `nvflare[catalog]` installs the qualified importers and drivers.

The executor continues to obtain the job’s Study configuration and handle request selection, cancellation, read budgets and per-dataset outcomes. The reader owns source access and ODCS production. The server handles results from authenticated sites. A reader failure marks that dataset unavailable and allows the site to return its other documents. A budget stop retains the existing `skipped` outcome. [Site executor in PR #5738](https://github.com/NVIDIA/NVFlare/pull/5738/files) · [Upstream import API](https://github.com/datacontract/datacontract-cli/blob/fc6452e457fc449c4bbd948db15d9cd4eccb73f2/datacontract/data_contract.py)

## How the site chooses the source

Use the existing Study-scoped `local/datasets.yaml` and the Config API. The dataset’s `type` and `uri` identify its source. The reader maps them to a Data Contract CLI importer and uses `options` and `secrets` for the metadata scope and credentials. A researcher can request a subset of configured datasets; source locations, credentials and reader implementations belong to the site. [Study dataset configuration API](https://github.com/NVIDIA/NVFlare/pull/5750)

```yaml
format_version: 1
studies:
  oncology:
    datasets:
      registry:
        type: sql
        uri: file:///data/oncology/registry/registry.sqlite
      visits:
        type: csv
        uri: file:///data/oncology/visits/visits.csv
      clinical_records:
        type: sql
        uri: postgresql://ehr-db.oncology.example:5432/clinical_warehouse
        options: {schema: omop, tables: [person, visit_occurrence]}
        secrets: {username: EHR_DB_USER, password: EHR_DB_PASSWORD}

usage:
  catalog:
    limits: {max_duration_s: 60, max_calls: 1000, max_records: 10000}
```

For `registry`, the job opens the SQLite database and reads its table definitions. For `visits`, it opens the CSV data file and reads its header. For `clinical_records`, it queries metadata for the `omop` schema in `clinical_warehouse` and publishes the two selected tables. Site-wide catalog limits apply across the selected datasets.

## What gets published

FLARE validates the generated document against the bundled ODCS v3.2.0 schema. A validation failure marks that dataset unavailable.

Qualify the importer for the actual backend and input form. Pass source options explicitly and resolve credentials through the per-dataset secret callback only when connecting. Keep site settings out of the returned document. Construct the public document from ODCS fields for object and property names, reported types, nullability, keys and relationships where the source supports them. Apply this projection recursively, including nested properties. Copying an importer’s entire `schema` subtree can carry examples, observed ranges, free text or local paths into the result. This fixed behavior belongs in the reader and requires no site-authored exclusion rules.

**CSV publication filter (file input).** Read the first CSV record and stage a header-only file with temporary column identifiers for the Data Contract CLI CSV importer. The importer receives header identifiers in batches of up to 64 columns and no source rows. Check the returned identifiers and publish one property per original column, in header order, with only its `name`. Use CSV header parsing rules for quoted delimiters and reject missing, empty or duplicate column names. A valid header with zero data rows still provides the full published schema. [Upstream CSV importer](https://github.com/datacontract/datacontract-cli/blob/fc6452e457fc449c4bbd948db15d9cd4eccb73f2/datacontract/imports/csv_importer.py)

For a CSV headed `visit_id,visit_date`, a generated document looks like the following. The site derives a stable contract ID from the Study, site and configured dataset identity. ODCS `apiVersion` identifies the standard, while `version` identifies the contract revision. A generated draft makes no steward assertion about the data.

```yaml
apiVersion: v3.2.0
kind: DataContract
id: urn:uuid:35c71462-a5bc-5727-a9a5-a25b8c455044
version: 1.0.0
status: draft
name: visits
schema:
  - name: visits
    properties:
      - name: visit_id
      - name: visit_date
```

## How a researcher gets and compares the results

Researchers call `get_catalog(env, sites=..., datasets=..., output_path=...)`. The server stores each generated ODCS document as a standalone output file, along with a small FLARE index recording the Study, job, site outcomes and which file belongs to each successful `(site, dataset)`. The controller assigns file paths under the job output directory from trusted identities or opaque IDs. Sites never supply output paths. Sites can publish the same dataset name or even the same contract ID without overwriting each other.

A downloaded job output can contain `catalog/index.json` and `catalog/contracts/<digest>.odcs.json`. Standard ODCS tooling can consume the contract files directly. The Python helper reads the index and files and returns a dict with `sites` and `datasets`; each successful entry has `site`, `dataset` and `contract`. If requested, `output_path` saves that assembled snapshot. Both the index and snapshot use FLARE `format_version: 2`. The controller isolates malformed documents or replies and records failures without discarding other sites’ results. It does not perform catalog-specific interpretation of ODCS fields.

```python
from nvflare.recipe.catalog import get_catalog

result = get_catalog(env, datasets={"site-1": ["visits"]})
for entry in result["datasets"]:
    for obj in entry["contract"].get("schema", []):
        columns = [prop["name"] for prop in obj.get("properties", [])]
        print(entry["site"], entry["dataset"], obj["name"], columns)
```

Keep the cross-site comparison in FLARE. Read ODCS object and property names directly, identify sources by site and dataset, and distinguish a missing property from a property whose type is unknown. In particular, a CSV header can confirm a column exists while leaving its type unknown. Use the site status to distinguish an unavailable dataset from a successful contract. The same offline viewer can also open saved version 1 snapshots. General-purpose ODCS browsing can use standard tools. [Upstream catalog tooling](https://github.com/datacontract/datacontract-cli)

## Runnable examples

The three-site simulator exercises CSV headers, Parquet footers and local SQLite and DuckDB table definitions. The provisioned Docker example reads PostgreSQL and Unity Catalog OSS. Its PostgreSQL role can discover table metadata without permission to select rows. A managed Databricks workspace requires separate qualification. Both examples use the same researcher API and offline viewer. [Core implementation](https://github.com/NVIDIA/NVFlare/pull/5738) · [Examples and user guide](https://github.com/NVIDIA/NVFlare/pull/5740)
