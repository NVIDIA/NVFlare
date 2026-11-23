# Federated data catalog

A researcher calls `get_catalog(env)` to discover the datasets available to a Study. Each site reads its Study configuration and returns one Open Data Contract Standard (ODCS) document for each dataset it can publish. The server collects the documents and reports the outcome for each site. [ODCS v3.2.0](https://github.com/bitol-io/open-data-contract-standard/releases/tag/v3.2.0)

## Site execution

For each selected dataset, `CatalogExecutor` calls `read_odcs(dataset, secret, limits, *, site)`. The reader either validates a steward-provided contract or builds a contract from the source metadata. Live discovery works on the first request and returns the document in memory.

| Site input | What `read_odcs` does |
| --- | --- |
| File | Read the site-configured file. Validate and return a steward-provided ODCS document. For CSV, pass a header-only projection to the importer. For Parquet or a local database, import the footer metadata or table definitions. |
| Live source | Connect to a running database, warehouse or catalog service using the site’s configured identity. Publish the structural fields from the returned ODCS document. |

Generated contracts come from the pinned Data Contract CLI importers. PostgreSQL uses the live `postgres` importer. Unity Catalog uses the `unity` importer with fully qualified table names; the Databricks SDK lists names in the configured catalog and schema when the site omits a table list. The `parquet` importer reads individual footers, including each part of a partitioned table. For local SQLite and DuckDB files the site extracts table DDL through a read-only connection and passes it to the upstream `sql` importer. The current live SQL path supports PostgreSQL. MySQL and Snowflake would require separate integration and qualification. [Upstream Python entry point](https://github.com/datacontract/datacontract-cli/blob/75160dcce44e4ffe3c0be1ca2a9093dd4a194705/datacontract/data_contract.py#L402)

At the per-dataset call site, the executor supplies the authenticated site name for generated contract IDs and a secret lookup scoped to that dataset. The reader chooses the source from site configuration.

```python
contract = read_odcs(dataset, secret, remaining_limits, site=site_name)
catalogs.append({"dataset": dataset.name, "contract": contract})
```

For generated contracts, the site publishes structural ODCS fields, including nested properties and declared keys. Importer-generated server descriptions, comments, storage locations and examples stay at the site. The wheel includes Bitol’s tagged schema; `nvflare[catalog]` installs the qualified importers and drivers.

The executor continues to obtain the job’s Study configuration and handle request selection, cancellation, read budgets and per-dataset outcomes. The reader owns source access and ODCS production. The server handles results from authenticated sites. A reader failure marks that dataset unavailable and allows the site to return its other documents. A budget stop retains the existing `skipped` outcome. [Site executor](https://github.com/chesterxgchen/NVFlare/blob/feature/fdn-catalog-core/nvflare/app_common/fdn/catalog/site/executor.py) · [Upstream import API](https://github.com/datacontract/datacontract-cli/blob/75160dcce44e4ffe3c0be1ca2a9093dd4a194705/datacontract/data_contract.py)

## How the site chooses the source

Use the existing Study-scoped `local/datasets.yaml` and the Config API. A dataset’s existing `type`, `uri`, `options` and `secrets` describe its live source. For a steward-provided contract, the optional setting is `usage.catalog.contract_uri`. The underlying dataset keeps its original type and URI for other jobs. A researcher can request a subset of configured datasets; source locations, credentials and reader implementations belong to the site. [Study dataset configuration API](https://github.com/NVIDIA/NVFlare/pull/5750)

```yaml
format_version: 1
studies:
  oncology:
    datasets:
      registry:
        type: sql
        uri: postgresql://registry.example/clinical
        usage:
          catalog:
            contract_uri: file:///data/oncology/contracts/registry.odcs.yaml
      visits:
        type: csv
        uri: file:///data/oncology/visits.csv
      measurements:
        type: sql
        uri: postgresql+psycopg://measurements.example/study
        options: {schema: approved}
        secrets: {username: STUDY_DB_USER, password: STUDY_DB_PASSWORD}
```

For this configuration, the job reads the registry contract without contacting PostgreSQL, imports the visits CSV and applies the header-name filter, and uses the upstream PostgreSQL importer to read the configured schema for measurements. An unreadable or invalid registry contract produces an unavailable result for that dataset. It never triggers live discovery. A deployed site makes the contract file available to its job through the same workspace or read-only Study mounts used for other file inputs.

The only dataset-level catalog setting is `contract_uri`. When the reference is absent, the configured source determines whether the reader opens a file or connects to a service. Publication boundaries remain the steward-approved file or the site’s configured source access. No catalog policy file is needed.

## What gets published

The site validates each document against ODCS v3.2.0. An unsupported version makes that dataset unavailable.

**Provided contracts.** The steward supplies a document approved for the receiving Study. A curated contract can carry descriptions, units and other declarations that a database cannot infer. Validate the complete document against Bitol’s tagged v3.2.0 JSON Schema and preserve its contract identity and version. A supplied file must be a self-contained YAML or JSON document with unique keys and no YAML aliases. The steward must remove source values and private connection details before sharing it. An ODCS file used internally may contain both.

**Generated contracts.** Qualify the importer for the actual backend and input form. Pass source options explicitly and resolve credentials through the per-dataset secret callback only when connecting. Keep site settings out of the returned document. Construct the public document from ODCS fields for object and property names, reported types, nullability, keys and relationships where the source supports them. Apply this projection recursively, including nested properties. Copying an importer’s entire `schema` subtree can carry examples, observed ranges, free text or local paths into the result. This fixed behavior belongs in the reader and requires no site-authored exclusion rules.

**CSV publication filter (file input).** Read the first CSV record and stage a header-only file with temporary column identifiers for the Data Contract CLI CSV importer. The importer receives no source rows. Check the returned identifiers and publish one property per original column, in header order, with only its `name`. Use CSV header parsing rules for quoted delimiters and reject missing, empty or duplicate column names. A valid header with zero data rows still provides the full published schema. [Upstream CSV importer](https://github.com/datacontract/datacontract-cli/blob/75160dcce44e4ffe3c0be1ca2a9093dd4a194705/datacontract/imports/csv_importer.py)

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

Researchers call `get_catalog(env, sites=..., datasets=..., output_path=...)`. The job stores each ODCS document as a standalone file, along with a small FLARE index recording the Study, job, site outcomes and which file belongs to each successful `(site, dataset)`. The controller assigns file paths under the job output directory from trusted identities or opaque IDs. Sites never supply output paths. Sites can publish the same dataset name or even the same contract ID without overwriting each other.

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

The three-site simulator exercises a supplied contract, a CSV header, Parquet footers and local SQLite and DuckDB table definitions. The provisioned Docker example reads PostgreSQL and Unity Catalog OSS. Its PostgreSQL role can discover table metadata without permission to select rows. A managed Databricks workspace requires separate qualification. Both examples use the same researcher API and offline viewer. [Core implementation](https://github.com/NVIDIA/NVFlare/pull/5738) · [Examples and user guide](https://github.com/NVIDIA/NVFlare/pull/5740)
