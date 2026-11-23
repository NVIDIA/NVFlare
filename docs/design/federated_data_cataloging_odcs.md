# Federated catalog reads using ODCS

A researcher runs `get_catalog(env)` to discover the datasets available to a Study. Each participating site returns one Open Data Contract Standard (ODCS) document per successfully read dataset. A site can publish a contract its steward maintains, or generate a structural description from a local source when the job runs. Both paths produce the same document format. [ODCS v3.2.0](https://github.com/bitol-io/open-data-contract-standard/releases/tag/v3.2.0) · [ODCS prototype in PR #5817](https://github.com/NVIDIA/NVFlare/pull/5817)

## Where the ODCS reader fits

Today `CatalogExecutor` selects a `CatalogConnector`, calls `read_catalog(dataset, secret, limits)`, and receives a FLARE `CatalogInfo`. Keep that call site. Replace the connector lookup and `CatalogInfo` return path with a single site-local function, `read_odcs(dataset, secret, limits)`, which returns a native ODCS document. This function is the only FLARE adapter needed. It can dispatch to ordinary functions. There is no need for another plugin registry, a wrapper around every upstream importer, or an ODCS-to-`CatalogInfo` conversion.

| Site input | What `read_odcs` does |
| --- | --- |
| Steward-provided ODCS file | Load the approved file from the site-configured location, check the supported version and document shape, and return the contract. This path opens no source connection and requires no database driver. |
| CSV source | Read the header locally and construct an ODCS schema object with one property name per column. |
| Source with a qualified importer | Translate the site’s source settings and secret callback into the arguments for `DataContract.import_from_source(...)`. Publish the structural fields from the returned ODCS object. |

The executor continues to obtain the job’s Study configuration and handle request selection, cancellation, read budgets and per-dataset outcomes. The reader owns source access and ODCS production. The server handles results from authenticated sites. A reader failure marks that dataset unavailable and allows the site to return its other documents. A budget stop retains the existing `skipped` outcome. [Existing executor call site](https://github.com/NVIDIA/NVFlare/blob/d03b6658ff7553b223da3c87a41d4c248125acf6/nvflare/app_common/fdn/catalog/site/executor.py) · [Upstream import API](https://github.com/datacontract/datacontract-cli/blob/75160dcce44e4ffe3c0be1ca2a9093dd4a194705/datacontract/data_contract.py)

## How the site chooses the source

Use the existing Study-scoped `local/datasets.yaml` and the Config API. A dataset’s existing `type`, `uri`, `options` and `secrets` describe its live source. For a steward-provided contract, the proposed single optional setting is `usage.catalog.contract_uri`. The underlying dataset keeps its original type and URI for other jobs. A researcher can request a subset of configured datasets; source locations, credentials and reader implementations belong to the site. [Study dataset configuration API](https://github.com/NVIDIA/NVFlare/pull/5750)

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
        uri: postgresql://measurements.example/study
        options: {schema: approved}
        secrets: {username: STUDY_DB_USER, password: STUDY_DB_PASSWORD}
```

For this configuration, the job reads the registry contract without contacting PostgreSQL, reads the visits header, and asks the PostgreSQL importer for the measurements schema. An unreadable or invalid registry contract produces an unavailable result for that dataset. It never triggers live discovery. A deployed site makes the contract file available to its job through the same workspace or read-only Study mounts used for other file inputs.

The core PR currently rejects every per-dataset `usage.catalog` entry. The ODCS implementation must accept exactly the new contract reference there and continue to reject obsolete sharing and exclusion rules, including a mixture of old and new keys. A dataset without the reference uses live discovery. Publication boundaries remain the steward-approved file or the site’s configured source access. No catalog policy file is needed.

## What gets published

The initial target is ODCS v3.2.0. Report an unsupported contract version as unavailable for that dataset.

**Provided contracts.** The steward supplies a document approved for the receiving Study. A curated contract can carry descriptions, units and other declarations that a database cannot infer. Validate the file locally and preserve its contract identity and version. The steward must remove source values and private connection details before sharing it. An ODCS file used internally may contain both.

**Generated contracts.** Qualify the importer for the actual backend and input form. Pass source options explicitly and resolve credentials through the per-dataset secret callback only when connecting. Keep site settings out of the returned document. Construct the public document from ODCS fields for object and property names, reported types, nullability, keys and relationships where the source supports them. Apply this projection recursively, including nested properties. Copying an importer’s entire `schema` subtree can carry examples, observed ranges, free text or local paths into the result. This fixed behavior belongs in the reader and requires no site-authored exclusion rules.

**CSV.** Read the first logical record with a CSV parser so quoted delimiters are handled correctly. Use the configured dataset name for the schema object and require nonempty, unique column names. Return the names in header order and leave column types, nullability, keys and examples unspecified. A header-only file produces a valid result; an absent or invalid header makes the dataset unavailable. The upstream CSV importer profiles rows and can attach sampled values to properties, so it does more work than this path needs. [Upstream CSV importer](https://github.com/datacontract/datacontract-cli/blob/75160dcce44e4ffe3c0be1ca2a9093dd4a194705/datacontract/imports/csv_importer.py)

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

Preserve the `get_catalog(env, sites=..., datasets=..., output_path=...)` entry point. The job stores each ODCS document as a standalone file, along with a small FLARE index recording the Study, job, site outcomes and which file belongs to each successful `(site, dataset)`. The controller assigns file paths under the job output directory from trusted identities or opaque IDs. Sites never supply output paths. Sites can publish the same dataset name or even the same contract ID without overwriting each other.

A downloaded job output can contain `catalog/index.json` and `catalog/contracts/0001.odcs.yaml`. Standard ODCS tooling can consume the contract files directly. The Python helper reads the index and files and returns a dict with `sites` and `datasets`; each successful entry has `site`, `dataset` and `contract`. If requested, `output_path` saves that assembled snapshot. Give the new index and snapshot a FLARE `format_version` of 2 to distinguish them from saved version 1 results. The controller isolates malformed documents or replies and records failures without discarding other sites’ results. It does not perform catalog-specific interpretation of ODCS fields.

```python
from nvflare.recipe.catalog import get_catalog

result = get_catalog(env, datasets={"site-1": ["visits"]})
for entry in result["datasets"]:
    for obj in entry["contract"].get("schema", []):
        columns = [prop["name"] for prop in obj.get("properties", [])]
        print(entry["site"], entry["dataset"], obj["name"], columns)
```

Keep the cross-site comparison in FLARE. Read ODCS object and property names directly, identify sources by site and dataset, and distinguish a missing property from a property whose type is unknown. In particular, a CSV header can confirm a column exists while leaving its type unknown. Use the site status to distinguish an unavailable dataset from a successful contract. Existing version 1 snapshots can still be opened with the version 1 viewer. General-purpose ODCS browsing can use standard tools. [Upstream catalog tooling](https://github.com/datacontract/datacontract-cli)

## Adoption in the implementation and examples

The existing [core implementation](https://github.com/NVIDIA/NVFlare/pull/5738) and [examples](https://github.com/NVIDIA/NVFlare/pull/5740) implement the FLARE-specific baseline. Switching the reader and switching its consumers must ship together. Adding the ODCS license to the core PR only records the standard’s license.

1. Qualify provided-file collection and header-only CSV end to end. Check that a missing supplied file never opens a live source and that a CSV containing distinctive sample values returns the same contract as its header alone.
2. Compare each proposed upstream importer with the sources the existing examples exercise. Single-file and directory Parquet are distinct cases. Live SQLite, DuckDB, PostgreSQL with selected views, and paginated Unity Catalog each need a check for selection scope, reported types and keys, site credentials, and bounded work. A generic SQL DDL importer does not establish live database support. Where upstream cannot meet a required case, use a focused source reader that emits ODCS directly or a steward-provided contract. Retire each baseline connector once its supported inputs have a verified replacement.
3. Exercise two sites publishing the same dataset name, a malformed document beside a valid one, and a selective read with an offline site. Validate the generated example against the pinned ODCS schema. Open the downloaded files with an ODCS consumer and verify that the comparison reads the native contracts, including header-only CSV and unknown types. [ODCS v3.2.0 JSON Schema](https://github.com/bitol-io/open-data-contract-standard/blob/v3.2.0/schema/odcs-json-schema-v3.2.0.json)
