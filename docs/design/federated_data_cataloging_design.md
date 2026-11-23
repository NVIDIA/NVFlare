# Federated Data Cataloging Design

**Status:** Proposed implementation design
**Date:** November 9, 2026

## 1. Purpose and assumptions

Discover and describe the datasets of an existing FLARE Study. A catalog read is an ordinary FLARE job. Each site reads the metadata of its own datasets, returns the metadata visible through the source identity and files its administrator configured. The result is saved with the job's workspace in the job store, and the user who submitted the read downloads it. The server keeps no separate catalog and merges nothing. Every read returns the complete current answer, and the latest read supersedes all earlier ones.

Use FLARE's existing concepts:

| Existing concept | Use in this design |
| --- | --- |
| Study | The existing Study name and membership; the job's Study selects which datasets a site reads |
| Participating users | Existing users of the Study; permissions come from FLARE authorization |
| Client | The participating site running the job; identity comes from FLARE |
| Dataset | A dataset the site declares for the Study in its [dataset configuration][dataset-config] |
| Job, Task, Shareable | Execution, request/result correlation, transport and serialization |
| Job result download | Delivery of the catalog to the user who ran the read |

The site chooses the datasets for the Study and configures the files, database views and credentials used to read their metadata. Catalog visibility never grants access to source records.

### 1.1 Architecture at a glance

**Figure 1. Where each part runs.** Sites decide what leaves them; the server never reads a source, and the only copy of a result is the one saved with its job in the job store.

![Architecture at a glance: user, FLARE server and each site, and what talks to what](../resources/catalog_design_f01_architecture.svg)

| To understand | Read |
| --- | --- |
| One read from request to result | Section 2, then Section 5 |
| How a site selects its Study datasets and reads their metadata | Section 3 |
| The metadata model and the checks that enforce it | Section 3.4, then the [schema details](federated_data_cataloging_schema.md) |
| The result the user receives | Section 4 |
| Running a read and using the result | Section 6 |
| Choices not taken | Section 9 |

## 2. Read flow

A read follows the arrows of Figure 1. The user submits the catalog job. The `CatalogController` sends each site a `catalog_read` task, and each `CatalogExecutor` reads its Study's dataset configuration and the source-visible metadata of each selected dataset, validates, and returns its result. The controller validates every entry and writes `catalog_results.json`. When the job finishes, the job runner saves the workspace to the job store, and the user downloads the result.

The user calls `get_catalog(...)` or runs `CatalogRecipe` from Python (Section 6). The job takes its Study from existing FLARE job metadata and the client identity from FLARE execution context. The only selector a request carries is an optional list of dataset names per site. No request supplies a filesystem path, database endpoint, credential, connector or Python import path.

Each read is a bounded full metadata read of every selected dataset. There is no incremental checkpoint, change feed, version counter or comparison with an earlier read. Reads are independent: two reads may run at the same time, and each returns its own complete result. The latest completed read is the current catalog and supersedes earlier reads; the server never merges one read into another.

## 3. Site side

**Figure 2. What the site executor does for one task.** Every selected dataset ends as `catalog`, `unavailable` or `skipped`; a configuration failure fails the whole read.

![What the site executor does for one task](../resources/catalog_design_f02_site_executor.svg)

### 3.1 Dataset configuration

The executor reads the site's datasets and connector settings only through the read-only [dataset configuration API][dataset-config]. The configuration of the job's Study is loaded once per job, so an edited configuration applies to the next read, which is a new job. It does not open configuration files, detect the deployment mode or parse launcher configuration.

| Catalog need | Source |
| --- | --- |
| The Study's datasets | `list_datasets()` |
| How to read one | `get_dataset(name)`: `type`, `uri` and `options` |
| Credentials | `get_secret(name, key)`, called by the connector when it connects |
| Read limits, extra connectors, installed operations | `site_usage("catalog")` (Section 3.3) |

Container launchers mount a file-based dataset at `/data/<study>/<dataset>`, and its configuration gives that location as `uri: file:///data/<study>/<dataset>`. The mount comes from the existing [Study runtime configuration](study_runtime_config.md). A process-launched site mounts nothing; its configuration gives the `file` URI where the data already is. Network sources, such as a SQL server or Unity Catalog, use site-owned options and secrets, never a mount.

### 3.2 Connectors

The dataset's `type` selects the connector class. The built-in classes are in `nvflare/app_opt/fdn/catalog/`:

| Dataset `type` | Connector class | Metadata read | Status |
| --- | --- | --- | --- |
| `parquet` | `ParquetConnector` | Column names, physical types and nullability from the file footers under the `file` URI, including partitioned files. Footers carry no keys or relationships, so none are published. | Implemented; demonstrated in the OMOP example. |
| `csv` | `CSVConnector` | PyArrow infers types from the first 64 KiB block of the file the `file` URI names, with at most 1 MiB staged for record boundaries. One UTF-8 file with unique nonempty headers. Arrow type names are preserved; nullability is unknown; keys are not inferred; leading zeros can be lost in inferred integers. Only schema metadata is published, never row values. | Implemented. |
| `sql` | `SQLConnector` | SQLAlchemy reflection of tables, columns, types, nullability and declared constraints. A `file` URI names a SQLite file, opened read-only. Any other URI is a SQLAlchemy database URL, such as `postgresql+psycopg://host:5432/db`, completed with the `username` and `password` secrets. `options` may set `schema` and `tables`. No query over application rows. | SQLite implemented. Each network database and version needs qualification and its installed driver. |
| `duckdb` | `DuckDBConnector` | DuckDB's own `information_schema` and `duckdb_constraints()` on the database file the `file` URI names, opened read-only with external access disabled. DuckDB is a SQL database too, but SQLAlchemy reflection fails on its catalog, so it has its own type and connector. | Not resolved automatically: the site declares `nvflare.app_opt.fdn.catalog.duckdb_connector.DuckDBConnector` under `usage.catalog.connectors`; requires `duckdb>=1.1`. A file held open for writing is unavailable. |
| `unity-catalog` | `UnityCatalogConnector` | The [list-tables REST API](https://docs.databricks.com/api/uc-tables/v1/list-tables) at `/api/2.1/unity-catalog`: column names, `type_text` and nullability. The `uri` is the HTTPS API root, `options` requires `catalog` and `schema`, with optional `tables`, and `token` is a secret. No table data, storage credentials or object paths are fetched, and no lineage or keys (list-table responses omit constraints). | Exercised against UC OSS; Databricks-shaped responses and pagination covered by contract tests; live workspace privileges need qualification. |

Site-wide settings may declare extra connector classes selected by the dataset's `type` (Section 3.3). An unknown name fails that dataset; there is no type guessing or fallback after a selected connector fails.

**Optional dependencies.** Sites install `nvflare[catalog]`, which supplies `pyarrow>=14`, `SQLAlchemy>=2.0,<3` and `duckdb>=1.1`; a network SQL database also needs its driver. Unity Catalog uses the existing Requests dependency.

**Interface.** A connector reads one dataset per call, so a bad source cannot fail a shared constructor or the other datasets.

```python
class CatalogConnector(ABC):
    source: str                                  # e.g. "parquet"; copied to provenance.source
    def read_catalog(self, dataset: DatasetConfig, secret: Callable[[str], str], limits: ReadLimits) -> CatalogInfo: ...
    def close(self) -> None: ...
```

`DatasetConfig` is the API's frozen dataset definition. `secret(key)` reads one of that dataset's secrets on demand and cannot reach another dataset's.

**Figure 3. Connectors.** The executor builds one connector per connector name its selected datasets use, and asks it for one dataset at a time.

![Connectors implementing the CatalogConnector interface](../resources/catalog_design_f03_connectors.svg)

**Custom reading logic.** A site that needs another reader writes a `CatalogConnector` subclass, installs it, and declares it under `usage.catalog.connectors`; the dataset's `type` selects its declared name. Code that reads a site's data therefore always comes from the site, never from a job, so the metadata-only guarantees hold for every connector the site chooses to install.

**Common contract.** Report what the source has: observed names, types and case are not rewritten or harmonized. Publish only what the source itself reports: keys and relationships appear only when the source exposes them as constraints, and a reported constraint is not proof that it is enforced. Descriptive and semantic fields that no source reports, such as a description, owner or declared standard, stay null; no other file supplies them. Any future authored annotations need a separate, explicit metadata contract; they are not publication rules.

### 3.3 Source-owned visibility and catalog settings

The catalog reads the datasets configured for the job's Study through the
[dataset configuration][dataset-config], and the site-wide `catalog` settings. Every configured
dataset is eligible for a read; an optional request selection can narrow that set but cannot add another source.
There is no catalog publication policy, sharing flag or table/column/metadata exclusion layer.

| Catalog need | API call |
| --- | --- |
| Which datasets the Study has at this site | `list_datasets()` |
| How to read a dataset | `get_dataset(name)`: the type selects the connector; URI, options and secrets locate the source |
| Connection credentials | `get_secret(name, key)`, called by the connector |
| Read limits, extra connectors, installed operations | `site_usage("catalog")` |

```yaml
format_version: 1
studies:
  oncology:
    datasets:
      omop:
        type: sql
        uri: postgresql+psycopg://ehr-db.example:5432/clinical
        options: {schema: study_oncology}
        secrets: {username: CATALOG_USER, password: CATALOG_PASSWORD}
usage:
  catalog:
    limits: {max_duration_s: 60, max_calls: 1000, max_records: 10000}
```

The database administrator owns metadata visibility: use a scoped identity and a source namespace or selected
projection whose schema may be disclosed to the Study. Verify metadata visibility for the chosen provider;
restricting access to rows does not necessarily hide names and columns in system catalogs. For SQL views, set `options.tables: [approved_view]` to reflect the intended view. Default SQL discovery lists base tables. For file sources, the
site administrator controls readable files/directories and mounts. Source-provided descriptions, references and
lineage are returned without catalog redaction, so that responsibility includes those fields.

**Site-wide settings** (`usage.catalog`):

| Field | Default | Meaning |
| --- | --- | --- |
| `limits` | `max_duration_s=60`, `max_calls=1000`, `max_records=10000` | Positive integers; omitted values keep defaults. |
| `connectors` | `{}` | Extra connector names mapped to `{class: "fully.qualified.Connector", args: {}}`. `args` is non-secret; a dataset's `type` selects the connector name. |
| `installed_operations` | `[]` | Installed site capabilities. Connector-advertised operations are intersected with this set; advertising grants no execution rights. |

Invalid site-wide settings fail the read with `CONFIG_UNAVAILABLE`. Per-dataset `usage.catalog` is no longer
supported. If any dataset in the Study retains these settings, the site returns `CONFIG_UNAVAILABLE`
without disclosing dataset names, even for a selective read. Other components' dataset usage remains independent.

The executor validates the complete source-reported catalog and returns no partial schema after validation or
source failure. It does not repair relationships or keys by deleting metadata. A later source permission change
is reflected by the next read; already downloaded snapshots are not retracted.

### 3.4 Data structures and metadata format

Class names follow the Unity Catalog SDK: one dataset is one `CatalogInfo`, with nested `TableInfo` and `ColumnInfo`.

```python
@dataclass
class ReadLimits:
    max_duration_s: int
    max_calls: int
    max_records: int

@dataclass(frozen=True)
class CatalogInfo:
    dataset: str                         # the dataset name in the Study's configuration
    # typed fields of the format below, built and validated through from_wire;
    # metadata_wire() returns the metadata dict
    observed_at: str | None              # source observation, RFC 3339 UTC; None means unknown
```

#### Metadata format (normative)

`format_version: 1` selects this schema. Every field is present; unknown values are JSON `null` and lists may be empty. Unknown fields are rejected. Identity is the enclosing `dataset` plus the authenticated Study and client.

| Field | Type and meaning |
| --- | --- |
| `display_name` | Nonempty string |
| `description`, `owner` | String or null; approved description and public steward alias |
| `tables` | List of table objects defined below |
| `resources` | List of FHIR resource objects defined below |
| `semantic_profile` | Profile object defined below; declarations, not inferred validation |
| `modalities` | Unique strings, e.g. `clinical_tabular` |
| `operations` | Unique site-chosen operation names, e.g. `cohort.count`, intersected with installed capabilities as in Section 3.3 |
| `source_version` | Opaque source revision string or null; never synthesized from read time |
| `lineage` | List of `{dataset, source_version, relation: "derived_from"}` reported by the source for this client; reference visibility is source-admin controlled |
| `provenance` | `{source, declaration_ref, validation_ref}`; `source` is the connector's `source` label, never a path, endpoint or credential |

Nested objects have exactly these fields:

- Table: `name`, `columns`, `primary_key: list[string]`, `relationships`. A relationship is `{columns, target_table, target_columns}` with equal, nonzero column-list lengths.
- Column: `name`, `type`, `nullable: bool or null`. `type` is exactly what the source reports, such as `int64` or `VARCHAR(20)`; the catalog maps no types.
- Resource: `type`, `supported_search: list[string]`.
- Profile: `standard`, `version`, `vocabulary_version`, `mapping_version`, `declaration_ref`, `validation_ref`, all strings or null. Declared by the site, never inferred.

Strings except `description` are nonempty and at most 256 characters; `description` is at most 4096 and may be empty. Names are unique within their list, and lineage is unique by `dataset`. Only `operations` is sorted; everything else is reported as given. Primary keys and relationships must resolve within the entry. `max_records` counts dataset, table, column, relationship, resource and lineage objects across the whole site read. A dataset name appearing twice in one result rejects every copy.

**Where the schema is enforced.** The site validates the source-reported metadata before returning each catalog. The server-side controller parses every entry again with `CatalogInfo.from_wire` and validates it with its own limits before writing the result, so a user never receives an entry that does not match this format. The model, both checks and example errors are in the [schema details](federated_data_cataloging_schema.md).

### 3.5 Task payload and site result

Use the existing `Task`, `Shareable`, return codes, authenticated client identity and FOBS serialization. The payload carries one application format version and no transport envelope, task IDs or identities.

```json
{"catalog_read": {"format_version": 1, "datasets": ["omop", "registry"], "time_budget_s": 238.5}}
```

An omitted `datasets` selects every dataset the Study's configuration lists. A selected name the configuration does not list is ignored, and the request cannot add one. `time_budget_s` is the reading time the site's task has left, set by the controller when it sends the task (Section 3.6).

A site result:

```json
{
  "catalog_read": {
    "format_version": 1,
    "status": "OK",
    "catalogs": [
      {
        "dataset": "omop",
        "metadata": {
          "display_name": "OMOP", "description": null, "owner": null,
          "tables": [], "resources": [],
          "semantic_profile": {
            "standard": null, "version": null, "vocabulary_version": null,
            "mapping_version": null, "declaration_ref": null, "validation_ref": null
          },
          "modalities": ["clinical_tabular"], "operations": [],
          "source_version": null, "lineage": [],
          "provenance": {"source": "parquet", "declaration_ref": null, "validation_ref": null}
        },
        "observed_at": null
      }
    ],
    "unavailable": ["registry"],
    "skipped": []
  }
}
```

| Outcome for a selected dataset | Where it appears |
| --- | --- |
| Selected, read and validated | `catalogs` |
| Configured, but its source was absent, inaccessible or invalid, or its connector was unknown | `unavailable` |
| Selected, but not read completely because a time or record budget ran out | `skipped` |
| Not configured for the job's Study | Not read or named as a dataset result |

The three lists are disjoint. A non-empty `skipped` list is the signal to raise the site's limits; it carries no reason code. `status` is `OK` when the read completed, even with unavailable or skipped datasets or no configured datasets. `FAILED`, with a safe `reason_code`, means the read as a whole could not run: `CONFIG_UNAVAILABLE` when the dataset configuration or the catalog settings are invalid (a missing configuration file reads as no datasets, and missing catalog settings use the defaults), `INTERNAL` otherwise. A failed read returns no catalogs. Any other status is read as `FAILED`. Source exceptions, paths and credentials never appear in a result; details go to the site log.

### 3.6 Budgets and failure isolation

Each connector call receives `max_calls` and the remaining whole-read time and record budgets; the executor counts records across datasets rather than granting each dataset a fresh allowance. A dataset that exceeds a budget is discarded whole, never truncated, and is `skipped`, not `unavailable`: the budget, not the source, stopped it. When the whole-read budget runs out, that dataset and the remaining selected datasets are `skipped`, and the completed catalogs are still returned. A corrupt footer, missing source or denied access affects only that dataset. The executor reopens a failed connection before the next dataset. The whole read ends by the site's `max_duration_s` or the task's `time_budget_s`, whichever comes first. The task timeout runs from when the server schedules the task, so the controller computes the budget when it sends the task to the site: the time the task has left, less a reserve of 20% of the timeout (between 1 and 30 seconds) to return completed catalogs. A connector must honor the remaining time it is given; one that blocks past it can still lose the site's result. A framework timeout, abort or disconnect produces no site result; the job result records that site as `NO_RESULT`.

## 4. The catalog result

The controller collects one result per site and writes `catalog_results.json` in the job's server workspace. When the job finishes, the existing job runner saves that workspace, including the file, to the job store, so the result persists with the job. The recipe downloads it after completion and loads the file (Section 6), and the same result can be downloaded later from the job store. FLARE's job download returns the whole job workspace (configuration, logs and other artifacts). After a completed read, the recipe keeps only `catalog_results.json` from it; after a failed read it keeps the whole workspace for diagnosis. `SimEnv` returns its live simulation workspace rather than a download; the recipe leaves it unchanged, since later runs reuse its site configuration. A download that fetches only a job's result files would avoid transferring the rest; that is a general FLARE improvement, not part of this design. There is no catalog store, index or search service beyond that.

**Last read wins.** Each read's result is complete on its own. The latest completed read supersedes the earlier ones: nothing is carried over from or merged with an earlier read. Earlier reads stay in the job store as ordinary finished jobs until they are deleted like any other job.

```json
{
  "format_version": 1,
  "study": "oncology",
  "job_id": "13fa1c55-...",
  "completed_at": "2026-11-09T18:40:12Z",
  "sites": {
    "site-1": {"status": "OK", "reason_code": null, "unavailable": ["registry"], "skipped": []},
    "site-2": {"status": "NO_RESULT", "reason_code": null, "unavailable": [], "skipped": []}
  },
  "datasets": [
    {"site": "site-1", "dataset": "omop", "metadata": {"display_name": "OMOP", "...": "..."}, "observed_at": null}
  ]
}
```

- `sites` has one entry per site the read targeted. `status` is the site's `OK` or `FAILED`, or `NO_RESULT` when no valid application result arrived: a timeout, disconnect or abort, or a non-OK framework return code such as `TASK_UNSUPPORTED`. A site named in `sites` that never joins the job (offline, or not scheduled) is `NO_RESULT` too; without `sites`, the entries are the sites the job ran on. A site's entry is the health report for this read; there is no stored health.
- `datasets` lists each accepted catalog with its site. Two sites with a dataset of the same name stay separate entries. The controller rejects an entry that fails `from_wire` or validation, adds that dataset to the site's `unavailable` list, and keeps the site's other entries. It does not merge schemas, infer equivalence or combine populations.
- `completed_at` is server time when the controller wrote the result. `observed_at` is the source's own observation time, stored as null when it is not valid RFC 3339 UTC or is more than 300 seconds after `completed_at`.

The result is the complete answer of that read. Dataset entries cover the configured sources selected and successfully read for this job. Source-reported lineage may also name other datasets. To compare two reads, the user compares their results.

## 5. Job execution

A catalog read is the standard Job Recipe `CatalogRecipe`. One job covers every targeted site of the Study. `CatalogController` sends the task with each site's optional dataset selection, waits up to `task_timeout_s` for each site, validates the results and writes `catalog_results.json`. `CatalogExecutor` reads the dataset configuration, reads the selected configured datasets and returns a `Shareable`. The job carries the meta property `fdn_catalog_read`; the executor rejects a task from a job without it before reading configuration or sources.

The job uses only the execute task. There is no server-side manager, job hook, admission check, result inbox or persistence. Cancellation uses the normal job and task lifecycle, and an interrupted read returns no result for that site.

**Trust.** The site is the trust boundary: each site configures the source identity, metadata visibility and readable files used by its Study, regardless of which server components a job carries. Once returned, the result is an ordinary job result. Who may download it follows the existing `download_job` authorization of the job's Study, as for any other job. Where that authorization denies the submitter a download, as the confidential-computing default policy does for `lead`, the submitter cannot retrieve the catalog either. Any member allowed to submit jobs to the Study may run a read.

**Latency.** Catalog tasks finish in well under a second, so the recipe sets the server's `task_request_interval` for this job to 0.1 s instead of the 2 s default; `task_request_interval_s` changes it. Otherwise each site waits up to 2 s after returning its result before it notices the run has ended.

**Refresh.** There is no special refresh command and no built-in scheduler. A user who wants regular reads runs the recipe from an external scheduler such as cron. Each read is a job in the job store, like any other job.

## 6. User experience

Users read the catalog from Python, in one of two ways:

| Need | Use |
| --- | --- |
| Read the catalog in one call | `get_catalog(...)` (Section 6.1) |
| Control the job, for example submitting without waiting | `CatalogRecipe` (Section 6.2) |

Both build the same job, so a read behaves the same whichever is used.

### 6.1 `get_catalog`: read in one call

```python
from nvflare.recipe import ProdEnv
from nvflare.recipe.catalog import get_catalog

env = ProdEnv(startup_kit_location="/path/to/admin", study="oncology")
results = get_catalog(
    env,
    datasets={"site-1": ["omop"]},
    output_path="catalog_results.json",
)
```

It builds a `CatalogRecipe`, runs it, waits for the job and returns the parsed `catalog_results.json` (Section 4). The recipe reports progress and a run summary in the simulator. Every call submits a new job that reads the sites again, so keep the returned result, or the saved file, and work from it rather than calling again to look at the same data.

**How it relates to `CatalogRecipe`.** `get_catalog` adds no behavior of its own: it is the recipe job of Section 6.2. The call above runs exactly this:

```python
from nvflare.recipe import ProdEnv
from nvflare.recipe.catalog import CatalogRecipe

recipe = CatalogRecipe(                            # the catalog read job
    sites=None,                                    # get_catalog(sites=...)
    datasets={"site-1": ["omop"]},                 # get_catalog(datasets=...)
    task_timeout_s=300,                            # get_catalog(timeout_s=...)
)
env = ProdEnv(startup_kit_location="/path/to/admin", study="oncology")  # get_catalog(env, ...)
run = recipe.execute(env)                          # submits the job
results = recipe.get_catalog_results(run, output_path="catalog_results.json")  # waits, downloads, loads
```

| `get_catalog` argument | Recipe equivalent |
| --- | --- |
| `sites`, `datasets`, `site_config_root` | `CatalogRecipe(sites=..., datasets=..., site_config_root=...)` |
| `timeout_s` | `CatalogRecipe(task_timeout_s=...)` |
| `task_request_interval_s` | `CatalogRecipe(task_request_interval_s=...)` |
| `env` | The environment passed to `recipe.execute(env)` |
| `output_path` | `recipe.get_catalog_results(run, output_path=...)` |

The job each call submits is the same `CatalogRecipe` job, named `get_catalog` by default, so it appears in the job list like any recipe job, and its result can be downloaded later from the job store. A user who outgrows `get_catalog`, for example to submit without waiting, moves to these lines without changing what the job does.

```python
def get_catalog(
    env: ExecEnv,                              # where the read runs: any recipe environment
    *,
    sites: list[str] | None = None,            # None: the sites participating at job start
    datasets: dict[str, list[str]] | None = None,  # None: all datasets configured for the Study at that site
    timeout_s: int = 300,                      # per site, as CatalogRecipe.task_timeout_s
    output_path: str | None = None,            # also save catalog_results.json here
    site_config_root: str | None = None,       # SimEnv only: the simulated sites' files
    task_request_interval_s: float = 0.1,      # as CatalogRecipe.task_request_interval_s
) -> dict: ...
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `env` | Required | Where the read runs, and which Study it reads. Any recipe environment works: `SimEnv`, `PocEnv` or `ProdEnv`. For example, `ProdEnv(startup_kit_location=..., study=...)` reads a deployed Study, and `SimEnv(clients=[...])` simulates the sites on one host in the `default` Study, using `site_config_root`. |
| `sites` | the sites participating at job start | An explicit list retains absent sites in the result as `NO_RESULT`. An empty list is an error. |
| `datasets` | all datasets configured for the Study at that site | `{site: [names]}`, as in `CatalogRecipe`. |
| `timeout_s` | 300 | Seconds to wait for each site before recording it as `NO_RESULT`. After the job is submitted, the call waits at most `timeout_s` plus 60 seconds for it to finish. In `SimEnv`, submitting runs the whole simulation before that wait starts, so the limit does not bound simulator startup or shutdown; the per-site timeout still bounds each site's task. |
| `output_path` | none | Also write the result there, ready for `CatalogVisualization`. |
| `site_config_root` | none | Required with `SimEnv` and refused in other environments, as in `CatalogRecipe` (Section 6.2): the folder that holds each simulated site's `local/datasets.yaml` and data. |
| `task_request_interval_s` | 0.1 | How often, in seconds, sites ask the server for work during this job, as in `CatalogRecipe`. |

A completed read returns normally even when some sites failed; each site's status is in `results["sites"]`. `get_catalog` raises `CatalogReadError`, which carries the job ID and status, when the job itself does not finish successfully or the wait expires. In `ProdEnv`, an expired wait does not abort the job.

To submit a read without waiting for it, use `CatalogRecipe.execute` (Section 6.2).

### 6.2 `CatalogRecipe`

`get_catalog` is built on `CatalogRecipe`, a standard Job Recipe. Use the recipe directly to choose the moment to wait, or to submit without waiting.

```python
from nvflare.recipe import ProdEnv
from nvflare.recipe.catalog import CatalogRecipe

recipe = CatalogRecipe(datasets={"site-1": ["omop"]})
run = recipe.execute(ProdEnv(startup_kit_location="/path/to/admin", study="oncology"))
results = recipe.get_catalog_results(run)          # waits, downloads, loads catalog_results.json
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `name` | `get_catalog` | Job name. |
| `sites` | the sites participating at job start | An explicit list retains absent sites in the result as `NO_RESULT`. An empty list is an error. |
| `datasets` | all datasets configured for the Study at that site | `{site: [names]}`. A site reads the requested names present in its Study configuration. A site absent from the mapping reads all its configured Study datasets. If `sites` is explicit, every key in `datasets` must also appear in `sites`. |
| `task_timeout_s` | 300 | Seconds to wait for each site before recording it as `NO_RESULT`. |
| `task_request_interval_s` | 0.1 | How often, in seconds, sites ask the server for work during this job: the server's `task_request_interval` for this job only. The default lets sites notice the end of the run right after returning their results. Must be positive. |
| `site_config_root` | none | `SimEnv` only: the parent folder of one folder per site, each with that site's `local/datasets.yaml` and its data. In `SimEnv` the executor reads that file instead of the job workspace's, the one exception to how jobs load the configuration. A simulated job runs in the `default` Study, so these files declare datasets under `studies: default:`. It is required with `SimEnv` and refused in other environments. |

`get_catalog_results(run, output_path=None, timeout=0)` waits for the job, at most `timeout` seconds when it is positive, downloads its result through the environment, and returns the parsed `catalog_results.json`. With `output_path`, it also saves the file there. It raises `CatalogReadError` when the job did not finish successfully or the file is missing. The same call works in `SimEnv`, `PocEnv` and `ProdEnv`; in `SimEnv` the file is read from the local simulation workspace.

Neither `get_catalog` nor the recipe carries a connection, credential, URI, connector or source type. Those belong to each site's dataset configuration, so the same call works unchanged when sites store the same dataset in Parquet, a SQL database or Unity Catalog.

**Notebook workflow and visualization.** The examples separate acquisition from visualization. `get_catalog.ipynb` calls `get_catalog(..., output_path="catalog_results.json")`; `catalog_explorer.ipynb` loads a saved result. Following the federated statistics visualization pattern, `CatalogVisualization("catalog_results.json")` provides dataset and source overviews, table inventories, schemas, table and column comparisons, relationships and per-site status. It identifies each source by `(site, dataset)`, supports several datasets at one site, and distinguishes unavailable, skipped and unknown values. It opens no federation or database connection, so it works offline.

## 7. Semantics and later work

**Semantic boundaries.** Names and types are structural evidence, not proof of scientific equivalence. An operation still defines units, terminology, observation windows and subject linkage. The catalog may advertise an operation only when the connector reports it and the site's installed capabilities include it; advertising authorizes nothing.

**Declared standards.** The metadata model has fields for a declared standard, release, vocabulary and mapping version. The built-in connectors leave them null, because no source they read reports them, and the catalog never guesses a standard from table names. Standards validation, such as checking an OMOP 5.4 schema, is a separate approved quality operation that names the specification it checked.

**Handoff to operations.** A user who selects a dataset passes its site and name to a separately admitted operation in the same Study. That operation resolves the dataset through the same dataset configuration API and revalidates its structure and permissions against live data. The catalog is a discovery view, not an access grant.

**Derived datasets.** When data preparation is enabled, committed outputs can appear as datasets through the dataset configuration API, with lineage to their source. They then need no change in the catalog.

## 8. Delivery and validation

Required checks:

- Read datasets only from the dataset configuration snapshot of the job's Study; a request cannot add a dataset, path, connector or credential. A missing configuration file reads as no datasets; an invalid one fails the read with `CONFIG_UNAVAILABLE`.
- Corrupt dataset B among A, B and C: A and C are returned, B is `unavailable` with no partial schema. Exhaust a dataset budget and the whole-read budget separately; budget-skipped datasets appear in `skipped`.
- Read all configured Study datasets without per-dataset rules; preserve source-reported tables, columns, keys, resources, metadata and lineage.
- Reject obsolete per-dataset publication rules instead of silently broadening disclosure. Keep site-wide settings and installed-operation filtering.
- Qualify source metadata visibility using the intended database identity/projection; a revoked or unreadable source is unavailable on the next read, without reusing an older schema.
- Two sites with a dataset of the same name stay separate entries. A site that times out is `NO_RESULT` and the other sites' results are complete.
- The controller rejects a malformed or over-budget site entry and keeps the site's other entries.
- Validate every metadata example against Section 3.4: unknown keys, duplicate or unresolved columns, source types reported unchanged, future-dated or malformed observations stored as null.
- `get_catalog_results` and `get_catalog` return the same structure in `SimEnv`, `PocEnv` and `ProdEnv`; `get_catalog` returns normally when one site is `NO_RESULT`, and raises `CatalogReadError` when the job fails or the wait expires.
- In `ProdEnv`, run two reads of the same Study at once: both complete with independent results.
- Benchmark job startup and end-to-end read time with representative sites before making latency claims.

## 9. Alternatives and trade-offs

| Area | Chosen | Set aside | Trade-off |
| --- | --- | --- | --- |
| Where the catalog lives | Returned to the submitter as the job's result; stored with the job in the job store | A server-side catalog store, merged across reads and searched in place | No reconciliation, persistence, locking, recovery or result-trust machinery on the server. The cost is that each user gets the catalog by running a read; there is no shared, always-on index to search, and an offline site is simply missing from that read. |
| Current state | The latest read is the complete answer | Merging reads, with versions and availability history | Nothing to keep consistent across reads. The cost is no catalog or schema version counters; comparing two reads is the user's choice. |
| Entry point | `get_catalog` for one-call reads, built on `CatalogRecipe` | A dedicated command-line tool | One Python surface, consistent with other recipes, with a single call for the common case. The cost is that a read needs Python, not a shell command. |
| Site configuration | The generic read-only [dataset configuration API][dataset-config] | Catalog-specific configuration files | A dataset is declared once for every component that uses it, and the catalog never opens configuration files. The cost is a new site file for operators to maintain. |
| How a read runs | A standard FLARE job per read | The server calling sites directly | The job reuses authentication, task delivery, launchers, abort, logs and result download. It costs one job per read in the job store. |
| What a read returns | Full bounded metadata per selected dataset | Incremental reads with checkpoints | No checkpoint protocol. The cost is more source reads, so incremental reads wait for measured cost. |
| Publication control | Source-admin credentials, namespaces and readable files; every configured Study dataset is eligible | Catalog-specific share and exclusion rules | One source-owned visibility boundary. Administrators must qualify schema visibility as well as row access; downloaded snapshots remain historical artifacts. |
| Unity Catalog, SQL and DuckDB | REST API, SQLAlchemy reflection, DuckDB's own catalog views | Spark or a vendor SDK | No heavy dependency at the site. The cost is that only what these interfaces expose can be cataloged. |
| Where connectors are chosen | By the site's dataset configuration | A connector option in the recipe | A request-supplied connector would let a job reach a source the site did not approve. The cost is that adding a backend, or custom reading logic, needs a site operator. |

[dataset-config]: https://github.com/NVIDIA/NVFlare/blob/0bca2a8efe11392df1561653d7cd9b654f499c8a/docs/design/dataset_configuration_design.md
