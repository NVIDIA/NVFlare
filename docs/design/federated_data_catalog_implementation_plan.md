# Federated Data Catalog Implementation Plan

Engineering plan for the [Federated Data Cataloging Design](federated_data_cataloging_design.md): modules, classes, signatures, algorithms and storage. Section references (§) point to that document. Anything marked **[verify]** is an assumption about existing FLARE code that must be checked before it is coded against (list in Section 12).

## 1. Existing code reused

| Need | Existing code |
| --- | --- |
| Dataset bindings, mounts | `nvflare/app_opt/job_launcher/study_data.py`: `StudyDatasetMount`, `DATA_MOUNT_ROOT = "/data"`, mounts at `/data/<study>/<dataset>` |
| Study of a job, client name in a job | `get_job_meta_study(meta)` (`nvflare/apis/job_def.py`), `FLContextKey.JOB_META`, `FLContextKey.CLIENT_NAME` |
| Study membership | `StudyRegistryService.get_registry().get_sites(study)`, `has_user(user, study)` |
| Admin commands | `CommandModule`, `CommandSpec`, `CommandModuleSpec`, `ConnProps` (`USER_NAME`, `USER_ROLE`, `ACTIVE_STUDY`), registered in `ServerCommandModules` (`server_cmd_modules.py`); reply with `conn.append_dict(payload, meta=make_meta(...))`, as in `study_cmds.py` |
| Job definition | `Recipe` / `FedJob` (`nvflare/recipe/spec.py`, `nvflare/job_config/api.py`), `FedJob(meta_props=...)`; any `ExecEnv` (`SimEnv`, `PocEnv`, `ProdEnv(study=...)`) runs it. Server-side submission: `JobDefManagerSpec.create(meta, uploaded_content, fl_ctx)`, `get_job`, `delete`, `get_storage_component`; meta keys in `JobMetaKey` |
| Job end notification in the server process | `EventType.JOB_COMPLETED` / `JOB_ABORTED` fired by `job_runner.py` |
| Server component registration | `components` list of the server `resources.json` (`master_template.yml` shows `job_scheduler`, `job_manager`) |
| Client-side task | `Executor.execute(task_name, shareable, fl_ctx, abort_signal)` |
| Server-side task | `Controller` / `ControllerSpec.send_and_wait`, `broadcast_and_wait` |
| Component authorization | `DEFAULT_CLASS_ALLOW_LIST` (`nvflare/app_common/default_component_policy.py`) |
| CLI plumbing | `nvflare/tool/cli_session.py`, `cli_output.py`, `cli_errors.py`, `cli_schema.py`, `job/job_cli.py` |

The server has no database today and this design adds none. The store (Section 7) keeps state in plain JSON files under the server workspace, written with atomic replace, and builds its search index in memory at startup.

## 2. Package layout

```
nvflare/recipe/catalog.py   # CatalogRecipe: the standard Job Recipe of a catalog read
nvflare/app_common/schedule/
  periodic_job_submitter.py # PeriodicJobSubmitter (FLComponent): generic, submits a recipe's job on a schedule
nvflare/app_common/fdn/catalog/
  common/          # wire formats shared by every side; imports nothing else from the package
    schema.py      # CatalogInfo, ReadLimits, v1 validator
    payload.py     # catalog_read task/result builders and parsers, manifest format, ResultStatus, ReasonCode, TASK_NAME, MANIFEST_FILE, CATALOG_JOB_META
  site/            # runs at the site; imports only common/
    executor.py    # CatalogExecutor (Executor), MountResolver
    policy.py      # StudyPolicy, DatasetRule, load_policy, apply_policy
    connector.py   # CatalogConnector ABC, ConnectorError
    site_config.py # SiteConfig, ConnectorSpec, load_site_config: the site's connectors, limits and operations
  server/          # runs in the server process or the server-side job
    controller.py  # CatalogController (Controller), manifest writing
    fileio.py      # atomic JSON file replacement
    artifacts.py   # private result inbox, signature checks and legacy archive redaction
    store.py       # CatalogStore (JSON files, in-memory state)
    reconcile.py   # Reconciler
    manager.py     # CatalogManager (FLComponent): applies finished jobs, answers queries
    search.py      # SearchQuery, CatalogEntry, run_search
    commands.py    # CatalogCommandModule
  admin/
    catalog_api.py # CatalogAdminApi, the admin-side caller used by the CLI
nvflare/app_opt/fdn/catalog/
  parquet_connector.py  # ParquetConnector (pyarrow, optional)
tests/unit_test/app_common/fdn/catalog/   # mirrors the layout above; site/layout_test.py enforces the import rule
tests/unit_test/app_common/schedule/, tests/unit_test/recipe/catalog_recipe_test.py, tests/unit_test/app_opt/fdn/catalog/
```

The site folder is the trust boundary: a site owner reviews `site/` and `common/` and nothing else, and `site/layout_test.py` fails if either imports server or admin code. `site/` also imports the Study runtime parser (`nvflare/app_opt/job_launcher/study_runtime.py`) to find process-mode datasets. `admin/` reuses the command names and result types of `server/commands.py` and `server/search.py`, and `TASK_NAME`, `MANIFEST_FILE` and `CATALOG_JOB_META` live in `common/payload.py`.

`schema.py`, `payload.py`, `policy.py` and `search.py` have no FLARE imports, and `schema.py`, `payload.py` and `policy.py` do no I/O beyond `policy.py` reading one file, so they run in unit tests and inside the job unchanged.

## 3. Contract layer (`schema.py`, `payload.py`)

**Typed model for code, plain dict on the wire.** Connector authors, the policy applier, the server and the end-user client work with typed, frozen dataclasses. The `Shareable` and the stored JSON carry only the v1 dictionary, so no custom serializer is needed. Conversion happens at two boundaries: `CatalogInfo.to_wire()` when the handler builds the result, and `CatalogInfo.from_wire()` when the server parses it (and when the client parses a search reply).

```python
@dataclass(frozen=True)
class ColumnInfo:    name: str; type: str; nullable: bool | None = None
@dataclass(frozen=True)
class Relationship:  columns: tuple[str, ...]; target_table: str; target_columns: tuple[str, ...]
@dataclass(frozen=True)
class TableInfo:     name: str; columns: tuple[ColumnInfo, ...]; primary_key: tuple[str, ...] = (); relationships: tuple[Relationship, ...] = ()
@dataclass(frozen=True)
class Resource:      type: str; supported_search: tuple[str, ...] = ()
@dataclass(frozen=True)
class SemanticProfile:
    standard: str | None = None; version: str | None = None; vocabulary_version: str | None = None
    mapping_version: str | None = None; declaration_ref: str | None = None; validation_ref: str | None = None
@dataclass(frozen=True)
class Lineage:       dataset: str; source_version: str | None; relation: str = "derived_from"
@dataclass(frozen=True)
class Provenance:    source: str; declaration_ref: str | None = None; validation_ref: str | None = None

@dataclass(frozen=True)
class ReadLimits:
    max_duration_s: int; max_calls: int; max_records: int

@dataclass(frozen=True)
class CatalogInfo:                    # the catalog of one dataset: one dataset, one CatalogInfo (§3.3)
    dataset: str                  # existing Study dataset name
    display_name: str
    tables: tuple[TableInfo, ...] = ()
    resources: tuple[Resource, ...] = ()
    semantic_profile: SemanticProfile = SemanticProfile()
    modalities: tuple[str, ...] = ()
    operations: tuple[str, ...] = ()
    description: str | None = None
    owner: str | None = None
    source_version: str | None = None
    lineage: tuple[Lineage, ...] = ()
    provenance: Provenance                    # source is the connector's own label; no default
    observed_at: str | None = None            # RFC 3339 UTC, from the source; None = unknown
    def to_wire(self) -> dict: ...            # {"dataset", "metadata": {...v1...}, "observed_at"}
    def metadata_wire(self) -> dict: ...      # only the v1 metadata object (what the server stores)
    @classmethod
    def from_wire(cls, d: dict) -> "CatalogInfo": ...   # rejects unknown keys and wrong types
    def digest(self) -> str: ...              # over metadata_wire() only, never observed_at
    def record_count(self) -> int: ...

class ValidationError(ValueError): ...

def validate(catalog: CatalogInfo, limits: ReadLimits) -> None: ...
```

- `from_wire` rejects unknown keys and wrong types while building the objects. `validate` enforces §3.3: string bounds (256, descriptions 4096), unique names per list, table/column references resolving inside the entry, relationship column-list lengths equal and nonzero, `record_count() <= max_records`. Both raise `ValidationError` with a safe message (no values from the source).
- Nothing is sorted, renamed or retyped: the catalog reports the site's lists, names, types and case as given. `digest()` is SHA-256 of the canonical JSON of `metadata_wire()` (object keys sorted and no whitespace, which fixes serialization only, never list order), so identical reported content gives an identical digest and any difference, including a different list order, counts as a change.

`payload.py`:

```python
class ResultStatus(str, Enum): OK="OK"; FAILED="FAILED"
class ReasonCode(str, Enum):
    POLICY_UNAVAILABLE; BUDGET_EXCEEDED; CANCELLED; INTERNAL

CATALOG_READ_KEY = "catalog_read"; FORMAT_VERSION = 1
TASK_NAME = "catalog.read"; MANIFEST_FILE = "catalog_results.json"; CATALOG_JOB_META = "fdn_catalog_read"
CONTEXT_FILE = "catalog_context.json"   # written by the manager in the job run dir before launch, read by the controller

def build_task_payload(datasets: list[str] | None, previous: list[str]) -> dict
def parse_task_payload(d: dict) -> TaskPayload              # raises PayloadError
def build_result(status, catalogs=(), withheld=(), unavailable=(), reason_code=None) -> dict
def build_manifest(entries: dict[str, ManifestEntry]) -> dict   # {"results": {client: {"rc", "result", "datasets": [...] | null}}}
def parse_manifest(d: dict) -> dict[str, ManifestEntry]       # each entry carries the datasets asked of that client
def job_key(master: bytes, job_id: str) -> bytes               # HMAC-SHA256(master, "job\n" + job_id): the key of one job
def manifest_mac(key: bytes, job_id: str, manifest: dict) -> str   # HMAC-SHA256 over the job id and the canonical manifest
def parse_result(d: dict, selection: Selection) -> ParsedResult   # Selection(datasets): the dataset names the job filters on; each site reads those it has; None = all
```

`parse_result` builds each catalog with `CatalogInfo.from_wire` (it checks keys and types, not bounds; the Reconciler validates each parsed catalog with the manager's `ReadLimits` before storing it, §8) and is deliberately tolerant per section, strict within a section: it extracts the `withheld` list first (valid names are those in the job's selection; others are dropped), then `unavailable` (same rule), then each catalog independently into `ParsedResult.catalogs` (typed `CatalogInfo`). A malformed named catalog is recorded in the internal `ParsedResult.rejected` list and never raises; the restriction lists are returned either way. Duplicate catalog names are also recorded as rejected. Rejected names remain bounded by the recorded selection and are not an additional wire field. A name appearing in more than one list is resolved in favor of `withheld`, then `unavailable`, and the catalog entry for it is dropped; a dataset that appears in more than one catalog entry has every copy dropped. `catalogs` count only when `status` is `OK`. Checked `unavailable` names count for both statuses, and `withheld` always applies. An unrecognized `status` is read as `FAILED`. `format_version` must be the integer 1 (`true` and `1.0` are not accepted). `ParsedResult.status` and `reason_code` come from the top level; an unknown or non-integer `format_version` returns an empty result flagged `unsupported`. An explicit empty `datasets` list in a task is rejected (omit it to select all).

## 4. Policy (`policy.py`)

```python
@dataclass(frozen=True)
class DatasetRule:
    share: bool
    exclude_tables: frozenset[str] = frozenset()
    exclude_columns: Mapping[str, frozenset[str]] = field(default_factory=dict)
    exclude_resources: frozenset[str] = frozenset()
    exclude_metadata: frozenset[str] = frozenset()
    operations: frozenset[str] = frozenset()

@dataclass(frozen=True)
class StudyPolicy:
    datasets: Mapping[str, DatasetRule]

class PolicyError(Exception): ...

def load_policy(path: Path) -> StudyPolicy
def apply_policy(rule: DatasetRule, catalog: CatalogInfo, installed_ops: set[str]) -> CatalogInfo
```

- `load_policy` reads the file once per call (no caching: replacements are atomic and must be seen on the next read), parses JSON and validates against the §3.1 table: `format_version == 1`; allowed top-level keys `format_version, datasets`; allowed rule keys exactly the six in the table; `share` required boolean; `exclude_metadata` a subset of the seven allowed names; strings and lists of the right types. Any violation raises `PolicyError`. Missing or unreadable file also raises `PolicyError`. Never a fallback.
- A dataset with no entry in `StudyPolicy.datasets` has no rule and is withheld. A `derived` rule for committed outputs belongs to the later derived-dataset reader and is not accepted yet.
- `apply_policy` runs in this order:
  1. Remove tables in `exclude_tables`; remove columns in `exclude_columns[table]`; remove resources in `exclude_resources`. Record the removed `(table, column)` and table names.
  2. For each remaining table: if any primary-key column was removed set `primary_key = []`; drop relationships whose source columns include a removed column or whose `target_table`/`target_columns` were removed (only those affected by exclusions; other broken references stay and fail validation later).
  3. For each name in `exclude_metadata` set the field to `None` (nullable fields) or `[]` (`modalities`, `lineage`), `semantic_profile` to an empty profile, and `provenance` to its source alone (`source` is required, so only the two references become `None`). `display_name` cannot be excluded; if it would be unusable, the caller withholds the dataset.
  4. `operations = sorted(set(catalog.operations) & rule.operations & installed_ops)`.
  Returns a new frozen `CatalogInfo` and never mutates its input; the caller validates it with `validate`.

## 5. Site side

### 5.1 `CatalogConnector` (`connector.py`)

```python
class ConnectorError(Exception):
    """Safe, dataset-scoped failure. Message must contain no paths or source text."""
    def __init__(self, safe_message: str, *, budget: bool = False): ...

class CatalogConnector(ABC):
    source: str     # what the connector connects to ("parquet", "duckdb", "catalog-export"); a safe label, never a path or endpoint
    def __init__(self, datasets: dict[str, Path]): self._datasets = dict(datasets)   # no I/O; the Study's mounts, empty if none
    @abstractmethod
    def read_catalog(self, dataset: str, limits: ReadLimits, options: dict | None = None) -> CatalogInfo: ...   # options: the dataset's non-secret settings from catalog_site.json
    def close(self) -> None: ...
```

**Plugin mechanism.** A connector is any class that subclasses `CatalogConnector`. There is no registry. The executor loads each connector from its `class`, a dotted class path under `connectors.<name>` in the site's own `catalog_site.json` (Section 6.3), never from the job, and constructs it once per read as `cls(mounts, **args)`, where `mounts` is the Study's mounted datasets and `args` are that connector's non-secret constants in the site file. A class that does not subclass `CatalogConnector` is refused. The class needs only to be importable in the job's Python environment, so a built-in, an optional `app_opt` module or a site's own package are loaded the same way. Rules every connector follows:

**Several connectors.** A Study has several datasets and they can come from different kinds of source: Parquet files, a SQL database, Databricks. So a connector is defined once, without datasets, and each dataset is then associated with a connector by name (`catalog_site.json`, Section 6.3). The executor builds only the connectors the selected, policy-shared datasets use and reads each dataset through its own, passing the dataset's non-secret `options` (a schema or table list) to `read_catalog`. A mounted dataset with no binding uses the site's default connector, which is the built-in Parquet connector unless the file says otherwise, so a Parquet-only site needs no connector section. A binding for a dataset that is also mounted wins. A remote dataset needs no mount and no entry in the Study runtime: the source's own access control, not the launcher's mounts, isolates the Study there, and the Study's `secret_env` in `study_runtime.yaml` supplies the identity. A connector that cannot be built costs only its own datasets: the read still returns the others, with reason `INTERNAL` (health `DEGRADED`), and the datasets it served are reported unavailable (hidden). If every connector fails, the read is `FAILED`/`INTERNAL` and every indexed selected name is withheld, as before.

- The constructor performs no I/O; `read_catalog` is called once per selected dataset and `close` once at the end.
- It reports one complete `CatalogInfo` or raises `ConnectorError`; it never returns a partial schema, a path, an endpoint or a credential, and it never writes to the source.
- It enforces `limits` while reading and sets `Provenance(source=self.source, ...)`.
- Optional third-party packages are imported lazily inside the connector, so a missing package fails that connector's load (the job returns `FAILED`/`INTERNAL` without detail), not the catalog package import.
- Anything site-specific comes from the site's own configuration, and secrets come from the Study runtime's secret environment or mounts, never from `connector_args`, the task or the result.

`ConnectorError(budget=True)` means a budget/cancel stop (no evidence about the source); `budget=False` means the source is absent, inaccessible or invalid.

### 5.2 `ParquetConnector` (`app_opt`)

Declares `source = "parquet"`. Every connector builds its catalogs with `Provenance(source=self.source, ...)`; the handler does not set or override it. A new reader chooses its own label, so no format version change is needed.

`read_catalog(dataset, limits, options=None)` (the Parquet connector takes no options):
1. `root = self._datasets[dataset]`; raise `ConnectorError` if missing or not a directory/file.
2. A file binding is read regardless of its mount alias extension. For a directory binding, enumerate root `*.parquet` files and recursively discover `*.parquet` parts within each top-level table directory in name order (the mount's listing order is not stable; the walk of a partitioned table stops with a budget outcome once it has found `max_calls` parts or passed the deadline, so a deep tree cannot overrun the budget while it is listed); one table per file stem or per subdirectory (a directory of parts is one table, taking the schema of its first part and verifying every other part with `pq.read_schema`; a dataset with more parts than `max_calls` fails with a budget outcome rather than publishing a schema that was not fully verified).
3. For each table read `pq.read_schema` / footer only (never row groups). Report each column's type as Arrow prints it (`str(field.type)`, for example `int64`, `string`, `timestamp[us]`), with no mapping. `nullable` from the Arrow field.
4. Optional sidecar `catalog.json` in `root` (site-authored, same v1 field names) supplies `display_name, description, owner, modalities, operations, semantic_profile, source_version, lineage, provenance, primary_key, relationships`; read-only, bounded to 1 MB, validated like any other metadata. The sidecar is an object with any of those fields (plus `observed_at`); `tables` is an object keyed by table name whose values carry `primary_key` and/or `relationships`, and `provenance` may carry only `declaration_ref` and `validation_ref` (the source label is always the connector's). Unknown keys fail. The connector merges the sidecar with the Arrow-derived tables into a full wire dict and builds `CatalogInfo` through `from_wire`, so types are checked. Without it `display_name = dataset`, other fields null/empty and `provenance.source` is the connector's `source`.
5. Check the deadline and `record_count()` against `limits` while building; exceeding raises `ConnectorError(budget=True)`; a corrupt footer raises `ConnectorError`.
6. `observed_at` from the sidecar only; never from file mtime.

Partition values and scientific semantics are not inferred.

### 5.2a Later connectors: Databricks Unity Catalog and Snowflake

Not part of the mounted-data pilot (design Section 3.2 and parent Section 12 require qualification of each site's bindings and credentials). They are in-job connectors that follow the existing credential pattern, add no new mechanism, and plug in the same way as `ParquetConnector`: a named entry under `connectors` in `catalog_site.json`, with each of their datasets bound to it under `studies`:

- `UnityCatalogConnector` (`source = "unity-catalog"`) and `SnowflakeConnector` (`source = "snowflake"`), optional `app_opt` modules that import `databricks-sdk` or the Snowflake client lazily.
- **Credentials.** The site creates the Secret ahead of time and lists it for the Study in `local/study_runtime.yaml` (`secret_env`, and `secret_mounts` for a key file). The launcher only adds references to the pod spec. The kubelet (Kubernetes) or the launcher's environment (Docker) supplies the values, so the connector reads them from its environment like any other job code. Non-secret settings such as the workspace host go in the Study's `env`.

```yaml
studies:
  lung-cancer:
    env:
      DATABRICKS_HOST: https://adb-123.azuredatabricks.net
    secret_env:
      DATABRICKS_CLIENT_ID:     {source: study-databricks, key: client_id}
      DATABRICKS_CLIENT_SECRET: {source: study-databricks, key: client_secret}
```

- **Rules.** Credentials never appear in `connector_args`, the task, the policy file, the result, logs or error messages. The connector raises `ConnectorError` with a fixed safe message on any authentication or network failure. The principal behind the secret holds metadata-only grants (catalog and schema visibility, no data access) [verify the exact grant names]. The job needs network egress to the workspace or account, which the site must allow.
- **Known limitation.** The job container can read these values, and the `secretKeyRef` entries are visible in the pod spec to anyone with `pods/get`. `study_runtime_config.md` reserves a launcher-side Databricks provider that keeps its token out of the pod; if it is built, the metadata read can move there without changing `CatalogInfo` or the handler.

**Remote connectors.** The open item is closed by `catalog_site.json`: it names the connectors and binds each remote dataset to one with its non-secret options. The `CatalogConnector` interface gains only the `options` argument of `read_catalog`.

Mapping from the source to `CatalogInfo`:

| `CatalogInfo` field | Unity Catalog | Snowflake |
| --- | --- | --- |
| `dataset` | Site-mapped name for one UC schema (or an explicit table list) | Site-mapped name for one schema |
| `tables[].name`, `columns[].name`, `nullable` | `TableInfo.name`, `ColumnInfo.name`, `nullable` | `INFORMATION_SCHEMA.TABLES` and `COLUMNS` |
| `columns[].type` | `ColumnInfo.type_text` unchanged | `DATA_TYPE` as Snowflake reports it, with precision and scale rendered the way Snowflake shows them [verify] |
| `primary_key`, `relationships` | `TableInfo.table_constraints` (primary key, foreign key) | Declared constraints (`TABLE_CONSTRAINTS`, key listings) [verify] |
| `description`, `owner` | `comment`, approved steward alias | `COMMENT`, approved steward alias |
| `source_version` | Null unless the source states an opaque revision [verify] | Null unless the source states an opaque revision [verify] |
| `observed_at` | Source-reported update time (`updated_at`), never read time [verify] | Source-reported `LAST_ALTERED`, never read time [verify] |
| `semantic_profile`, `modalities`, `operations`, `lineage` | Site declaration sidecar, as for Parquet | Site declaration sidecar, as for Parquet |

Dropped on purpose, as in the Unity Catalog comparison: storage locations, connection and share names, `properties`, view SQL, row filters and masks, ownership identities, timestamps other than `observed_at`. `max_calls` counts API calls or queries, and `max_records` counts the objects built, as for Parquet.

### 5.3 `CatalogExecutor(Executor)` (`executor.py`)

```python
class CatalogExecutor(Executor):
    def __init__(self): ...   # no arguments: a job cannot configure the executor
    def execute(self, task_name, shareable, fl_ctx, abort_signal) -> Shareable: ...
```

The site, not the job, decides everything about the read. `CatalogExecutor()` takes no constructor arguments, so a job config cannot set a connector, a policy location or limits (a site's component allow-list sees only the class name, never its arguments). The executor reads `<policy_root>/catalog_site.json` with `format_version` 1, `connectors` and per-Study dataset bindings (below), `limits` (`max_duration_s`, `max_calls`, `max_records`) and `installed_operations`. Secrets never appear in it; connectors read them from the Study runtime's environment. A missing or invalid site file is handled like a missing policy file: nothing is shared and previously indexed names are withheld. **Locations by launcher.** `policy_root` is `/fdn/policy` when that directory exists, which is how Docker, Kubernetes and Slurm container modes expose it: the site adds a read-only `secret_mounts` entry for the Study in `local/study_runtime.yaml` (a host directory for Docker and Slurm, a Kubernetes Secret for Kubernetes). Otherwise, as in process mode, it is `<workspace>/local/fdn/`. The same two files are used either way. Docker binds the whole `local/` folder read-only into the job, so the `local/fdn/` fallback also works there without the `secret_mounts` entry. Two small launcher-side changes: the strict Study-runtime parser accepts `datasets` in process mode (only the catalog executor reads it; the process launcher mounts nothing), and `workspace_cell_transfer.py` excludes `local/fdn/` from Kubernetes workspace bundles.

Example `<policy_root>/catalog_site.json` (`/fdn/policy/` in container modes, `<workspace>/local/fdn/` in process mode). Every key but `format_version` is optional; `{"format_version": 1}` alone means every mounted dataset is read by the built-in Parquet connector, with the default limits and no installed operations. A site whose Studies mix Parquet files, a SQL database and Databricks defines each connector once and associates each dataset with one by name:

```json
{
  "format_version": 1,
  "connectors": {
    "claims-db": {"class": "<a SqlConnector>", "args": {"host": "db.hospital.org", "database": "claims"}},
    "lake":      {"class": "<a DatabricksConnector>", "args": {"workspace": "https://adb-1234.azuredatabricks.net"}}
  },
  "studies": {
    "oncology": {
      "claims":       {"connector": "claims-db", "options": {"schema": "public"}},
      "imaging-meta": {"connector": "lake",      "options": {"catalog": "onc", "schema": "imaging"}}
    },
    "cardiology": {
      "echo":         {"connector": "lake",      "options": {"catalog": "cardio", "schema": "echo"}}
    }
  },
  "limits": {"max_duration_s": 60, "max_calls": 1000, "max_records": 10000},
  "installed_operations": ["cohort.count", "cohort.sum"]
}
```

- **`connectors`:** named, each with a `class` (a `CatalogConnector` subclass) and optional non-secret `args`. The connector `parquet` is built in and always exists; a file may redefine it. Names are letters, digits, `.`, `_` and `-`. A connector says nothing about Studies or datasets.
- **`studies`:** per Study, the datasets that are not simply mounted Parquet, each with a `connector` (a defined name) and optional non-secret `options`. A dataset name follows the Study dataset-name rule. A mounted dataset that is not listed is read by `default_connector` (default `parquet`). A dataset that is listed is read by its connector, whether or not it is mounted.
- **`limits` and `installed_operations`:** apply to the whole site. Each `limits` value is a positive integer and an omitted key keeps the default. Operation names are free-form strings the site chooses, which must match what the connector reports and the Study policy allows.
- **Errors:** an unknown key, a wrong type, an undefined connector name or another `format_version` makes the file invalid, and the site then shares nothing. Credentials never appear in the file.

**Data locations.** Container launchers mount each `datasets` entry at `/data/<study>/<dataset>` and the executor finds them there. Process mode mounts nothing, so the `datasets` key of `local/study_runtime.yaml` is accepted for process mode (the process launcher ignores it) and the executor reads each entry's `source` as the real path, which process mode requires to be absolute (a relative one would resolve against the job's working directory), so a site keeps its data anywhere and declares each dataset once. Slurm sandbox `none` rejects `datasets` and `secret_mounts` in the launcher, so catalog reads are not supported there yet. Tests set `policy_root` and `resolver` on the instance.

`MountResolver.resolve(study, runtime_file=None) -> dict[str, Path]` lists the immediate files and subdirectories of `<mount_root>/<study>/` whose names match the dataset-name pattern used by `study_data.py`; that is the set of current bindings. When that directory does not exist (process mode) and `runtime_file` is the workspace's `local/study_runtime.yaml`, it returns the Study's `datasets` sources that are existing files or directories. An unreadable runtime file gives no bindings.

`execute` (task name `catalog.read`; any other name returns `ReturnCode.TASK_UNKNOWN`):

```
study  = get_job_meta_study(fl_ctx.get_prop(FLContextKey.JOB_META))
task   = parse_task_payload(shareable[CATALOG_READ_KEY])            # PayloadError -> FAILED/INTERNAL
# Every failure before the per-dataset reads shares nothing and withholds every selected name the server already holds
# (previous): the permitted view cannot be rebuilt, so a narrowed policy cannot be applied to what the server has.
asked = task.datasets if task.datasets is not None else sorted(task.previous)   # the names a failure before listing can speak about
try:    mounts = MountResolver.resolve(study)
except OSError:  return result(FAILED, reason=INTERNAL, withheld=set(task.previous) & set(asked))
try:    site = load_site_config(policy_root/"catalog_site.json"); policy = load_policy(policy_root/f"{study}.json")
except (PolicyError, SiteConfigError):
    return result(FAILED, reason=POLICY_UNAVAILABLE, withheld=set(task.previous) & set(asked))   # nothing else
deadline = monotonic() + site.limits.max_duration_s
withheld, unavailable, catalogs = [], [], []
routes = {n: Binding(site.default_connector) for n in mounts} | site.bindings(study)   # the Study's datasets: mounted ones, plus those the file binds
selected = task.datasets if task.datasets is not None else sorted(set(routes) | set(task.previous))
needed = {routes[n].connector for n in selected if n in routes and policy.shares(n)}
connectors = {c: build(site.connectors[c])(mounts, **site.connectors[c].args) for c in needed}   # each must subclass CatalogConnector; a build failure records INTERNAL; if needed but none built -> FAILED/INTERNAL, withheld = previous & selected
for name in selected:
    rule = policy.datasets.get(name)
    if rule is None or not rule.share:   # only a previously indexed name is reported; naming a never-shared mount would disclose it
        if name in task.previous: withheld.append(name)
        continue
    if abort_signal.triggered or monotonic() >= deadline or budget_exhausted():
        if name in task.previous: withheld.append(name)       # skipped, previously indexed
        reason = BUDGET_EXCEEDED or CANCELLED; continue
    route = routes.get(name); connector = connectors[route.connector] if route else None
    if connector is None:   unavailable.append(name); continue      # no connector reads it now, or its connector is down
    try:
        cat = connector.read_catalog(name, per_dataset_limits(remaining=deadline-now, records=max_records-used_records), route.options)
        cat = apply_policy(rule, cat, installed_ops)
        require cat.dataset == name; record count within the remaining records (else budget)
        validate(cat, limits)
        catalogs.append(cat)
    except ConnectorError as e:
        if not e.budget:   unavailable.append(name)
        elif name in task.previous: withheld.append(name)          # budget skip of an indexed name
    except ValidationError:  unavailable.append(name)
    except Exception:        log locally; unavailable.append(name)     # safe, no detail returned
for connector in connectors.values(): connector.close()   # built ones only
status = OK if catalogs or unavailable or withheld or reason is None else FAILED
return result(status, catalogs, withheld, unavailable, reason)
```

Rules the code must keep: every selected, previously indexed name ends in exactly one list; the three lists are disjoint (assert before returning); no exception text, path or source detail enters the `Shareable`. The return code is always `ReturnCode.OK` for handled outcomes.

A completed policy-only read returns `OK` even when it publishes no datasets and has no previously indexed names to withhold. Only connectors needed for selected, policy-shared datasets are opened. A budget/cancellation stop with no usable dataset outcome remains `FAILED`.

## 6. Server side

### 6.1 `CatalogRecipe` and `CatalogController`

A catalog read is an ordinary job, defined by a Job Recipe like every other FLARE example and run with `python job.py` through any execution environment. The Study comes from the environment (for example `ProdEnv(study="oncology")`), not from the recipe.

```python
class CatalogRecipe(Recipe):
    def __init__(self, name="catalog_read", sites: list[str] | None = None,
                 datasets: dict[str, list[str]] | None = None, task_timeout_s: int = 300): ...
```

`sites=None` deploys to every site of the Study (`@ALL`); a list deploys to those sites only. `datasets` maps a site to the dataset names to read there: the site reads the named datasets it has and skips the others (a name a site never shared is not reported, so no site learns another's names). A site that is not in the mapping, or `datasets=None`, reads everything its policy shares. Sites have their own datasets, so the names are per site, and a site named in `datasets` must be read (a name in `datasets` that is not in `sites` is refused). `min_clients` is 1, so a site that is offline ends as `TIMEOUT` and does not fail the others. The recipe adds `CatalogController(datasets, task_timeout_s)` to the server and `CatalogExecutor()` (no arguments) for task `catalog.read` to the clients, and sets the job meta property `fdn_catalog_read` (`CATALOG_JOB_META`) so the server can recognize the job. It has no connector, credential, limit or policy argument: each site decides those from its own files. `CatalogController` and `CatalogExecutor` are in `DEFAULT_CLASS_ALLOW_LIST`, so non-BYOC jobs may use them under the default component authorization. A catalog job is one job per Study covering all its sites, not one per site.

```python
class CatalogController(Controller):
    def __init__(self, datasets: dict[str, list[str]] | None = None, task_timeout_s: int = 300): ...   # validated; kept as attributes
    def start_controller(self, fl_ctx): study from the job meta; server workspace root; manifest path under the run dir; the job's clients
    def control_flow(self, abort_signal, fl_ctx):
        for each client: previous = names from catalog_context.json, narrowed to `datasets`
                         broadcast Task("catalog.read", Shareable{catalog_read: payload}, timeout,
                                        result_received_cb=self._record, task_done_cb=<keyword-argument callback>)
        wait for every task to finish (or cancel them on abort), then write the manifest atomically
    def _record(self, client_task, fl_ctx):
        check JSON serialization per site; isolate an invalid result as FAILED/INTERNAL, preserving valid restrictions
        store {rc, result} under the framework's client name; never trust payload identity
    def stop_controller(self, fl_ctx): write whatever has arrived
```

`previous` (the names the server already holds for a site, which the site needs to withhold fail-closed) is not read from the store by the job: the manager puts it in `catalog_context.json` in the job's run directory before the job starts (see the handoff in this section), for the clients of the job's own Study only, so the job process never touches the server's catalog files.

The framework calls both callbacks with keyword arguments, so their parameter names must be `client_task, fl_ctx` and `task, fl_ctx`.

`catalog_results.json` holds `{"manifest": {"results": {client: {"rc", "result", "datasets"}}}, "mac": "<hex>"}` at the top of the job's run directory. The controller records the framework's client identity and selection. The signature is an HMAC-SHA256 over the job id and canonical manifest, using a key derived for that job from `<server workspace>/fdn/manifest.key` (32 random bytes, mode 0600). Only the holder's result with a valid signature is applied. A packaged file or a signature from another job establishes no publication. A job author with BYOC permission executes server-side code and is outside this boundary.

**Handoff.** On `BEFORE_JOB_LAUNCH`, the manager writes the holder's `catalog_context.json` with its derived key and the previously indexed names of its own Study. The job's container sees this through its run directory and never accesses the parent catalog store. The controller reads and removes its copy. A missing context produces an unsigned result.

**Private return path.** On `BEFORE_JOB_WORKSPACE_ARCHIVE`, the parent captures the result in `<server workspace>/fdn/catalog-results/<job_id>.json`, mode 0600. It then removes the result, context and their temporary files from the run directory, including files returned by container uploads. Failure to save or remove a private file blocks archiving. On completion the manager verifies and ingests the private copy, then removes it, releasing the Study even when the result is missing or unreadable. Downloadable workspaces retain ordinary logs and configuration, not catalog publications or restricted dataset names.

**Earlier pilot archives.** The fallback reader accepts `catalog_results.json` at the archive root, the real filesystem-storage layout, and the older `<job_id>/catalog_results.json` layout. Before preparing any job download, the generic `BEFORE_JOB_DOWNLOAD` event lets the manager remove catalog result/context files and their temporary copies from a catalog job's stored workspace archive. A result still awaiting ingestion is captured privately before this replacement. Failure to secure an archive blocks the download. Download filtering and ingestion are serialized, so a concurrent download cannot lose or apply a result twice.

### 6.2 Scheduled submission (`PeriodicJobSubmitter`)

Scheduling is a generic server component that knows nothing about the catalog; see Section 9.

## 7. Store (`store.py`)

No database. The store keeps one JSON file per `(study, client)` under `<server workspace>/fdn/catalog/`. A `threading.RLock` serializes the manager and command threads in the process (no catalog job reads the files; the manager hands each job the names it needs); the first design supports one active server, so there is no cross-process writer. The files must be on a local persistent disk.

```
fdn/catalog/<study>/<client>.json    # everything the server knows about one client's catalogs
```

The client file is the unit of atomicity. A result touches exactly one client, so replacing that one file atomically (write `<client>.json.tmp`, `fsync`, `os.replace`) commits every dataset update, the version counters and health together. `format` is a file-level integer for future migrations; an unknown value refuses to load rather than guessing.

```json
{
  "format": 2,
  "client": {
    "health": "OK", "reason_code": null, "failures": 0,
    "last_attempt_at": "...", "last_success_at": "..."
  },
  "datasets": {
    "clinical-research": {
      "metadata": { },
      "metadata_digest": "...",
      "catalog_version": 3, "schema_version": 2,
      "observed_at": "...", "published_at": "...",
      "available": true, "last_checked_at": "...",
      "withheld": false
    }
  }
}
```

- **One read per Study.** Results need no ordering: only one catalog job per Study runs at a time (Section 9.1), so results are applied in the order the jobs ran, and each job's `previous` (the names the server holds, handed to it at launch) already includes every earlier result. A withdrawal therefore always reaches the server as `withheld` for a name it holds, and a read cannot publish a name a later read has already left out. `datasets[*].metadata` is the latest accepted permitted metadata, exactly as accepted, retained while the dataset is unavailable. Only the current metadata is stored; there is no per-version history (see below).

**Startup.** The store loads every client file into memory and builds the search structures from it: a dict keyed by `(study, client)` holding each client's state. Search scans it; there are no inverted maps, because a Study has at most a few thousand entries. A file that fails to parse or has an unknown `format` marks that client `UNKNOWN` and is reported in health; it never blocks other clients. A leftover `*.tmp` file is deleted.

**Commit.** `commit(study, client, state)` writes the client's whole state to a temporary file, `fsync`s it and replaces the client file; only then is the in-memory copy updated, so a failed replace changes neither. A result is applied from the job's saved workspace in one replacement per client, so a crash before the replace loses that run's result for the client until the next run, and a crash after it leaves the result applied. There is no separate log to keep consistent.

**Lost state.** The client files are the only copy of the accepted state, so they are what to back up. A file that is missing or unreadable starts the client empty and unknown, and its next read refreshes it from the sources (reads are repeatable). Catalog and schema versions then restart at 1. The server does not delete a finished catalog job, so the job store grows by one job per Study per interval for scheduled reads; delete old ones with `nvflare job delete`.

**Why files, and when to revisit.** Every operation the catalog performs is a read or a replacement of one client's state, plus an append. Atomic replace gives that without a server dependency, and per-job deployments do not carry a database. Revisit if one Study exceeds roughly 10^5 datasets, if more than one server process must write, or if the parent design's `FDNStore` moves to a different backend; the `CatalogStore` interface below does not change in that case.

```python
class CatalogStore:
    def __init__(self, workspace_root: str): ...                      # loads and indexes all client files under <root>/fdn/catalog
    def record_no_result(self, study, client, reason, now) -> None   # health FAILED, failures + 1; availability unchanged
    def search(self, study, query: SearchQuery) -> SearchPage
    def health(self, study) -> list[dict]
    def client_state(self, study, client) / commit(self, study, client, state)
    def indexed(self, study) -> dict[str, list[str]]              # names held per client, for the job context

def manifest_key(workspace_root) -> bytes | None                  # the master key; a missing, empty or wrong-length file is replaced whole
```

Every mutating method loads nothing from disk: it edits the in-memory client state under the lock, then persists that client's file with atomic replace before releasing the lock. `write_atomically` creates its files owner-only (0600): stored catalogs and the job context hold withheld names and a job key. The master key `<server workspace>/fdn/manifest.key` (32 bytes, 0600) is written to a temporary file, `fsync`ed and moved into place with `os.replace`, so a crash never leaves an empty key that would disable ingestion; a missing, empty or wrong-length key is replaced, and jobs in flight then fail their signature check once.

## 8. Reconciler (`reconcile.py`)

`Reconciler.apply(study, client, selection, raw_result, now) -> APPLIED | UNSUPPORTED` holds the store lock, builds the client's new state from a copy, and calls `store.commit(study, client, state)`, which atomically replaces the client file (Section 7). `apply_result` is not a store method:

```
row = client record for (study, client)
parsed = parse_result(raw_result, selection)                     # selection comes from the manifest
# every parsed catalog is checked with validate(cat, read_limits); a failure moves it to parsed.rejected

# 1. restrictions first (they survive any failure below)
for name in parsed.withheld:    if indexed: set withheld = 1
# 2. per dataset, independently
for cat in parsed.catalogs:
    prev = row(study, client, cat.dataset)
    digest = cat.digest()
    obs  = cat.observed_at if valid and <= now + 300s else None
    changed = (prev is None or digest != prev.digest)
    if changed: catalog_version += 1; schema_version += 1 only if tables/resources differ
                replace metadata and digest
    set observed_at = obs, available = 1, last_checked_at = now, withheld = 0
    validate(cat, read_limits) first; an invalid or over-bound catalog is rejected
for name in rejected: if indexed: set withheld = 1; preserve metadata, versions, availability and check time
for name in parsed.unavailable: if indexed: set available = 0, last_checked_at = now
# 3. health
row.last_attempt_at = now; health = OK/DEGRADED(reason_code), failures = 0, last_success_at = now (OK) | FAILED, failures + 1
```

Rejected replacements set `reason_code = INTERNAL`. With a healthy catalog or explicit checked unavailability the read is `DEGRADED`; if every catalog is rejected it is `FAILED` and does not advance `last_success_at`. Only a later accepted catalog restores the hidden entry. This also covers a site whose read budget exceeds the server validation ceiling.

Properties (each has a test): an unchanged digest creates no new version; a timestamp-only change creates no new version; a catalog that fails validation does not stop others; unselected rows are never touched; `withheld` wins when a name is also in `catalogs`; the schema version moves only when `tables` or `resources` change.

`store.record_no_result` increments `failures`, sets health `FAILED`, the reason and `last_attempt_at`, and does not touch availability or check times.

## 9. Manager and scheduling

### 9.1 `CatalogManager` (`manager.py`)

```python
class CatalogManager(FLComponent):
    def __init__(self, stale_after_s: int = 172800, read_limits: dict | None = None,
                 register_commands: bool = True): ...
    # read_limits is the server's validation budget for catalog metadata
    def handle_event(self, event_type, fl_ctx):
        # SYSTEM_START: open the store and private inbox, restore DISPATCHED/RUNNING holders,
        #   start the manager thread; SYSTEM_END: stop;
        # SUBMIT_JOB: refuse a catalog job while its Study's read is held or unfinished;
        # BEFORE_JOB_LAUNCH: hand the holder its context; BEFORE_JOB_WORKSPACE_ARCHIVE: capture and remove private files;
        # BEFORE_JOB_DOWNLOAD: redact private files from legacy archives, or block the download;
        # JOB_COMPLETED / JOB_ABORTED: queue the job id for the manager thread
```

The manager schedules nothing. It runs one catalog read per Study at a time and applies its result, however the job was submitted (`python job.py`, `nvflare catalog read` or a `PeriodicJobSubmitter`); a catalog job is one that carries the `fdn_catalog_read` meta property.

- **One read per Study.** The manager keeps `_in_flight`, Study → the job that holds its read. *At launch* (the guarantee): on `BEFORE_JOB_LAUNCH` only the holder, the first catalog job launched for the Study or the job already holding it, gets `catalog_context.json`; another catalog job for the Study runs without a key, so its manifest is unsigned and never applied. This covers every way a job is created, including `clone_job`, which does not fire `SUBMIT_JOB`. *At submit* (a convenience): on `SUBMIT_JOB` a catalog job is refused with `JOB_BLOCK_REASON` ("a catalog read for study X is still running or being applied: job …") while the Study has a holder or an unfinished (`SUBMITTED`, `APPROVED`, `DISPATCHED` or `RUNNING`) catalog job in the job store, so the user is told instead of running a job for nothing. The holder is released once its result, or its missing result, has been applied. On start the manager restores holders only for `DISPATCHED` or `RUNNING` jobs. Pending `SUBMITTED`/`APPROVED` jobs already block through the job-store check and do not acquire a holder, so aborting one before launch cannot strand the Study. The manager also checks holders for terminal or deleted jobs when its event queue is idle, covering terminal transitions without lifecycle events. A scheduled job refused because a manual read is running is retried an interval later; the manual read has just refreshed the catalog.

- **Applying a job.** The completion events arrive on the job runner's shared completion thread, so the id is queued and handled on the manager thread. For the job that holds its Study's read, the manager reads the job's meta (Study, deploy map) and its result from the private inbox (with the legacy archive fallback in Section 6.1), and verifies its signature, then, for each client in the manifest, applies an `OK` entry with the Reconciler or records no result (return code, or `UNSUPPORTED_RESULT`). One client's failure does not stop the others. `JOB_ABORTED` and `JOB_COMPLETED` both arrive for an aborted job; only the holder is applied, and it is released after, so the second event changes nothing. A job without a valid manifest (none written, unsigned or not signed for this job) records `NO_RESULT` for each of its sites: the sites of its deploy map, or, for `@ALL`, every site the catalog knows in the Study. Storage reads and cleanup are inside the holder-release failure path: a missing workspace records `NO_RESULT` and cannot strand the Study. A job that is not a catalog job, or does not hold its Study's read, is ignored.
- **Missed events.** The result of a job that completes while the server is down is not applied; the next run replaces it. A site that enrolls later is read at the next run.
- **Health** (`health(study)`): `client, health, reason_code, failures, last_attempt_at, last_success_at, stale` (`stale` when there was no success within `stale_after_s`, default two days). A budget-limited read shows as `DEGRADED` with `reason_code = BUDGET_EXCEEDED`; the skipped names are not listed.
- **Authorization.** A user needs access to the Study to submit the recipe job; that is ordinary job authorization, so the catalog adds no role check of its own. Reading the catalog is Section 10.1.

### 9.2 `PeriodicJobSubmitter` (`nvflare/app_common/schedule/periodic_job_submitter.py`)

```python
class PeriodicJobSubmitter(FLComponent):
    def __init__(self, recipe: str, recipe_args: dict | None = None, interval_s: int = 86400,
                 studies: list[str] | None = None, jitter_fraction: float = 0.1,
                 startup_delay_s: float = 60.0, submit_timeout_s: int = 3600,
                 signer_key: str | None = None, signer_cert: str | None = None): ...
```

A generic server component: every `interval_s` it builds `recipe(**recipe_args)` (the recipe is a dotted class path), exports it, signs it and submits it once per Study. It has no catalog logic, so any recipe can be scheduled with it.

- **Studies.** The configured `studies`, else every registered Study, else the default Study when there is no registry. The first job of each Study starts within `startup_delay_s` of the server starting; later ones an interval (with jitter) after the previous submission.
- **One at a time.** A Study whose previous job has not finished is skipped. A finished or deleted job is forgotten; a job still `SUBMITTED` after `submit_timeout_s` (it never started, for example no site connected) is aborted through `job_runner.abort_before_start`, so the next one can be submitted; a job the runner is starting at that moment is left to run. The job is never deleted under the runner. A running job is left to its own timeouts. The submitter checks every 5 seconds.
- **Failure.** A recipe that cannot be built or a failed submit is logged and retried one interval later, not every tick.
- **Signing.** A deployment that requires signed jobs (`require_signed_jobs`, the default when `rootCA.pem` exists) accepts a job only if each folder carries a signature and a certificate that chains to the root CA and is an admin certificate. The submitter signs every job with `sign_folders`, the same routine the admin client uses, with a service identity the operator provisions as an admin-role user: take an admin-role startup kit's `client.key` and `client.crt` and store them under the server's `private/scheduled_jobs/` directory, then name them in `signer_key` and `signer_cert` (paths relative to the server workspace). Use owner-only directory/key permissions. Both configured and resolved paths must stay under that directory; job-visible paths and symlinks escaping it are refused. The directory is never mounted or bundled into a job. The certificate's name, organization and role are recorded as the job's submitter. Without a signer the jobs are unsigned, recorded with submitter name `scheduled-job-submitter` and role `project_admin`, and a deployment that requires signed jobs rejects them; the component logs that at startup.
- **Submission.** The job is validated with `JobMetaValidator`, gets `STUDY`, the submitter and `JOB_FOLDER_NAME`, passes the `SUBMIT_JOB` event like a user submission (so `cc_manager` can block a BYOC job in confidential-computing mode and job metrics count it; a `JOB_BLOCK_REASON` makes it a failed submit), and is stored with `job_def_manager.create`. It skips the admin command's submit-token records, because it is the server's own service job. The default scheduler then schedules it like any other job. The server never deletes a finished job.

## 10. Commands, client and CLI

### 10.1 `CatalogCommandModule` (`commands.py`)

`CatalogManager.__init__` registers the module with `ServerCommandModules.register_cmd_module` (it replaces any earlier registration). Server components are built while the config is parsed, before the admin server reads that list, so the commands are registered in time. This is established from the code (`fl_conf.py`, `server_deployer.py`) and unit-tested; it has not been run against a live server and admin session.

| Command | Auth (`authz_func`) | Request | Reply |
| --- | --- | --- | --- |
| `fdn_catalog_search` | the user is an admin role (`project_admin` or `org_admin`, which pass the membership check for every Study by design), or the Study is the default Study, a registry member (`StudyRegistryService.has_user`) or in the session certificate (`ConnProps.CERT_STUDIES`); the check returns `REQUIRE_AUTHZ`, so the site's `authorization.json` then decides the right (`view`) | `{filters, cursor, page_size}` | `{entries: [...], next_cursor, facets, total}` |
| `fdn_catalog_status` | same as search (`view`) | `{}` | per-client health from Section 9.1 |

Rules: Study comes from `conn.get_prop(ConnProps.ACTIVE_STUDY)`, user and role from `ConnProps`; the request body never carries them. Membership is re-checked on every call and page. `page_size` is capped (for example 100); a non-null cursor must bind the Study and the filter hash or it is rejected. Request parsing rejects unknown keys and values over fixed bounds; no raw SQL or paths. The request dictionary reaches the handler as `conn.get_prop(ConnProps.CMD_PROPS)`.

Failure replies reuse the `_error` shape from `study_cmds.py` (`error_code`, `message`).

### 10.2 Search (`search.py`)

```python
@dataclass
class SearchQuery:
    client: str | None; dataset: str | None; text: str | None
    modality: str | None; standard: str | None; standard_version: str | None
    operation: str | None; available: bool | None; schema_version: int | None; mapping_version: str | None
    cursor: str | None; page_size: int
```

One pass over the in-memory entries per call: start from the Study's entries with `withheld` false, then apply the optional filters (`modality`, `operation` and `standard*` against the entry's lists and profile; `text` is a case-insensitive substring match on `display_name`). Order and cursor are over a stable sort of `(client, dataset)`. Facets and `total` use the same filtered set, so suppression applies to them. Projection: if `available` is false return only `{client, dataset, available, last_checked_at}` and metadata filters (`text`, `modality`, `standard*`, `operation`, `schema_version`, `mapping_version`) do not match such entries; otherwise return `{client, dataset, metadata, catalog_version, schema_version, observed_at, published_at, available, last_checked_at}`. The stored metadata is already the permitted view; no field is removed at query time.

### 10.3 `CatalogAdminApi` (`admin/catalog_api.py`)

Users get typed results, not raw dictionaries:

```python
@dataclass(frozen=True)
class CatalogEntry:                       # one search result
    client: str; dataset: str
    available: bool; last_checked_at: str | None
    catalog: CatalogInfo | None               # None while unavailable
    catalog_version: int | None; schema_version: int | None
    published_at: str | None

@dataclass(frozen=True)
class SearchPage:    entries: tuple[CatalogEntry, ...]; next_cursor: str | None; total: int; facets: Mapping[str, Mapping[str, int]]
@dataclass(frozen=True)
class ClientHealth:  client: str; health: str; reason_code: str | None; failures: int; last_attempt_at: str | None; last_success_at: str | None; stale: bool

class CatalogAdminApi:
    def __init__(self, session: Session): ...
    def search(self, **filters) -> Iterator[CatalogEntry]        # follows next_cursor
    def search_page(self, cursor: str | None = None, page_size: int = 50, **filters) -> SearchPage
    def status(self) -> list[ClientHealth]
```

The same typed classes serve scripts, notebooks, the CLI output and any later agent client. Each method builds a bounded request dict, calls `session.do_command(<command>, props=request)`, checks the response meta status, rebuilds typed objects with `CatalogInfo.from_wire` and friends (from the `{dataset, metadata, observed_at}` part of each entry; the entry's `client` and version fields are read separately), and raises a typed error from the reply's `error_code`. It owns no connection or login; the Session carries the Study.

### 10.4 CLI (`nvflare/tool/catalog/catalog_cli.py`)

Two subcommands in the style of `nvflare study` (`--study`, `--format json`, `--schema`, startup-kit selection, shared error and exit-code helpers). `nvflare catalog show` lists the stored catalog through `CatalogAdminApi`. `nvflare catalog read [--study S] [--sites a,b] [--only SITE:NAME]... [--timeout N] [--no-wait]` (`--only` is repeatable and builds the per-site `datasets` mapping; it is not `show --dataset`, which only filters the listing) builds and exports a `CatalogRecipe`, then submits it through the shared CLI Session for the Study, so no job file is needed. It waits for the actual terminal job status by default without downloading results; `--no-wait` returns the job id immediately after submission. When waiting, successful completion exits 0; failed, aborted or unschedulable jobs exit 1. Connection/authentication failures exit 2, timeouts 3, bad arguments 4 and unexpected errors 5. The server applies the result when the job completes, and every read refreshes what is already stored, so there is no separate `refresh` command. `read` adds no server command: Study access, signing and authorization are those of any job submission. The search filters and `fdn_catalog_search`'s other options stay in the server command and the Python client, not the CLI.

**`nvflare catalog show [--study S] [--site NAME] [--dataset NAME]`** shows the catalog.

- With neither `--site` nor `--dataset`: one summary row per dataset (`site, dataset, display_name, available, catalog_version, schema_version, published_at, last_checked_at`; an unavailable dataset shows only `site, dataset, available, last_checked_at`). `--site` or `--dataset` alone narrows the list.
- With both: that one dataset in full (`metadata` with tables, columns, relationships, semantic profile, lineage and provenance, plus `observed_at`). A missing dataset is `DATASET_NOT_FOUND` with a hint.
- JSON output is `{study, total, datasets, sites}` or `{study, dataset, sites}`; `sites` is `[{site, health, reason_code, last_success_at, stale}]`. The human view prints a table (or the dataset's tables and columns) and actionable warnings for FAILED, DEGRADED and stale sites, even when no datasets are visible. In JSON mode these diagnostics go to stderr.

Both commands default to human-readable text and support `--format json`: stdout contains exactly one versioned shared envelope, with data under `data`; progress and warnings go to stderr. `read` data is `{study, job_id, status}` by default, or `{study, job_id}` with `--no-wait`. With `--no-wait`, exit 0 confirms submission, not job completion. `--schema` is local discovery and requires no kit or connection. `read` mutates state and is not idempotent; it offers no retry-token deduplication. `show` is read-only and idempotent.

Errors use the shared helpers: a bad argument exits 4, an unauthorized command is `NOT_AUTHORIZED`, and server replies carry `error_code`, `message` and `hint`. The CLI says `site` where the admin commands say `client`.

## 11. Wiring and configuration

- **Server** `resources.json` `components`: add `CatalogManager` (applies results, answers `nvflare catalog show`) and, for periodic reads, a `PeriodicJobSubmitter` for the recipe. Both are absent unless configured, so existing deployments are unaffected. `CatalogManager` registers `CatalogCommandModule` itself (Section 10.1).

```json
{"id": "fdn_catalog", "path": "nvflare.app_common.fdn.catalog.server.manager.CatalogManager", "args": {}},
{"id": "fdn_catalog_schedule", "path": "nvflare.app_common.schedule.periodic_job_submitter.PeriodicJobSubmitter",
 "args": {"recipe": "nvflare.recipe.catalog.CatalogRecipe", "interval_s": 86400,
          "signer_key": "private/scheduled_jobs/admin.key", "signer_cert": "private/scheduled_jobs/admin.crt"}}
```

  `signer_key` and `signer_cert` are an admin-role startup kit's `client.key` and `client.crt` stored under the server's `private/scheduled_jobs/` directory with owner-only permissions (needed when the server requires signed jobs). `local/`, `startup/` and job directories must contain no copies of the signing key. `studies` narrows the scheduled Studies; the default is every registered Study.
- **Site**: the Study policy file at `<policy_root>/<study>.json` and the site config at `<policy_root>/catalog_site.json` (`policy_root` is `/fdn/policy`), mounted read-only into the job by the site's `secret_mounts` entry (container modes), or kept in `<workspace>/local/fdn/` (process mode); `local/study_data.yaml` unchanged. `local/fdn/` is added to the workspace-bundle excludes (`workspace_cell_transfer.py`), so Kubernetes never ships a copy inside a bundle and a pod without the mount fails closed.
- **Job**: carries only the optional dataset selection and the task timeout (the recipe arguments). The connectors, their arguments, limits and installed operations come from each site's `catalog_site.json`; the manager's `read_limits` is only the ceiling the server validates results against.
- **CLI**: `nvflare catalog show` and `nvflare catalog read` (Section 10.4), registered in `nvflare/cli.py`. `read` runs `CatalogRecipe` (Section 6.1).
- **Authorization**: `fdn_catalog_search` and `fdn_catalog_status` are `VIEW` in `COMMAND_CATEGORIES` (`nvflare/security/security.py`), so the site's authorization policy decides them like other view commands.
- **Job runner**: before archiving a finished job's workspace, `JobRunner` fires the generic event `BEFORE_JOB_WORKSPACE_ARCHIVE` (with `CURRENT_JOB_ID`); a component removes the private files it put in the run directory, or sets `JOB_BLOCK_REASON` to stop the archive, which is then retried like any failed archive. Core code names no catalog file. `CatalogManager` captures the holder's result in its private inbox and removes both `catalog_results.json` and `catalog_context.json` (and their `.tmp` files). The generic `BEFORE_JOB_DOWNLOAD` event also lets it redact legacy archived copies before any artifact download is prepared; a failure blocks that download.

## 12. Existing-code assumptions

Checked against the code or by running it:

1. The exported recipe job passes the real `JobMetaValidator` (the job zip needs explicit folder entries). A server component can set `STUDY` and a service submitter and call `job_def_manager.create`; `create` stamps `submit_time`. `FedJob(meta_props=...)` writes the marker into `meta.json`.
2. `JobRunner` captures private files before archiving and publishes terminal status after its archive attempt. The manager reads the private inbox on completion. Regression coverage uses the actual filesystem archiver: directory sources lose their top-level directory name, so the legacy manifest is at `catalog_results.json`, not `<job_id>/catalog_results.json`.
3. `JOB_COMPLETED` and `JOB_ABORTED` reach a component registered in the server `components` list with the job id in `CURRENT_JOB_ID`. They do **not** fire for a job that is never scheduled or is aborted before it starts, so such a job has no result, and the submitter aborts a scheduled one that never started.
4. `Session.do_command(command, props)` delivers `props` to the handler as `ConnProps.CMD_PROPS`.
5. A real simulator run of `CatalogRecipe` (two clients) produces a valid `catalog_results.json` with both clients; the framework calls task callbacks with keyword arguments. The default component allow list rejected the controller until it was added to `DEFAULT_CLASS_ALLOW_LIST`.

6. A live two-site POC with signed jobs (admin kit key and certificate as the signer) passed: the scheduled job ran on both sites about 20 s after the server started and `nvflare catalog show` listed both datasets; a hand-run `CatalogRecipe` (also with `sites` and `datasets`) updated the check times without a new version; a site policy changed to `share: false` withdrew that site's entry; scheduled jobs did not pile up. Findings that stay as documented limits:
   - A site that is not connected when the job starts is not in the job, so it is not in the manifest. Its stored entry is left as it was and shows as stale only after `stale_after_s`.
   - A site that dies during the job holds the job RUNNING until the heartbeat timeout (600 s in the POC); the result is still applied, and the manifest also applies after `FINISHED:ABNORMAL`.
   - Finished catalog jobs are not deleted, so the job store gains one small job per Study per interval.
   - Catalog reads of one Study never overlap: a second one is refused at submit, or runs without a key and is not applied if it reached launch another way. A scheduled read refused because a manual one runs is retried an interval later.
   - `operations` in the stored metadata is what the connector reports, the policy allows and the site has in `installed_operations`; with none installed it is empty.

Still to verify:

7. `FLContextKey.JOB_META` and `/data/<study>/<dataset>` plus the policy `secret_mounts` entry under the Docker, Kubernetes and Slurm container launchers (read-only, atomic-replacement visibility). Slurm sandbox `none` is unsupported for catalog reads.
8. A blocked `pyarrow` read can be abandoned when `abort_signal` fires; the executor checks the signal between datasets only.
9. Under the Kubernetes launcher, `catalog_context.json` reaches the server-job pod before the controller starts (the bundle is built relative to `BEFORE_JOB_LAUNCH`), and the manifest comes back through the container result upload before private capture and workspace archiving.

## 13. Test plan by module

| Module | Cases |
| --- | --- |
| `schema` | every §3.3 example validates; unknown key, duplicate name, unresolved reference, over-bound sizes fail; source types and list order reported unchanged; digest stable and sensitive to order; `to_wire`/`from_wire` round trip; `record_count` |
| `payload` | build/parse round trip; unknown `format_version`; overlapping lists resolve to `withheld`; names outside the selection dropped; malformed catalog yields a rejected entry but keeps `withheld` |
| `policy` | each §3.1 key; unlisted dataset has no rule; `share:false`; invalid file variants (unknown key, wrong type, bad version, missing `share`, missing file); excluded table/column removes dependent relationship and empties a broken primary key; `exclude_metadata` nulls; `operations` intersection |
| `site_config` / `executor` (connectors) | connectors defined without datasets; each dataset bound to a connector by name per Study; a Study mixing several source types; the dataset's `options` reach its connector; a binding belongs to its Study and beats the default connector for a mounted dataset; the built-in Parquet connector needs no section; an undefined connector name, bad name or unknown key refused without echoing values; a connector that cannot be built costs only its own datasets, and all failing withholds indexed names; every built connector closed |
| `parquet_connector` | schema mapping, partitioned dataset, corrupt footer → `ConnectorError`, sidecar validated, budget stop |
| `executor` | corrupt B among A/B/C; removed binding → unavailable; budget skip of a previously indexed name → withheld; policy missing → `POLICY_UNAVAILABLE` with previous names withheld; request-supplied path/class ignored; lists disjoint; no exception text in the result |
| `store` | atomic replace leaves no partial file; a failed replace changes neither the disk nor the memory copy; state survives a restart; corrupt, unknown-format or wrongly typed file starts that client empty and is logged, and never breaks a search of the Study; stored files are owner-only; the master key is created owner-only, kept, and replaced when empty or short; unsafe names cannot escape the store; `indexed(study)` lists the names held per client; cursor pagination |
| `reconcile` | first publication; unchanged digest adds no version; descriptive change bumps the catalog version only, structural change bumps both; bad `observed_at` stored as null with metadata accepted; withheld first and survives a bad catalog; unavailable hides metadata but retains it; restore on a later catalog; a failed result changes no availability and counts a failure; no-result keeps availability; unsupported format changes nothing; identical dataset names at two clients stay separate |
| `controller` | arguments validated; `previous` and the job key read from `catalog_context.json` (then removed), narrowed by that site's `datasets`; a missing context gives an unsigned manifest; manifest round trip and malformed manifests; client name from the framework, not the payload |
| `recipe` | exported job: `@ALL` or named sites, marker in `meta.json`, controller arguments, executor with no arguments |
| `manager` | a submit refused while the Study's read is held or an earlier catalog job waits to launch, other Studies and other jobs not refused; only the first job launched for a Study gets a context; the next read launches once the result is applied; a job refused at launch is never applied; a job surviving a restart keeps its Study's read; context owner-only and limited to the job's Study; real filesystem archive layout; private result capture before archiving and cleanup after ingestion; failed capture/removal blocks the archive; legacy download redaction preserves logs and blocks on failure; pending pre-start abort and terminal/deleted holder recovery; missing workspace releases the holder; every client of a manifest applied in the job's Study; forged, unsigned or other-job manifests apply nothing; a job without a manifest records `NO_RESULT` for its deploy-map sites (all known sites for `@ALL`); an aborted job reported twice is applied once; one client's failure does not stop the others; events queued; stale reporting |
| `periodic_job_submitter` | one job per Study with the service identity; not resubmitted while running; next job an interval later; startup delay; failed submit retried an interval later; never-started job aborted through the job runner, a starting one left alone; `SUBMIT_JOB` fired and a block reason refuses the job; Study resolution; signing and missing-signer report; parent-only key/certificate paths, traversal/symlink rejection, and exclusion from real job bundles |
| `commands`/`search` | Study from the connection only; membership checked per call; withheld never returned, including facets and totals; unavailable projection; cursor bound to Study and filters; unknown keys rejected |
| integration | simulator run of `CatalogRecipe` (done); POC with a real server, the scheduled and the hand-run job, and `nvflare catalog show` |
