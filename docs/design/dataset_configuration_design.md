# Site Dataset Configuration: Read-Only Access API

**Status:** Proposed design
**Date:** November 9, 2026

## 1. Purpose

A site describes the datasets it offers to each Study in one site-owned configuration. Components running inside a job read that configuration through one read-only API instead of opening and parsing their own files. Examples of such components are a federated data catalog, cohort counting and training data loaders.

Goals:

- One place where a site declares, per Study, which datasets exist, where a job finds them, how to connect to them, and how each component may use them.
- One read-only API, bound to the job's Study, that every component uses. Components never learn the file's location or format.
- Component-specific rules, such as a catalog's publication policy, live next to the dataset they govern but are interpreted only by their component.

Non-goals:

- Writing, registering or discovering datasets through the API. The API has no write methods; the site operator edits the configuration.
- Placing data. How a source reaches the job (Docker or Kubernetes mounts, secret projection, the container image) stays in the launcher's [study runtime configuration](study_runtime_config.md). This configuration describes datasets as the job sees them.
- Per-user access rules. Visibility follows the job's Study and the existing FLARE authorization.

## 2. Configuration structure

### 2.1 Location and ownership

The configuration is the site-owned file `<workspace>/local/datasets.yaml`, next to `study_runtime.yaml`. Like that file, it covers every Study at the site and belongs to the site. Jobs cannot supply or replace it: it is never part of a job package, and the API ignores any copy inside the job's app directory. In Kubernetes the file can come from a ConfigMap mounted into the parent pod's `local/` folder and edited with `kubectl`.

A site's per-Study configuration is two files in `local/`:

| File | Read by | Holds | Since |
| --- | --- | --- | --- |
| `study_runtime.yaml` | The launcher | How each Study's jobs are run: image, mounts, environment, secret sources, launcher settings | FLARE 2.9 (process mode: 2.10) |
| `datasets.yaml` | Components, through this API | What each Study's datasets are and how components may use them | FLARE 2.10 (this design) |

`study_data.yaml`, the 2.8 mount-only file that `study_runtime.yaml` replaces, is legacy. Docker and Kubernetes still read it for kits that have not migrated, Slurm rejects it, and process mode ignores it. New sites, and this design, use `study_runtime.yaml`.

A job loads the configuration once, when a component first asks for it, and keeps it for the job's life (Section 3.4). An edit applies from the next job, in every launch mode, without a restart of the site.

### 2.2 Format

```yaml
format_version: 1
studies:
  oncology:                       # Study name
    datasets:
      omop:                       # dataset name: how recipes and results refer to it
        type: parquet             # source type; selects how the dataset is read
        uri: file:///data/oncology/omop   # required; where the container launcher mounts this dataset
                                          # (three slashes: empty host, then the path /data/oncology/omop)
        usage:                    # how this dataset may be used, per FLARE component
          catalog:                # rules for the federated data catalog
            share: true           # publish this dataset's metadata
            exclude_columns: {person: [person_source_value]}   # table person, column person_source_value

      registry:                   # dataset name; here the dataset is a whole SQLite database
        type: sql                 # SQLite is SQL; the file URI says it is a local database file
        uri: file:///srv/oncology/registry.sqlite    # explicit file URI: the path is /srv/oncology/registry.sqlite
        usage:
          catalog: {share: true}

      ehr:                        # dataset name; here part of a network database
        type: sql
        uri: postgresql+psycopg://ehr-db.oncology.example:5432/ehr   # network URL: no credentials in it
        options: {schema: public, tables: [visits, labs]}           # read only these tables; others are not read
        secrets: {username: EHR_DB_USER, password: EHR_DB_PASSWORD}  # environment variable names, not values
        usage:
          catalog: {share: true}

      warehouse:                  # dataset name; here a Unity Catalog schema
        type: unity-catalog
        uri: https://uc.oncology.example/api/2.1/unity-catalog       # network URL: the API root
        options: {catalog: main, schema: oncology}
        secrets: {token: UC_TOKEN}
        # no usage.catalog: the catalog never publishes this dataset

usage:                            # site-wide settings, per FLARE component
  catalog:
    limits: {max_duration_s: 60, max_calls: 1000, max_records: 10000}
```

| Key | Required / default | Meaning |
| --- | --- | --- |
| `format_version` | Required; integer `1` | Booleans and other versions are rejected. |
| `studies` | Optional; `{}` | Maps a Study name to the data this site offers that Study. Each entry holds only `datasets`; it is not the Study's own configuration (Section 2.4). A Study that is absent offers no datasets. |
| `studies.<study>.datasets` | Optional; `{}` | Maps a dataset name to its definition. Study and dataset names follow the Study runtime naming rule: 1 to 63 characters of lowercase letters, digits, `-` and `_`, starting and ending with a letter or digit. |
| `<dataset>.type` | Required | What kind of source the dataset is: `parquet`, `csv`, `sql`, `duckdb`, `unity-catalog`, or a type a site-installed component understands. One value per kind of source, whether it is a file or a service: a SQLite file and a PostgreSQL server are both `sql`. The API does not interpret the type. |
| `<dataset>.uri` | Required | Where the dataset is, as an absolute URI. There is no default. Section 2.3 gives the forms, the mount convention and the validation rules. |
| `<dataset>.options` | Optional; `{}` | Non-secret, type-specific settings, such as a database schema or table list. |
| `<dataset>.secrets` | Optional; `{}` | Maps a logical key to the **name** of an environment variable that holds the secret. Never a value. The variable reaches the job through the Study runtime `secret_env` (containers) or the site's process environment. |
| `<dataset>.usage` | Optional; `{}` | How the dataset may be used, per FLARE component: maps a component name, such as `catalog`, to that component's rules for this dataset. It never lists tables; a source's tables are discovered when it is read. A component absent from the map has no rules, and each component decides what that means (the catalog does not share the dataset). |
| `usage` | Optional; `{}` | Site-wide settings per component, such as the catalog's read limits. |

Unknown keys outside `usage` sections, wrong value types, a missing `type` or `uri`, a `uri` that breaks the rules of Section 2.3, and a secret value given where a variable name is expected are rejected; the whole file is then invalid. Contents of `usage` sections are not validated by the API: each component validates its own section and rejects what it does not understand.

**Format version.** Every file states `format_version`, as `study_runtime.yaml` does; it is the gate for changing the format later. This design is version 1.

- The API reads the versions this FLARE release supports and rejects any other value, including a boolean, so a file written for a newer format fails closed on an older FLARE instead of being half-read.
- The version changes only for an incompatible change: removing or renaming a key, or changing what a key means. A new optional key or a new dataset `type` keeps the version; because unknown keys are rejected, a file that uses one needs a FLARE release that knows it.
- `usage` sections are versioned by their components, not by this number: each component validates its own section.
- The file is versioned on its own, independently of `study_runtime.yaml`'s `format_version`. In Kubernetes, the part the parent sends a job keeps the site file's version.

### 2.3 Dataset URI

`uri` says where a dataset is. It has two forms:

| Form | Example | Meaning |
| --- | --- | --- |
| `file` URI | `file:///srv/oncology/registry.sqlite` | A path in the job's filesystem: a Parquet directory, a CSV file, or a SQLite or DuckDB database file. |
| Network URL | `postgresql+psycopg://ehr-db.example:5432/ehr`, `https://uc.example/api/2.1/unity-catalog` | Any other scheme: an address the job connects to, such as a SQLAlchemy database URL or an HTTPS API root. |

**Required, with no default.** Every dataset states its `uri`. A missing `uri` makes the file invalid, so a network dataset or a process-launched site can never fall back to a mount path by mistake.

**Mount convention.** The container launchers mount the Study runtime dataset of the same name at `/data/<study>/<dataset>`. A dataset mounted that way uses

```text
file:///data/<study>/<dataset>
```

with the Study and dataset names filled in, for example `file:///data/oncology/omop`. A process-launched site mounts nothing, so it uses a `file` URI where its data already is.

**Three slashes.** A `file` URI is `file://`, then a host, then an absolute path. The host is always empty here, so the path's leading `/` makes three slashes: `file:///data/oncology/omop` is the path `/data/oncology/omop`. `file://data/oncology/omop` would make `data` a host and is rejected.

**Rules.** The API rejects the whole file when a `uri`:

- is missing;
- is not absolute (it has no scheme);
- is a `file` URI with a host, a query or a fragment, or a path that is not absolute;
- contains a password (`user:password@`). A URI is not secret; credentials go in `secrets`.

The API does not check that a `file` path exists or that a network address answers; the component that reads the dataset reports that. Percent-encoding follows RFC 3986, so a space in a path is written `%20`. `DatasetConfig.path` returns the decoded local path of a `file` URI.

### 2.4 Boundary with Study runtime configuration

Every data source has two halves. `study_runtime.yaml` owns **access**: what the job can reach, applied by the launcher and enforced outside the job; this is also where Studies are isolated. `datasets.yaml` owns **description**: what the source is and how FLARE components may use it; it never grants access.

| Responsibility | `study_runtime.yaml` | `datasets.yaml` |
| --- | --- | --- |
| Read by | The launcher, before the job starts | FLARE components in the job, through this API |
| How a Study's jobs run: image, pod template, Docker and Slurm settings, plain `env` | Yes | No |
| Access and Study isolation: mounts, delivery of credentials | Yes | No |
| What a source is and where and how to read it: `type`, `uri`, `options` | No | Yes |
| Secrets | Delivers the values (`secret_env`, `secret_mounts`) | Names the variable each dataset uses (`secrets`) |
| How components may use a source: `usage` | No | Yes |
| Enforced outside job code | Yes | No: FLARE components honor it |

How that splits for each kind of source:

| Data source | Access and isolation (`study_runtime.yaml`) | Description and use (`datasets.yaml`) |
| --- | --- | --- |
| Files on a volume (Parquet, CSV, SQLite, images) | Mount only the Study's volume at `/data/<study>/<dataset>` (`datasets` mount entry) | `type: parquet`, `uri: file:///data/<study>/<dataset>`, `usage` |
| Database | Deliver only the Study's database-role credentials (`secret_env`); the role limits what can be read | `type: sql`, `uri: postgresql://…`, `options`, `secrets` names, `usage` |
| Databricks / Unity Catalog | Deliver the Study's token or downscoped credentials (`secret_env`) | `type: unity-catalog`, `uri`, `options`, `secrets` names, `usage` |
| Object storage, REST API | Deliver the Study's credentials (`secret_env`, `secret_mounts`) | `type`, address, `secrets` names, `usage` |

- **Where the halves meet:** at the mount path for a volume, and at the environment variable holding the credential otherwise. A database needs no Study runtime `datasets` entry.
- **`usage` rules are not access control.** `exclude_columns` keeps the catalog from publishing a column, but job code with the access can still read it. What a job can read at all is set by the access half.
- **Unreachable sources** are reported by the component that tries to read them; the API does not check paths or connect to sources.

| Concern | Owner |
| --- | --- |
| The Study itself: its definition, description and membership | The federation's Study registry on the server |
| Mounts, images, launcher settings, secret sources | Study runtime configuration of the launcher |
| Secret values | The site's secret store, delivered as environment variables |
| Observed structure (tables, columns, types) | The dataset itself, read by the component that uses it |

**A Study entry is this site's data for that Study, not the Study's configuration.** `studies.<study>` holds only `datasets`. The Study's own description and settings live on the server, once for the whole federation; a copy here would drift from it at every site. For the same reason there is no Study-level `description`, `notes` or `options`:

- **Notes.** Use YAML comments. Parsing drops them, so they stay with the site operator and never reach a job; in Kubernetes the parent sends the job only the parsed Study part.
- **Settings shared by several datasets.** Use a YAML anchor (`&name`) and merge key (`<<: *name`) instead of Study-level defaults. `yaml.safe_load` resolves them, so the API and the job see ordinary, complete dataset entries, and the format needs no merge rules:

```yaml
studies:
  oncology:
    datasets:
      ehr_visits:
        <<: &ehr_db                      # shared settings, defined once
          type: sql
          uri: postgresql+psycopg://ehr-db.oncology.example:5432/ehr
          secrets: {username: EHR_DB_USER, password: EHR_DB_PASSWORD}
        options: {schema: public, tables: [visits]}
      ehr_labs:
        <<: *ehr_db                      # reused
        options: {schema: public, tables: [labs]}
```

### 2.5 Examples

**A process-launched site.** Nothing is mounted, so each `file` URI is where the data already is. A job submitted without a Study is in the `default` Study.

```yaml
# <workspace>/local/datasets.yaml
format_version: 1
studies:
  default:
    datasets:
      patients:
        type: csv
        uri: file:///home/site-1/data/patients.csv
        usage:
          catalog: {share: true}
```

**A Kubernetes site.** The site's two files work together and are linked by Study and dataset names. `study_runtime.yaml`, read by the launcher, mounts the data and delivers the secrets. `datasets.yaml` describes what the job finds. A network dataset has no Study runtime entry; only its secrets do.

```yaml
# <workspace>/local/study_runtime.yaml: what the launcher places in the job
format_version: 2
studies:
  oncology:
    datasets:
      omop: {source: oncology-omop, mode: ro}         # PVC, mounted at /data/oncology/omop
    secret_env:
      EHR_DB_USER: {source: ehr-db, key: username}     # from Kubernetes Secret ehr-db
      EHR_DB_PASSWORD: {source: ehr-db, key: password}
```

```yaml
# <workspace>/local/datasets.yaml: what the job finds, and how components may use it
format_version: 1
studies:
  oncology:
    datasets:
      omop:
        type: parquet
        uri: file:///data/oncology/omop                # where study_runtime.yaml mounts dataset omop
        usage:
          catalog: {share: true}
      ehr:
        type: sql
        uri: postgresql+psycopg://ehr-db.oncology.example:5432/ehr   # not mounted: the job connects
        secrets: {username: EHR_DB_USER, password: EHR_DB_PASSWORD}  # variables set by secret_env
        usage:
          catalog: {share: true}
```

**Several Studies at one site.** Each job sees only its own Study.

```yaml
format_version: 1
studies:
  oncology:
    datasets:
      omop: {type: parquet, uri: file:///data/oncology/omop, usage: {catalog: {share: true}}}
  cardiology:
    datasets:
      echo: {type: csv, uri: file:///data/cardiology/echo.csv}   # no usage.catalog: never published
```

| Job's Study | `list_datasets()` | Notes |
| --- | --- | --- |
| `oncology` | `["omop"]` | `get_dataset("echo")` raises `UnknownDatasetError`. |
| `cardiology` | `["echo"]` | `get_dataset("echo").usage_rules("catalog")` is `None`, so the catalog does not share it. |
| `default`, or any Study not in the file | `[]` | The site offers that Study no datasets. |

A component reads the configuration the same way in every case:

```python
config = get_study_dataset_config(fl_ctx)          # the job's Study only
for name in config.list_datasets():
    dataset = config.get_dataset(name)
    rules = dataset.usage_rules("catalog")          # None: this component may not use the dataset
    if rules and rules.get("share"):
        read(dataset.type, dataset.path or dataset.uri, dataset.options)
```

## 3. API

### 3.1 Specification

The spec lives in `nvflare/apis/dataset_config_spec.py` and has no I/O or optional dependencies. Its objects mirror the file: a Study holds datasets, a dataset holds its `usage` rules, and site-wide `usage` settings sit beside the Studies.

```python
class DatasetConfigError(Exception):
    """The configuration is missing, unreadable or invalid, or a secret is unset."""


class UnknownDatasetError(DatasetConfigError):
    """The job's Study has no dataset with this name."""


@dataclass(frozen=True)
class DatasetConfig:
    """One dataset of the job's Study: studies.<study>.datasets.<name> in the file."""

    study: str                         # the job's Study
    name: str                          # the dataset name: the key in the file
    type: str                          # source type, e.g. "parquet", "sql", "unity-catalog"
    uri: str                           # absolute URI, required in the file, e.g. file:///data/oncology/omop
    path: str | None                   # decoded local path of a file URI, e.g. /data/oncology/omop;
                                       # None for a network URL
    options: Mapping[str, Any]         # read-only, non-secret, type-specific; {} when omitted
    secret_keys: tuple[str, ...]       # keys of `secrets`, e.g. ("username", "password"); never values
    usage: Mapping[str, Mapping[str, Any]]  # read-only: component name -> that component's rules;
                                            # {} when omitted

    def usage_rules(self, component: str) -> Mapping[str, Any] | None:
        """This dataset's rules for one component, e.g. "catalog"; None when it has none."""


class StudyDatasetConfig(ABC):
    """Read-only snapshot of the job's Study in the site's dataset configuration."""

    @property
    @abstractmethod
    def study(self) -> str:
        """The job's Study."""

    @property
    @abstractmethod
    def revision(self) -> str:
        """Opaque identifier of the snapshot; changes whenever the configuration changes."""

    @abstractmethod
    def list_datasets(self) -> list[str]:
        """The Study's dataset names, in file order; [] when the Study has none."""

    @abstractmethod
    def get_dataset(self, name: str) -> DatasetConfig:
        """One dataset. Raises UnknownDatasetError when the Study has no such dataset."""

    @abstractmethod
    def site_usage(self, component: str) -> Mapping[str, Any] | None:
        """Site-wide settings for one component (top-level usage.<component>); None when absent."""

    @abstractmethod
    def get_secret(self, name: str, key: str) -> str:
        """Value of one of the dataset's secrets, read from its environment variable at call time.

        Raises UnknownDatasetError for an unknown dataset, and DatasetConfigError when the key is
        not configured or the variable is unset. Never logs or returns the value in an error.
        """


def get_study_dataset_config(fl_ctx: FLContext) -> StudyDatasetConfig:
    """The configuration of the job's Study.

    The Study comes from the job metadata in fl_ctx ("default" when the job names none). The
    configuration is loaded on the first call in the job and kept for the job's life (Section 3.4).
    Raises DatasetConfigError when it is unreadable or invalid. A site with no configuration
    yields a snapshot with no datasets.
    """
```

| File | API |
| --- | --- |
| `studies.<study>` (the job's Study) | `get_study_dataset_config(fl_ctx)` returns `StudyDatasetConfig` |
| `studies.<study>.datasets` keys | `list_datasets()` |
| `studies.<study>.datasets.<name>` | `get_dataset(name)` returns `DatasetConfig` |
| `<name>.type`, `uri`, `options` | `DatasetConfig.type`, `.uri` (with `.path`), `.options` |
| `<name>.secrets` | `DatasetConfig.secret_keys` for the keys; `get_secret(name, key)` for a value |
| `<name>.usage.<component>` | `DatasetConfig.usage_rules(component)` |
| Top-level `usage.<component>` | `site_usage(component)` |

A component in a job uses it like this:

```python
from nvflare.apis.dataset_config_spec import get_study_dataset_config

config = get_study_dataset_config(fl_ctx)              # the job's Study only
limits = config.site_usage("catalog")                  # site-wide catalog settings, or None
for name in config.list_datasets():
    dataset = config.get_dataset(name)
    rules = dataset.usage_rules("catalog")             # None: the catalog does not share it
    if rules is None:
        continue
    ...                                                # dataset.type selects how to read it
```

### 3.2 Contract

- **Study-bound.** `get_study_dataset_config(fl_ctx)` returns the job's Study only; a job without an explicit Study is in the `default` Study, as everywhere in FLARE. No method accepts a Study name.
- **Read-only and immutable.** There are no write methods. `DatasetConfig` is frozen, its mappings are read-only views and its tuples are copies, so one component cannot change what another sees.
- **Snapshot.** The configuration is validated as a whole when it is loaded. Every method answers from that content, and `revision` identifies it. The job keeps it for its life, so all of its components and tasks see the same configuration.
- **Fail closed.** An unreadable or invalid configuration raises `DatasetConfigError` from `get_study_dataset_config`; a snapshot is never partly loaded. A site with no configuration is not an error: the snapshot has no datasets.
- **Secrets stay out of the model.** `DatasetConfig` lists which secret keys exist but carries no values. `get_secret` reads the environment variable at call time. Error messages name the dataset and key, never the value.
- **No interpretation.** The API applies the rules and defaults of Section 2 and nothing else. It does not resolve connectors, read sources, check that a URI is reachable or validate `usage` contents; each component validates its own rules.

### 3.3 Implementation

`DatasetConfigAPI` in `nvflare/app_common/dataset_config/api.py` implements the spec. It is built from a configuration dictionary, the content Section 2 describes, and does not depend on where that dictionary came from. Validation, defaults and the frozen objects work on the dictionary; `revision` is a hash of its content.

A loader produces the dictionary. In every launch mode it reads `local/datasets.yaml` as the job sees it (Section 3.4); only it decodes the file format (YAML, with `yaml.safe_load`). Components depend on `StudyDatasetConfig`, so another source, such as a site service, can be added as a loader without changing them.

### 3.4 Loading in each launch mode

In every mode the job reads `local/datasets.yaml` from its own workspace. What differs is how `local/` reaches the job, and that is already decided by each launcher.

| Mode | How the job sees `local/datasets.yaml` |
| --- | --- |
| Process | The site file, in the site workspace |
| Docker | The site file, through the `local/` folder the launcher mounts read-only |
| Slurm, jobs and task workers | The site file, on the site's shared storage (bound read-only in a sandbox) |
| Kubernetes, job pods and task workers | A copy holding only the job's Study, written by the parent into the job's workspace transfer |

**Kubernetes.** A job pod does not see the parent's workspace. The parent sends the job its workspace (`local/` and the job folder) through the workspace transfer, which already leaves out the site files the job must not have, `study_runtime.yaml` and `study_data.yaml`. For `datasets.yaml` the transfer sends the job's part instead of the whole file: the parent loads the site file, validates all of it, and writes a `local/datasets.yaml` with the job's Study entry and the site-wide `usage` settings. In the persistent-workspace profile the transferred `local/` sits on the jobs PVC, and task worker pods mount it read-only, so they read the same part. Task workers receive only the site's environment, so the configuration travels as this file, not in the job's environment.

- No site file: the transfer sends none, and the job sees no datasets.
- An invalid site file: the transfer sends a `local/datasets.yaml` that the API rejects, stating that the site configuration is invalid, so components fail as they would on a broken file instead of seeing no datasets.

The part is written when the job pod downloads its workspace, at launch. The pod and its workers never receive another Study's entries.

**Process, Docker and Slurm.** The job reads the site file, as it can reach the rest of `local/`, and the API returns only the job's Study. As with `study_runtime.yaml` in these modes, the file itself is readable by job code; the API's Study binding is not a security boundary there.

**Fixed for the job's life.** A job loads the configuration once, on first use, and keeps it. An edit to the site file applies from the next job; a running job keeps the configuration it started with.

## 4. Design decisions

| Choice | Set aside | Reason |
| --- | --- | --- |
| One file per site with a `usage` section per component | A separate configuration file for each component | A dataset is described once. Each new component would otherwise add its own file, its own loader and its own copy of the dataset list. |
| A separate file from Study runtime configuration | Adding these fields to `study_runtime.yaml` | Different readers, content, validation and owners; see Section 4.1. |
| Explicit dataset list | Listing the directories under `/data/<study>` | A mount alone does not say what type of source it is or whether any component may use it, and process-launched sites have no mounts. |
| Secret variable names in the configuration | Secret values, or a secret-store reference scheme | Reuses how Study runtime already delivers secrets. The file never holds a secret, so it can be read and reviewed freely. |
| In Kubernetes, the workspace transfer sends the job only its Study's part of the file | Leaving the file out, or passing the part in the job's environment | The transfer already decides what a job pod gets of `local/`, and task worker pods read the job's transferred `local/` but not its environment. Every mode then loads the same file the same way. |

### 4.1 Why a separate file instead of `study_runtime.yaml`

**1. Different reader, different trust level.**

- `study_runtime.yaml` is read only by the launcher on the site's main process, before the job starts. It holds what the job must never see: host paths and storage-volume names, container images, Kubernetes pod templates, Docker settings, Slurm accounts, and which secret store each credential comes from.
- `datasets.yaml` exists to be read by the job, through the API.
- Merging them would mean the job gets the launcher's file, so every container launcher would need to strip those fields before handing it over. A new launcher field added later would leak until someone remembered to strip it too. With two files, the trust boundary sits at the file boundary: the launcher keeps one file and gives the job the other, filtered to its Study.

**2. They describe different things.**

- Study runtime describes how data gets into the job's environment: "mount this volume read-only here".
- The dataset configuration describes what the data is and how components may use it: "this is Parquet, publish its metadata to the catalog, hide column X".
- Some datasets exist in only one of the two. A network database needs no mount, so it has no Study runtime entry, only secret variables. A process-launched site has no mounts at all. Study runtime in process mode allows only environment settings (`env` and `secret_env`), and its parser rejects a `datasets` section. So a process site couldn't describe its datasets in `study_runtime.yaml` at all.

**3. Different validation and change rules.**

- `study_runtime.yaml` is checked strictly for each launcher type. The same entry is valid for Kubernetes and rejected for Slurm.
- The dataset configuration is the same for every launcher, and its `usage` sections are deliberately left unchecked by the API, because each component checks its own.
- One file would tie both sets of rules and versions together. An invalid catalog rule would then fail every job launch for the Study, not just the catalog read.

**4. They're edited by different people, at different times.**

- Infrastructure owners change mounts, images and secret sources.
- Data stewards change which datasets exist and what may be shared.
- Separate files keep those edits apart, which also helps file permissions and change review.

The cost is two files to keep in step for mounted datasets. The design keeps that small: the dataset name is the only link between them, and a mounted dataset always lands at `/data/<study>/<dataset>`.

Secrets are not entered in two places. A secret is configured once, in Study runtime `secret_env` (or, for a process-launched site, the site's environment), which puts its value in an environment variable. `datasets.yaml` holds only that variable's name, such as `secrets: {password: EHR_DB_PASSWORD}`, so a component can ask for "the database password" without knowing where it comes from. Neither file holds a secret value.
