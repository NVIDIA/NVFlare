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

The configuration is the site-owned file `<workspace>/local/datasets.yaml`, next to `study_runtime.yaml`. Like that file, it covers every Study at the site and belongs to the site's parent process. A container job never sees it; the launcher gives the job only its own Study's part (Section 3.4). Jobs cannot supply or replace the configuration: it is never part of a job package, and the API ignores any copy inside the job's app directory.

When an edited file takes effect depends on the launch mode. A container job receives its Study's part once, when it is launched, so an edit applies from the next job. A process job reads the file at each snapshot (Section 3.2), so an edit applies from the next snapshot. Neither needs a restart of the site.

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

### 2.4 What the configuration does not hold

| Concern | Owner |
| --- | --- |
| The Study itself: its definition, description and membership | The federation's Study registry on the server |
| Host source paths, mount modes, images, secret sources | Study runtime configuration of the launcher |
| Secret values | The site's secret store, delivered as environment variables |
| Observed structure (tables, columns, types) | The dataset itself, read by the component that uses it |

A dataset named in both places is linked by its name: Study runtime mounts the host source at `/data/<study>/<dataset>`, and this configuration describes what is there. A dataset declared here whose URI cannot be reached is reported by the component that tries to read it; the API does not check paths or connect to sources.

**A Study entry is this site's data for that Study, not the Study's configuration.** `studies.<study>` holds only `datasets`. The Study's own description and settings live on the server, once for the whole federation; a copy here would drift from it at every site. For the same reason there is no Study-level `description`, `notes` or `options`:

- **Notes.** Use YAML comments. Parsing drops them, so they stay with the site operator and never reach a job; the launcher writes only the parsed Study part into the job.
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
    """Load and validate a snapshot of the job's Study.

    The Study comes from the job metadata in fl_ctx ("default" when the job names none). In a
    container job the file is the Study part the launcher delivered (NVFLARE_DATASET_CONFIG);
    in a process job it is <workspace>/local/datasets.yaml. Raises DatasetConfigError when
    the file is unreadable or invalid, or when NVFLARE_DATASET_CONFIG names a file that does
    not exist. Only a missing <workspace>/local/datasets.yaml yields a snapshot with no datasets.
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

- **Study-bound.** `get_study_dataset_config(fl_ctx)` returns the job's Study only; a job without an explicit Study is in the `default` Study, as everywhere in FLARE. No method accepts a Study name. In container modes this is a security boundary enforced by the launcher, not by the API (Section 3.4).
- **Read-only and immutable.** There are no write methods. `DatasetConfig` is frozen, its mappings are read-only views and its tuples are copies, so one component cannot change what another sees.
- **Snapshot.** One call loads and validates the whole file once. Every method of that snapshot answers from the same content, and `revision` identifies it. A component takes a new snapshot per task. In a process job, that snapshot sees edits to the site file. In a container job, every snapshot reads the Study part delivered at launch, so edits apply from the next job.
- **Fail closed.** An unreadable or invalid file raises `DatasetConfigError` from `get_study_dataset_config`; a snapshot is never partly loaded. So does a missing launcher-delivered file: when `NVFLARE_DATASET_CONFIG` is set, the file it names must exist. The one exception is a missing site file, `<workspace>/local/datasets.yaml`, which yields a snapshot with no datasets: a site that offers no datasets needs no file.
- **Secrets stay out of the model.** `DatasetConfig` lists which secret keys exist but carries no values. `get_secret` reads the environment variable at call time. Error messages name the dataset and key, never the value.
- **No interpretation.** The API applies the rules and defaults of Section 2 and nothing else. It does not resolve connectors, read sources, check that a URI is reachable or validate `usage` contents; each component validates its own rules.

### 3.3 Implementation

`DatasetConfigAPI` in `nvflare/app_common/dataset_config/api.py` implements the spec. It currently supports loading: it reads the file described in Section 2, validates it and returns frozen objects. Only one function in it decodes the file format (YAML, with `yaml.safe_load`), so a different format changes that function alone. Components depend on `StudyDatasetConfig`, so another site-owned source, such as a site service, can replace the file later without changing them.

### 3.4 Isolation between Studies

In container modes (Docker, Kubernetes and containerized Slurm), Study isolation is a security measure: a job can access only its own Study's data. The launchers already enforce it for data. They mount only the job's Study datasets and inject only that Study's `secret_env`. The dataset configuration follows the same boundary:

- **The launcher delivers the job's Study only.** When it starts a job, the launcher in the parent process loads `local/datasets.yaml`, validates it, and extracts the job's Study. It hands that part to the job as a read-only file, and sets `NVFLARE_DATASET_CONFIG` in the job to the file's path. The API in the job reads that file. A job therefore cannot read another Study's dataset names, URIs, options, rules or secret variable names, even by opening files directly.
- **The site file stays out of the job.** The launchers withhold `local/datasets.yaml` from every container job, as the Kubernetes and Slurm workspace transfer already withholds `study_runtime.yaml` and `study_data.yaml`. Kubernetes leaves it out of the job workspace it transfers. Docker binds `local/` entry by entry without it, as it already binds `startup/` file by file. Slurm sandboxes bind a copy of `local/` without it. Until a launcher delivers the job's Study part, a container job therefore sees no datasets.
- **Read-only to the job.** The Study's part is mounted or projected read-only, outside the job's writable workspace: a read-only bind in Docker and Slurm, and a projected volume in Kubernetes. A job cannot loosen its own Study's rules.
- **Invalid configuration fails closed.** If the site file is invalid, the launcher still starts the job and sets `NVFLARE_DATASET_CONFIG`, but delivers no file, so the API raises `DatasetConfigError` and components fail as they would on a broken file. A missing site file is not an error: the launcher delivers a valid Study part with no datasets.
- **Fixed for the job's life.** The Study part is delivered once, at launch. An edit to the site file applies from the next job; a running job keeps the configuration it started with.

Process mode is not isolated, for data or for configuration. A process job is a local process with the site's filesystem and environment. There, the API reads `local/datasets.yaml` directly and returns the job's Study only, as a convenience. Process mode is for development and proof-of-concept sites, as Study runtime already documents.

## 4. Design decisions

Each subsection compares the chosen design (**✓**) with the option not taken.

### 4.1 Separate configuration files for datasets and Study runtime

| | **✓ Two files:** `datasets.yaml` and `study_runtime.yaml` | One file: datasets as a section of `study_runtime.yaml` |
| --- | --- | --- |
| Reader | Components, through the API; the launcher reads only Study runtime | Launcher and components read the same file |
| What a container job receives | Its Study's part of `datasets.yaml` only | The launcher must strip host paths, images, pod templates and secret sources, including any field added later |
| Datasets covered | Mounted, network and process-site datasets | Process-mode Study runtime rejects `datasets`, so process sites cannot describe theirs |
| Validation | Same for every launcher; components validate their own `usage` | Per launcher mode, for both parts |
| Effect of a bad catalog rule | Only components that read the configuration fail | Every job launch for the Study fails |
| Edited by | Data stewards (`datasets.yaml`), infrastructure operators (`study_runtime.yaml`) | Both, in one file |

Cost: a mounted dataset appears in both files, linked only by its name and the `/data/<study>/<dataset>` path (Section 2.3). Secrets are still configured in one place: Study runtime `secret_env` (or the site environment) sets the value; `datasets.yaml` names only the variable.

### 4.2 Dataset configuration owned by the site, not the server

| | **✓ At the site** | On the server (Study registry or submitted job) |
| --- | --- | --- |
| Who decides what a job may read | The site | The server or the job's submitter |
| Paths and connection details | Stay at the site | Published to the federation |
| Mapping to real locations | Direct | Still needs a site-side mapping |

### 4.3 One file for all components, with a usage section per component

| | **✓ One file, `usage.<component>` per dataset** | One file per component |
| --- | --- | --- |
| Dataset described | Once | Once per component, and the copies drift |
| New component adds | A `usage` key | A file, a loader and a dataset list |

### 4.4 An explicit dataset list

| | **✓ Each dataset listed with type and URI** | Discover the folders mounted under `/data/<study>` |
| --- | --- | --- |
| Source type and permitted use | Stated | Unknown from a mount |
| Process sites and network datasets | Covered | Not mounted, so not found |

### 4.5 Secret variable names, not values

| | **✓ Environment variable names** | Secret values, or secret-store references |
| --- | --- | --- |
| Secret in the file | Never; the file can be read and reviewed freely | Values: yes; references: a new scheme to resolve |
| Delivery | Reuses Study runtime `secret_env` | New mechanism |

### 4.6 The launcher delivers the job's Study part

| | **✓ Parent extracts and delivers the job's Study** | Each job reads the whole file and filters by Study |
| --- | --- | --- |
| Other Studies' configuration in the container | Absent | Present; job code can read it |
| Consistent with | How launchers already isolate data and secrets | — |
