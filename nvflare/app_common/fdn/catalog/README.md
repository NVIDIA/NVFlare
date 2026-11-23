# Federated data catalog: implementation guide

This guide maps the catalog to its code. The behavior and the reasons for it are in the federated data cataloging
design (`docs/design/federated_data_cataloging_design.md`) and the site dataset configuration design
(`docs/design/dataset_configuration_design.md`); this guide does not repeat their rules.

A catalog read is an ordinary FLARE job, built by `CatalogRecipe`. Its server half, `CatalogController`, asks each
site for its metadata and writes `catalog_results.json` in the job workspace. The job runner saves that workspace to
the job store with the job, and the user downloads it with the job result. The server keeps no catalog of its own.
At each site, `CatalogExecutor` reads the site's dataset configuration and returns metadata visible through the source identity or files the site administrator configured.

## 1. The pieces

| File | Class or function | Role |
| --- | --- | --- |
| `nvflare/recipe/catalog.py` | `CatalogRecipe`, `get_catalog`, `CatalogReadError` | Build and run the job; download and load its result. `get_catalog(env, ...)` is the recipe in one call. |
| `server/controller.py` | `CatalogController` | Sends one `catalog_read` task per site, validates every returned entry with its own limits, and writes `catalog_results.json`. |
| `site/executor.py` | `CatalogExecutor` | Runs the task at a site: takes a Study dataset configuration snapshot, reads selected datasets through their source identities, and validates the result. |
| `site/settings.py` | `parse_settings` | Validate site-wide `usage.catalog` read limits, connector declarations and installed capabilities. |
| `site/connector.py` | `CatalogConnector`, `ConnectorError`, `BUILTIN_CONNECTORS` | The connector interface, and the connector class each built-in dataset `type` selects. |
| `common/schema.py` | `CatalogInfo`, `validate`, `ReadLimits` | The metadata model and its checks. No FLARE or I/O imports. |
| `common/payload.py` | `build_task_payload`, `build_result`, `parse_result`, `build_catalog_results` | The task payload, a site's result, and the `catalog_results.json` document. No FLARE or I/O imports. |
| `simulation.py` | `SimulationCatalogExecutor` | In `SimEnv`, reads each site's `local/datasets.yaml` under `site_config_root/<site>/`. |
| `nvflare/apis/dataset_config_spec.py` | `get_study_dataset_config`, `StudyDatasetConfig`, `DatasetConfig` | The read-only dataset configuration API the executor uses. |
| `nvflare/app_common/dataset_config/api.py` | `load_job_study_config`, `load_study_config` | Its implementation: `load_job_study_config` reads `local/datasets.yaml` in the job's workspace once per job; `load_study_config` reads one file (used by `SimulationCatalogExecutor`). |

Site code never imports server code, and `common/` imports only `common/`; `site/layout_test.py` checks this.

## 2. Life of one read

1. `CatalogRecipe` builds a job with the `fdn_catalog_read` meta property, the controller on the server and the
   executor at the sites. It sets the server's `task_request_interval` to 0.1 s, so sites notice the end of the run
   right after returning their results.
2. `CatalogController.start_controller` refuses a job without the meta property and removes any
   `catalog_results.json` shipped with the job.
3. `control_flow` sends each site one task with that site's optional dataset selection and waits for every task or
   an abort.
4. `_record` parses each reply with `parse_result`, using the server's own limits. A malformed or over-budget entry
   is dropped and its dataset is listed as unavailable; the site's other entries still count. A reply that is not an
   OK result is recorded as `NO_RESULT`.
5. `_write_results` writes `catalog_results.json` atomically: the per-site statuses and every accepted catalog with
   its site. An observation time that is malformed or more than 300 s after `completed_at` is written as null.
6. The job ends. The job runner saves the workspace to the job store, and `get_catalog_results` downloads it and
   loads the file.

## 3. One read at a site

1. `execute` refuses a task from a job without the meta property (`TASK_UNSUPPORTED`).
2. `load_config` takes a dataset configuration snapshot of the job's Study, and `parse_settings` validates the
   site-wide catalog settings. Either failing returns `FAILED` with `CONFIG_UNAVAILABLE`.
3. Select the requested datasets from the Study's configured datasets, or all of them when no selection is given.
   Per-dataset `usage.catalog` is no longer supported: remove old sharing/exclusion rules after configuring source
   metadata permissions. If any dataset retains these settings, the site returns `CONFIG_UNAVAILABLE` without dataset names.
4. Each dataset's `type` selects a built-in or site-declared connector, constructed once per name.
5. The connector reads the source metadata. The executor checks the dataset name, limits advertised operations to
   `installed_operations`, and validates the complete result without publication filtering. Reported structure,
   metadata and lineage are preserved; their visibility is the source administrator's responsibility.
6. A source or validation failure makes the dataset unavailable. A budget stop makes it, and every later selected
   dataset, skipped. An abort returns `TASK_ABORTED`.

Configure a source identity and schema/table projection whose metadata is suitable for the Study. Row-level data
permissions do not necessarily hide schema metadata. File sources require a suitable directory or file projection.
Results are snapshots: a later source restriction affects the next read; it does not retract downloaded results.
