# Federated data catalog: implementation guide

This guide maps the catalog to its code. The behavior and the reasons for it are in the federated data cataloging
design (`docs/design/federated_data_cataloging_design.md`) and the site dataset configuration design
(`docs/design/dataset_configuration_design.md`); this guide does not repeat their rules.

A catalog read is an ordinary FLARE job, built by `CatalogRecipe`. Its server half, `CatalogController`, asks each
site for its metadata and writes `catalog_results.json` in the job workspace. The job runner saves that workspace to
the job store with the job, and the user downloads it with the job result. The server keeps no catalog of its own.
At each site, `CatalogExecutor` reads the site's dataset configuration and returns only what the site shares.

## 1. The pieces

| File | Class or function | Role |
| --- | --- | --- |
| `nvflare/recipe/catalog.py` | `CatalogRecipe`, `get_catalog`, `CatalogReadError` | Build and run the job; download and load its result. `get_catalog(env, ...)` is the recipe in one call. |
| `server/controller.py` | `CatalogController` | Sends one `catalog_read` task per site, validates every returned entry with its own limits, and writes `catalog_results.json`. |
| `site/executor.py` | `CatalogExecutor` | Runs the task at a site: takes a dataset configuration snapshot, applies the catalog rules, reads each shared dataset through its connector, and validates the result. |
| `site/rules.py` | `parse_rule`, `parse_settings`, `apply_policy` | Validate a dataset's `usage.catalog` rules and the site-wide `usage.catalog` settings; apply a dataset's rules to its catalog. |
| `site/connector.py` | `CatalogConnector`, `ConnectorError`, `BUILTIN_CONNECTORS` | The connector interface, and the connector class each built-in dataset `type` selects. |
| `common/schema.py` | `CatalogInfo`, `validate`, `ReadLimits` | The metadata model and its checks. No FLARE or I/O imports. |
| `common/payload.py` | `build_task_payload`, `build_result`, `parse_result`, `build_catalog_results` | The task payload, a site's result, and the `catalog_results.json` document. No FLARE or I/O imports. |
| `simulation.py` | `SimulationCatalogExecutor` | In `SimEnv`, reads each site's `local/datasets.yaml` under `site_config_root/<site>/`. |
| `nvflare/apis/dataset_config_spec.py` | `get_study_dataset_config`, `StudyDatasetConfig`, `DatasetConfig` | The read-only dataset configuration API the executor uses. |
| `nvflare/app_common/dataset_config/api.py` | `load_study_config` | Its implementation over `local/datasets.yaml`, or the file `NVFLARE_DATASET_CONFIG` names. |

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
3. Every dataset's `usage.catalog` rules are parsed. A dataset without rules or with `share: false` is not shared
   and never named. Invalid rules make a selected dataset unavailable.
4. For each selected, shared dataset, the connector is the one its rules name, else the one its `type` selects.
   Connectors are built once per name, from built-in classes or classes the site declares.
5. The connector reads the dataset. The executor checks the dataset name, applies the rules, removes lineage to
   datasets the site does not share, and validates the result against the remaining budget.
6. A source or validation failure makes the dataset unavailable. A budget stop makes it, and every later shared
   dataset, skipped. An abort returns `TASK_ABORTED`.
