# Process Mode Study Runtime Configuration

Process launchers apply the selected study's `env` and `secret_env` from the site's
`local/study_runtime.yaml`. Both server and client jobs inherit this behavior from `ProcessJobLauncher`.
The goal is configuration convenience and consistency with Docker, with full parent environment inheritance
preserved. This feature does not isolate studies.

## Configuration

```yaml
format_version: 2
studies:
  lung-cancer:
    env:
      DB_HOST: db.internal
      TRAINING_DATA: /srv/studies/lung-cancer/training
      DB_CA_FILE: /srv/studies/lung-cancer/secrets/ca.crt
      OPTIONAL_SETTING: ""
    secret_env:
      DB_PASSWORD: {source: LUNG_DB_PASSWORD}
```

`env` supplies literal scalar values. Numbers become strings, booleans become `true` or `false`, and empty
strings are allowed. No shell expansion takes place. `secret_env` resolves values from the parent environment;
its optional `key` is accepted but ignored, allowing existing Docker entries to be reused. A missing source
fails launch, even if the destination exists in the inherited environment. A present empty source is valid.

Only `env` and `secret_env` are supported. Reject container, dataset, secret-mount, and other launcher-specific
settings, including empty mappings. Applications can read existing host paths from environment variables.
Process mode does not create mounts, copy files, change permissions, or translate hard-coded paths.

## Launch behavior

1. Locate `local/study_runtime.yaml` under `Workspace.get_root_dir()`. If absent, retain the existing behavior.
   If present alongside `local/study_data.yaml`, reject the ambiguous configuration. A legacy file alone remains
   unused, as before.
2. Load the file with the existing strict parser in `process` mode. Select the study through
   `get_job_meta_study`, which defaults missing study metadata to `default`.
3. Resolve the selected study with the existing resolver. An empty file mapping or missing study entry adds
   no settings; retain the existing warning behavior. Invalid configuration anywhere in the file fails launch,
   but only the selected study's secret sources need to exist.
4. Resolve secrets from the launcher's copy of the parent environment before applying study overrides. Overlay
   `env` and resolved `secret_env`, then apply the existing custom Python path and bootstrap credentials.
5. Launch through the existing `spawn_process` path. Never mutate the parent environment with study settings.

Each launch has its own environment dictionary. The precedence is inherited environment, study settings,
then launcher-owned settings. Both client and server jobs use their own site's configuration and parent
variables; secrets are not distributed between sites by this feature.

## Validation and reloading

For process mode, destination and secret source names must be valid POSIX environment variable names.
Reuse the parser's reserved-name checks for both: `PYTHONPATH`, workspace-transfer variables, and
`JobProcessEnv.ALL` cannot be assigned or remapped. Reject duplicate destinations across `env` and
`secret_env`. Configured literal values must be valid UTF-8 text without NUL; inherited secret values are
passed through unchanged. These process rules do not change Docker or Kubernetes validation.

Read the file for each launch. Operators should replace it atomically. Running jobs retain their environment;
subsequent jobs use the new settings. Changing a shell or service-manager environment does not change a
running parent's environment, so secret source rotation normally requires restarting the parent service.
Missing sources and configuration errors fail before creating a child. Diagnostics name settings rather than
resolved secret values; the launcher does not log the environment or put study secrets in command arguments.

On upgrading, an existing `study_runtime.yaml` becomes active for process jobs. Operators must review it:
supported settings start applying and unsupported container or mount settings fail launch. This activation is
called out in the admin guide. Custom launchers that override `launch_job` must integrate support themselves.

## Relationship to Docker

Both modes use the same study selection, literal values, and parent-variable-to-job-variable mapping.
Docker starts with its configured container environment; process mode starts with the entire parent
environment. Source variables and other studies' secrets therefore remain inherited in process mode.
Environment filtering and sandboxing are outside this feature's scope. Use a suitably configured Docker
deployment when jobs require isolation.

Docker mounts host resources at container paths; process jobs access existing host paths under their OS
identity's permissions. Application code can be shared when it reads the paths from environment variables.
See [Study Runtime Configuration](study_runtime_config.md) for the other launchers' semantics.

## Implementation and verification

The implementation extends only the shared process launcher and existing study runtime parser, without
adding a provider interface, launcher hierarchy, constructor option, or schema version.

Parser tests cover the supported field subset, source-only and Docker-compatible secret references, empty
values, invalid names and values, duplicate destinations, and reserved variables. Launcher tests cover both
client and server classes, default and named studies, precedence, unchanged parent state, missing-source and
configuration failures before spawn, independent concurrent launches, and configuration reloads. Real-child
tests verify environment delivery through the process spawning utility, including its subprocess fallback.
The tests explicitly retain inherited secret sources and use synthetic values without logging them.
