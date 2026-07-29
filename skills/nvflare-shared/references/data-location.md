# Data Location

This reference is the canonical data-location policy for NVFLARE conversion
skills. Load it when source training reads a file or directory.

## Preserve Site-Local Data

Pass each data location into the generated client through configurable
`train_args` or `per_site_config`. Keep the client free of fixed site paths and
let each deployment site override the value.

Point at the original dataset. Do not copy project or private site data into the
generated job with `add_client_file(...)` or an equivalent packaging API.
Packaging is acceptable only for an explicitly requested synthetic or public
validation fixture; identify that artifact as validation-only.

## Resolve Source-Relative Paths

Interpret a source-relative path such as `data/`, `./train.csv`, or
`../datasets/train.csv` relative to the original source root that made the
training program valid. Resolve it in `job.py` before recipe construction, then
pass the resolved value as a configurable argument:

```python
source_root = Path(__file__).resolve().parent
data_dir = (source_root / "data").resolve()
train_args = shlex.join(["--data-dir", str(data_dir)])
```

This example assumes generated `job.py` is beside the original training source,
which is the default writable-source layout. If the user selected another
generated-source destination, retain the inspected source root as an explicit
job-builder input instead of treating that destination as the source root.

Preserve any additional training arguments when building the final quoted
argument string. A shared absolute value is acceptable for local simulation;
real sites use `per_site_config` when their locations differ.

For normal project or site data, do not pass a bare relative value and rely on
the simulator or client process working directory. Do not repair that ambiguity
in `client.py` by resolving the value against `Path(__file__).parent`: inside an
exported or simulated job, `__file__` identifies the packaged app, not the
original source root. The client must consume the configured path as given. An
explicitly packaged validation-only fixture may intentionally use its packaged
location.

## Validate The Effective Path

Before full simulation:

- parse the final argument string with the generated client's real parser;
- confirm the local-simulation path resolves to the original dataset and exists;
- inspect exported client configuration for the same configurable argument;
- confirm no project/private dataset was copied into an app's `custom/` folder;
- use a fresh runtime workspace so a prior working directory or stale packaged
  file cannot hide a path error.

Report the local-simulation value and explain that production sites must
override it when their paths differ.
