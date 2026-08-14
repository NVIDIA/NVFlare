# Job Import Contract

The importer parses `job.py` without executing it. When the job selects its
training recipe through CLI control flow, pass the same fixed selector to the
runner through `--base-args`, for example:

```bash
python "$RUNNER" initialize ./job.py --base-args "--mode training" [other options]
```

The importer applies recognized argparse overrides while resolving simple
`if`/`elif` branches, and the runner preserves those arguments for every
baseline and candidate. Evaluation-only, statistics-only, and unsupported
nested-application recipes stop during import with an actionable
`import.support.reason`; they must not start a baseline.

Every simulator trial receives an isolated temporary workspace. For recipes
without a literal name, the runner accepts a standard printed result path or
the sole changed direct child of that trial workspace, validates that the root
cannot escape the workspace, and persists the discovered name for cleanup,
stall monitoring, and subsequent candidates. Ambiguous or out-of-workspace
results fail closed.

Simulator child processes — baseline and candidate runs and the runner's
capability probe — do not inherit the full host environment. The runner
forwards a fixed runtime allowlist (interpreter and virtualenv/conda paths,
`HOME` and temp directories, locale, proxy and CA-bundle settings, threading
limits, CUDA/NVIDIA device visibility, and dynamic-library paths), so host
secrets never reach campaign-executed code and host-environment drift cannot
silently change candidate behavior. Declare job-specific variables such as
`DATASET_DIR` by name in `environment.simulator_env_passthrough` in
`autofl.yaml`; values are read from the host at run time and never stored.
Generated configs include `simulator_env_passthrough: []`, a missing field
means no extra variables, and entries that are not valid environment-variable
names fail closed.
