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

The importer resolves the optimization direction from an explicit
`key_metric_mode`, a same-metric `stop_cond`, or NVFLARE's default `max`, and
records that provenance in `autofl.yaml`. Declare raw loss-like metrics with
`key_metric_mode="min"`. If a requested metric differs from the job's
`key_metric`, `mutation_schema.yaml` must name the requested and optimization
metrics and may declare their mode; otherwise the bridged metric uses NVFLARE's
default `max` independently of the job key metric's direction.
A schema cannot override the direction of the job's own key metric. The
importer preserves that native direction separately as `job_key_metric_mode`
so candidate validation can detect changes hidden by an alternate metric
bridge. A custom `model_selector` makes direction unresolved because it
supersedes `key_metric_mode` and its behavior cannot be imported statically.
A job constructor that passes `**kwargs` also leaves `key_metric` unresolved
unless that keyword is explicit in the call. Its direction remains unresolved
unless `key_metric_mode`, `model_selector`, and `stop_cond` are all explicit,
because any omitted value may be supplied by the splat. Rewrite the call to
spell out these safety-critical keywords before initializing. A mutation-schema
bridge may declare the direction of a differing requested metric, but the job
key metric's own direction can only be declared explicitly in `job.py`.

For a new campaign, import and admission complete in memory before the runner
creates campaign files or acquires the workspace lock. An obvious
lower-is-better metric that relies only on the implicit `max` default is
rejected with `AUTOFL_METRIC_DIRECTION_CONFLICT`; set the job's
`key_metric_mode="min"` and initialize again. Unknown custom metrics retain
NVFLARE's `max` default.

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
