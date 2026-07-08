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

For simulator recipes without a literal name, the runner serializes discovery
within that simulator workspace. It accepts a standard printed result path or
the sole changed direct child of the workspace, validates that the root cannot
escape the simulator directory, and persists the discovered name for cleanup,
stall monitoring, and subsequent candidates. Ambiguous or out-of-workspace
results fail closed.
