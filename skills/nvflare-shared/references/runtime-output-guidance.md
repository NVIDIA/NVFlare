# Runtime And Output Location Guidance

Use this reference when choosing generated source layout, export locations,
simulation workspaces, and validation output locations.

## Generated Source Layout

When the current project or job source root is writable, use it as the default
generated FLARE source location. Add or update standard source files such as
`client.py`, `job.py`, `model.py`, `requirements.txt`, and small config files
beside the existing training files unless the user explicitly asks for another
target directory.

Do not create an extra wrapper folder such as `nvflare_jobs/<job_name>/` by
default. Preserve original training files such as `train.py` as references
instead of renaming or overwriting them unless the user explicitly asks for an
in-place rewrite.

## Read-Only Source Roots

Do not assume the source root is writable; it may be owned by another user or
mounted read-only. Check writability before writing generated source, using
`os.access(path, os.W_OK)` or a small removable probe write when metadata is
insufficient. Do not learn this from a failed generated-file write, and do not
retry after a denial.

If the source root is read-only, read code and data from it but write nothing
into it. Generate job source to a writable directory: a user-provided path when
available, otherwise `/tmp/nvflare/job_config/<job_name>/src/`. Point the job at
the original data by absolute path, for example through `train_args`, and report
both the read-only source root and generated job source location.

## Runtime And Export Outputs

Keep generated source and runtime artifacts separate. The source root may hold
generated source files, but it must not be the default root for exported jobs,
simulation workspaces, generated model artifacts, caches, or validation results.
Use explicit runtime locations under `/tmp/nvflare/` unless the user provides
another path. `.gitignore` entries or NVFLARE current-directory defaults are not
user approval to place runtime, export, or result artifacts under the source
root.

Recommended defaults:

- exported job config: `/tmp/nvflare/job_config/<job_name>/`;
- simulation workspace: `/tmp/nvflare/workspaces/<job_name>/`;
- generated validation outputs or evaluation records:
  `/tmp/nvflare/results/<job_name>/`.

If the user asks to keep runtime artifacts in the project workspace, do so and
report the chosen location. Otherwise keep generated runtime output separate
from source edits so code review and agent context are not polluted by simulator
or artifact files.

Before declaring conversion complete, inspect generated entry points and report
whether their default workspace, export, and result locations are outside the
source root. If any default remains project-local, either move it to the runtime
location convention above or state the explicit user instruction that requires
the project-local location.

## Export Directory

For generated `job.py` files that support NVFLARE export, use the exported job
config location selected above unless the user provides an export directory. The
exported job root is the exact directory later passed to
`nvflare job submit -j`.
