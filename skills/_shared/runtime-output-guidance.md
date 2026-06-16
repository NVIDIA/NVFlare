# Runtime And Output Location Guidance

Use this reference when choosing generated source layout, export locations,
simulation workspaces, and validation output locations.

## Generated Source Layout

Use the current project or job source root as the default generated FLARE source
location. Add or update standard source files such as `client.py`, `job.py`,
`model.py`, `requirements.txt`, and small config files beside the existing
training files unless the user explicitly asks for another target directory.

Do not create an extra wrapper folder such as `nvflare_jobs/<job_name>/` by
default. Preserve original training files such as `train.py` as references
instead of renaming or overwriting them unless the user explicitly asks for an
in-place rewrite.

## Runtime And Export Outputs

Do not put exported jobs, simulation workspaces, generated model artifacts, or
temporary vocab/cache files in the original source root by default. Use explicit
runtime locations under `/tmp/nvflare/` unless the user provides another path.

Recommended defaults:

- exported job config: `/tmp/nvflare/job_config/<job_name>/`;
- simulation workspace: `/tmp/nvflare/workspaces/<job_name>/`;
- generated validation outputs or evaluation records:
  `/tmp/nvflare/results/<job_name>/`.

If the user asks to keep runtime artifacts in the project workspace, do so and
report the chosen location. Otherwise keep generated runtime output separate
from source edits so code review and agent context are not polluted by simulator
or artifact files.

## Export Directory

For generated `job.py` files that support NVFLARE export, default
`--export-dir` to `/tmp/nvflare/job_config/<job_name>` unless the user provides
an export directory. The exported job root is the exact directory later passed
to `nvflare job submit -j`.
