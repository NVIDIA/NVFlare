# Runtime And Output Location Guidance

Use this reference when choosing generated source layout, export locations,
simulation workspaces, and validation output locations.

## Private Runtime Root

Do not derive a runtime directory from a job name and do not reuse a
predictable path such as `/tmp/nvflare/<job_name>`. Before writing runtime
artifacts:

1. Select the platform's canonical trusted temporary base (for example,
   `tempfile.gettempdir()`) or a user-provided secure runtime base. Do not
   blindly follow a source-provided temp path.
2. Create a per-user NVFLARE root owned by the current user with mode `0700`.
   If that root already exists, inspect it with `lstat`: it must be a real
   directory, owned by the current user, with no group/world permissions.
   Reject it otherwise. Starting at the trusted base, reject symlinked path
   components; do not use `resolve()` as a substitute for this check.
3. Atomically create a new unpredictable run directory inside that root with
   `tempfile.mkdtemp(prefix="run-", dir=private_root)` or an equivalent secure
   primitive. Keep it `0700`; never accept or reuse a pre-existing run
   directory.
4. Create the run's `src`, `job_config`, `workspace`, `results`, `logs`, and
   `reports` subdirectories with exclusive directory creation and mode `0700`.
   Reject an existing or symlinked component. For files, use no-follow and
   exclusive creation where the platform supports them, then verify ownership
   and type.

Treat the job name as untrusted display metadata only. Sanitize it to a short
label containing a conservative character set such as letters, digits, dot,
underscore, and hyphen, but do not use even the sanitized label as a path
component, shell argument, or security identifier.

The run directory is unpredictable, but its contents stay discoverable. At
creation time, record a run ID, the sanitized display label, and the exact
absolute paths for the run root, source, exported job, workspace, results,
reports, and logs in both harness/controller state and a `run-manifest.json`
inside the run root. Write the manifest atomically, omit secrets, and include
the same exact paths in the final report. Use the recorded path for later RCA;
do not create a mutable `latest` symlink or reconstruct a path from the job
name.

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
available and verified safe, otherwise the recorded `<run-root>/src/`. Point
the job at the original data through a configurable `train_args` value rather
than a path hardcoded in the generated code — see the "Data Location" rule in
`conversion-workflow.md` — and report both the read-only source root and the
exact generated job source location.

## Runtime And Export Outputs

Keep generated source and runtime artifacts separate. The source root may hold
generated source files, but it must not be the default root for exported jobs,
simulation workspaces, generated model artifacts, caches, or validation results.
Use explicit locations under the newly created private run root unless the user
provides another secure path. Apply the same ownership, permission, no-follow,
and exclusive-creation checks to an override; ask or fail closed if its safety
cannot be established. `.gitignore` entries or NVFLARE current-directory
defaults are not user approval to place runtime, export, or result artifacts
under the source root.

Defaults inside one recorded run root:

- exported job config: `<run-root>/job_config/`;
- simulation workspace: `<run-root>/workspace/`;
- generated validation outputs or evaluation records:
  `<run-root>/results/`;
- diagnostic logs: `<run-root>/logs/`;
- generated reports: `<run-root>/reports/`.

If the user asks to keep runtime artifacts in the project workspace, do so and
report the exact chosen location after applying the safe-path checks above.
Otherwise keep generated runtime output separate from source edits so code
review and agent context are not polluted by simulator or artifact files.

Before declaring conversion complete, inspect generated entry points and report
whether their default workspace, export, and result locations are outside the
source root. If any default remains project-local, either move it to the runtime
location convention above or state the explicit user instruction that requires
the project-local location.

## Export Directory

For generated `job.py` files that support NVFLARE export, use the exported job
config location selected above unless the user provides an export directory. The
exported job root is the exact directory later passed to
`nvflare job submit -j`. Record that exact directory in the run manifest and
final report; do not reconstruct it from the display job name.
