.. _recipe_api:

Recipe API Reference
====================

This page describes the stable Recipe APIs you can rely on when writing,
exporting, and running recipe-based jobs. It also calls out which generated-job
details are internal implementation details rather than user-facing Recipe APIs.

How Recipe Jobs Fit Together
----------------------------

Most recipe scripts use three pieces:

* A concrete recipe, such as ``FedAvgRecipe`` or ``CyclicRecipe``, describes the
  job you want to run. Set its constructor arguments, call Recipe helpers when
  needed, then call ``export()`` or ``execute()``.
* An execution environment, such as ``SimEnv``, ``PocEnv``, or ``ProdEnv``,
  describes where the job runs. Most scripts pass it to ``recipe.execute(env)``.
* A ``Run`` handle, returned by ``recipe.execute(env)``, lets you check the job
  ID, check status, fetch results, or abort the job.

For automation and tooling, ``nvflare recipe list --format json`` returns a
machine-readable recipe catalog. The catalog schema is linked at the end of
this page.

Start With A Concrete Recipe
----------------------------

Start recipe scripts by creating a concrete recipe class. Examples include
``FedAvgRecipe``, ``CyclicRecipe``, ``FedOptRecipe``, ``FedStatsRecipe``, and
the XGBoost recipes listed by ``nvflare recipe list``.

Constructor parameters are public when they are documented in the recipe guide
or returned by:

.. code-block:: shell

   nvflare recipe show <recipe-name> --format json

The base ``Recipe`` type defines behavior shared by concrete recipes. Most
scripts do not instantiate the base type directly. The public constructor
surface is the documented constructor of each concrete recipe.

Recipe Execution
----------------

All concrete recipes support the common execution surface:

``recipe.export(job_dir, server_exec_params=None, client_exec_params=None, env=None)``
   Export the current recipe definition as a deployable NVFlare job directory.
   If ``env`` is supplied, the recipe can apply environment-specific processing
   before export. Execution parameter dictionaries become part of the job
   definition and must not contain secret values; see :ref:`recipe_secrets`.

``recipe.execute(env, server_exec_params=None, client_exec_params=None)``
   Execute the current recipe definition through an execution environment and
   return a ``Run`` handle. Execution parameter dictionaries must not contain
   secret values. When the Python process receives the Recipe export flags,
   ``execute`` exports instead of submitting:

   .. code-block:: shell

      python job.py --export --export-dir /tmp/nvflare/job

``recipe.run(env, server_exec_params=None, client_exec_params=None)``
   Submit directly through the environment and return a ``Run`` handle. Most
   user examples should use ``execute`` so export flags continue to work.
   Execution parameter dictionaries must not contain secret values.

Recipe helpers mutate the recipe object. Call helpers before ``export()`` or
``execute()`` for the job you are about to create. After a job has been
exported or submitted, later helper calls do not change that already-created
job; they only affect future exports or runs from the same recipe object.

Recipe Customization Helpers
----------------------------

Recipes already provide public helpers for common generated-job customization.
Use these helpers instead of editing generated configuration files by hand.

Config helpers:

``recipe.add_client_config(config, clients=None)``
   Add top-level generated client app configuration parameters. If ``clients``
   is omitted, the config applies to all generated client apps. The dictionary
   is stored in clear text and must not contain secret values.

``recipe.add_server_config(config)``
   Add top-level generated server app configuration parameters. The dictionary
   is stored in clear text and must not contain secret values.

These config helpers are for documented top-level configuration knobs such as
timeouts and streaming chunk sizes. They are not a component-placement API; do
not use them to replace generated ``components``, ``workflows``, or
``executors`` lists.

File packaging helpers:

``recipe.add_client_file(file_path, clients=None)``
   Bundle a file or directory into generated client app packages.

``recipe.add_server_file(file_path)``
   Bundle a file or directory into the generated server app package.

Helpers that accept ``clients`` target specific generated client apps. This
requires per-site client apps: construct the recipe with the
``per_site_config`` constructor argument on recipes that support it, and each
name in ``clients`` must match an existing per-site client app. With the
default all-clients topology, targeted calls raise an error rather than
silently dropping the change from the generated job, and unknown site names
raise an error rather than deploying a bare app to that site. Calling
``set_per_site_config`` after construction records the configuration for
``configured_sites()`` but does not yet rebuild an existing all-clients app
into per-site apps; recipes will interpret helper-provided per-site config as
follow-up work.

Filter helpers:

``recipe.add_client_input_filter(filter, tasks=None, clients=None)``
   Add a client-side input filter for incoming task data from the server.

``recipe.add_client_output_filter(filter, tasks=None, clients=None)``
   Add a client-side output filter for outgoing task results to the server.

``recipe.add_server_input_filter(filter, tasks=None)``
   Add a server-side input filter for incoming task results from clients.

``recipe.add_server_output_filter(filter, tasks=None)``
   Add a server-side output filter for outgoing task data to clients.

Shared helpers:

``recipe.add_decomposers(decomposers)``
   Register decomposers on the generated server and client app packages.

``recipe.enable_log_streaming(*file_names)``
   Add the default Recipe log-streaming components. If no file names are given,
   the recipe streams ``log.json``.

Utility helpers:

``add_experiment_tracking(recipe, tracking_type, tracking_config=None, client_side=False, server_side=True, clients=None)``
   Add supported experiment tracking receivers such as TensorBoard, MLflow, or
   Weights & Biases. With ``client_side=True``, ``clients`` limits which sites
   receive the client-side receiver; call once per site with different
   ``tracking_config`` values for per-site tracking destinations. Keep tracking
   credentials in the executing site's environment or mounted secret files;
   never put them in ``tracking_config``.

``add_cross_site_evaluation(recipe, submit_model_timeout=600, validation_timeout=6000, participating_clients=None)``
   Add cross-site evaluation to a training recipe when the recipe/framework
   supports it.

Per-Site And Metadata Helpers
-----------------------------

For NVFlare 2.9, the public Recipe configuration surface also includes:

``set_per_site_config(recipe, config)``
   Provide site-keyed, recipe-specific configuration. Each concrete recipe
   interprets the site dictionaries for its own workflow. Nested values become
   part of the job definition and must not contain secret values.

``recipe.configured_sites()``
   Return top-level site names from helper-provided per-site config when
   present. For backward compatibility, recipes may return site names from
   legacy constructor ``per_site_config`` when available. This method does not
   indicate that sites are connected or replace the execution environment.

``set_recipe_meta(recipe, key, value)``
   Set selected generated job metadata by ``JobMetaKey``. The accepted keys are
   exactly the members of
   :data:`nvflare.apis.job_def.USER_SETTABLE_JOB_META_KEYS`. Not settable
   through this helper: runtime/submission metadata managed by NVFlare
   internals, keys with dedicated constructor fields (``min_clients``,
   ``mandatory_clients``), and ``study``, which the server assigns from the
   admin session at submission. See the Recipe Metadata section of the Recipe
   guide for per-key value shapes and examples. Metadata is stored in clear
   text and must not contain secret values.

Repeated metadata calls for the same key replace that key's previous helper
value. Different metadata keys accumulate. Keys that overlap a dedicated
constructor field are rejected rather than given precedence; for accepted
keys, the helper value is what appears in the generated ``meta.json``
(metadata merges last). When ``RESOURCE_SPEC`` overrides per-site resource
specs on the generated job, a warning is emitted for specs already registered
when the helper is called, but specs added afterwards are overridden without
one.

.. _recipe_secrets:

Keeping Secrets Out Of Recipe Parameters
----------------------------------------

Recipe parameters are part of the job definition, not a secret transport.
Values supplied through recipe constructors and mutating helpers can be
serialized in clear text into generated configuration files. This includes
``train_args``, ``task_args``, ``eval_args``, ``per_site_config``, task data and
metadata, server/client config override dictionaries, execution parameters,
recipe metadata, tracking configuration, and dictionaries passed to
``add_client_config`` / ``add_server_config``. These values must **never**
contain actual passwords, API keys, access tokens, private keys, or other
credentials.

This contract applies to nested strings too, not just top-level parameter
values. A secret's environment-variable name, a reference placeholder, or the
path to a mounted secret file is configuration and can be supplied. The secret
value itself must remain at the executing site.

To catch mistakes, recipes scan their parameters with heuristics (well-known
token formats, password-like flag and key names, high-entropy strings) and
emit a ``nvflare.recipe.secrets.PotentialSecretWarning`` when a value looks
like an actual secret. ``recipe.export()`` additionally scans the generated
config files of the exported job. The scan is best-effort: it neither finds
every possible credential nor makes a supplied value safe. Absence of a
warning does not prove that a parameter is safe. Investigate each warning and
keep any actual secret at the site, using a reference only at a supported
runtime boundary; use the standard
``warnings.filterwarnings`` machinery only after establishing that a finding
is a false positive. NVFlare emits detector warnings from a synthetic source
location so Python's warning formatter cannot echo a user source line that
contains the flagged value. A valid reference in a known unsupported parameter
emits ``UnsupportedSecretRefWarning`` instead of being silently accepted.

There are two supported ways to make a secret available at runtime:

* **Read it from the site environment (preferred).** Set an environment
  variable or mount a secret file (for example, a Kubernetes Secret volume) on each site,
  and read it inside your training script with ``os.environ`` or by opening
  the mounted file. Nothing secret ever enters the job definition. Passing a
  *path* to a mounted secret file in a recipe parameter is fine -- the path
  is not the secret.

* **Use a secret reference at a supported runtime boundary.** Put a placeholder
  in a supported recipe value instead of the actual secret. Use ``secret_ref``
  for an environment variable and ``secret_file_ref`` for a file containing
  the secret:

  .. code-block:: python

     from nvflare.recipe.secrets import secret_file_ref, secret_ref

     recipe = FedAvgRecipe(
         ...,
         train_args=f"--epochs 5 --api-key {secret_ref('MY_API_KEY')}",
     )
     recipe.add_client_config(
         {"service_password": secret_file_ref("/var/run/secrets/service/password")}
     )

     # In site-side component code:
     from nvflare.utils.configs import get_client_config_value

     service_password = get_client_config_value(fl_ctx, "service_password")

  The exported job contains only ``${secret:MY_API_KEY}`` and
  ``${secret:file:/var/run/secrets/service/password}``. References resolve only
  at these explicit runtime boundaries:

  * Command arguments consumed by NVFlare's task script runner or subprocess
    launcher. This includes recipe ``train_args``, ``task_args``, ``eval_args``,
    and ``script_args`` when those arguments use these runners. NVFlare expands
    ordinary configuration variables first, tokenizes the command, and then
    resolves each reference immediately before the script or process starts.
    Resolving after tokenization keeps a secret containing spaces in one
    argument.

  * Values explicitly read from a runtime job JSON file with
    ``get_job_config_value``, ``get_client_config_value``, or
    ``get_server_config_value``. Typically, a recipe adds a top-level value with
    ``add_client_config`` or ``add_server_config`` and site-side code reads it
    through the matching specialized getter. References in nested string values
    resolve recursively when the value is read; dictionary keys are not resolved.
    These raw-file helpers do not expand ordinary placeholders such as
    ``{SITE_NAME}``, so do not combine those placeholders in a value consumed
    this way.

  Arbitrary component constructor arguments, job metadata, packaged custom
  files, and other job artifacts keep references as placeholders and are not
  secret delivery mechanisms. Read the site environment or mounted file inside
  user code for those cases. In particular, Flower ``extra_env`` and
  ``run_config`` values do not support secret references.

  If an environment variable or file is unavailable, the config getter or
  script/process launch fails with an error that identifies the missing
  reference but never includes a secret value. Resolved values exist only in
  runtime memory and are not written back to generated job configuration.

Environment variables must be set in the environment of the server or client
job process that uses the reference. For native-process POC and production
deployments, this can be the environment inherited by that process. Docker and
Kubernetes job containers do not automatically inherit arbitrary host shell
variables or host mounts: configure the launcher/container spec to inject the
variable or mount the file into the actual job container. A referenced file
must exist at the same whitespace- and brace-free path inside the consuming process or
container and should be readable only by that identity.

For Kubernetes, project a Secret key into the job container as an environment
variable or mounted file, then use the corresponding reference helper. The
Recipe API does not query a Kubernetes Secret directly by Secret name and key.

Note that with external-process execution, command-line arguments (including
resolved secret references) are visible to local process listings on the
executing site, as with any command-line tool. Reading the secret from the
environment or mounted file inside the training script avoids this too. Code
and configured components must also avoid printing resolved values: reference
resolution keeps values out of the exported job but cannot sanitize output
produced by user code.

Threaded simulation runs in-process client scripts concurrently and shares the
process-global ``sys.argv`` and environment. Use external-process execution for
secret-bearing command arguments in the simulator, or preferably read a
site-local secret directly inside code, rather than relying on in-process
per-client argument isolation.

Execution Environments
----------------------

Most scripts pass an environment to ``recipe.execute(env)``. Built-in
environments include ``SimEnv``, ``PocEnv``, and ``ProdEnv``.

Most recipe scripts do not call environment methods directly. They are listed
here for anyone implementing an execution environment.

``deploy(job) -> str``
   Deploy the job produced by the recipe and return a job ID.

``get_job_status(job_id) -> Optional[str]``
   Return the current job status when supported.

``abort_job(job_id) -> None``
   Request that a running job stop.

``get_job_result(job_id, timeout=0.0) -> Optional[str]``
   Return the result workspace path when the job has completed, or ``None`` if
   the result is not ready or not supported.

``stop(clean_up=False) -> None``
   Stop environment resources and optionally clean up temporary workspaces.

These methods are primarily for environment implementers. User code
should prefer ``recipe.execute(env)`` over calling an environment directly.

Run Handles
-----------

``recipe.execute`` and ``recipe.run`` return a ``Run`` object when the job is
submitted. ``Run`` exposes:

``run.get_job_id()``
   Return the environment job ID.

``run.get_status()``
   Return the latest status when available.

``run.get_result(timeout=0.0, clean_up=True)``
   Wait for the result, cache the final status/result, stop the environment,
   and return the result workspace path when available.

``run.abort()``
   Request that the environment abort the running job.

What You Can Rely On
--------------------

You can rely on documented concrete recipe constructor parameters, documented
Recipe methods, documented helper functions, ``ExecEnv`` methods, ``Run``
methods, and documented JSON fields from ``nvflare recipe`` commands.

The following are internal details and should not be used in recipe scripts:

* private attributes, including attributes whose names start with ``_``;
* generated job internals and nested generated job objects;
* the internal generated job attribute of a recipe;
* direct mutation of generated metadata dictionaries;
* generated deploy-map internals;
* internal fields used by Recipe helpers.

If a workflow needs arbitrary component placement that is not covered by a named
Recipe helper, use the lower-level Job API workflow or add a new named Recipe
helper for the repeated pattern.

Recipe Catalog JSON
-------------------

``nvflare recipe list --format json`` returns a machine-readable list of the
available recipes. The structure of a successful response is stable and
described by
:download:`recipe_catalog.schema.json <../../schemas/recipe_catalog.schema.json>`;
tools that discover recipes can rely on the fields documented there.

``nvflare recipe list --schema`` describes the command-line arguments for the
command. It is not the same as the catalog output schema.
