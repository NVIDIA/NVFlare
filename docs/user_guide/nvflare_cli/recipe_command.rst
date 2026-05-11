.. _recipe_command:

#########################
Recipe Command
#########################

``nvflare recipe`` lists built-in Job Recipe API recipes and shows structured
metadata that agents and scripts can use to choose a recipe.

***********************
Command Usage
***********************

.. code-block:: none

   nvflare recipe -h

   usage: nvflare recipe [-h]  ...

   recipe subcommands:
     list      list available recipes (default)
     show      show structured metadata for a recipe

****************
List Recipes
****************

Use ``nvflare recipe list`` to display the available built-in recipes:

.. code-block:: shell

   nvflare recipe list

In text mode, the command prints ``Loading installed recipe catalog...`` before
importing and inspecting installed recipe metadata. This is a local Python
environment operation; it does not connect to a FLARE server.

Filter by framework:

.. code-block:: shell

   nvflare recipe list --framework pytorch

``--framework`` is a shorthand for ``--filter framework=<framework>``.

Supported framework filter values:

- ``core``
- ``numpy``
- ``pytorch``
- ``tensorflow``
- ``sklearn``
- ``xgboost``

Filter by recipe metadata:

.. code-block:: shell

   nvflare recipe list --filter framework=pytorch --filter algorithm=fedavg
   nvflare recipe list --filter privacy=homomorphic_encryption

Supported ``--filter`` keys:

- ``framework``
- ``privacy``
- ``algorithm``
- ``aggregation``
- ``state_exchange``

``--filter`` is repeatable. Filters for different keys are combined together;
repeating the same key matches any of the provided values. Hyphens and
underscores are normalized in filter values, so ``homomorphic-encryption`` and
``homomorphic_encryption`` are equivalent.

Metadata Filter Values
======================

Filter values are recipe metadata values. A value is valid only when at least
one recipe in the installed catalog declares that value. In practice, most
metadata values apply to specific frameworks, so combine metadata filters with
``framework`` when you want a precise result.

Algorithm values:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Value
     - Frameworks
     - Example
   * - ``cyclic``
     - ``core``, ``pytorch``, ``tensorflow``
     - ``nvflare recipe list --filter algorithm=cyclic``
   * - ``fedavg``
     - ``core``, ``numpy``, ``pytorch``, ``sklearn``, ``tensorflow``
     - ``nvflare recipe list --filter algorithm=fedavg``
   * - ``fedavg_logistic_regression``
     - ``numpy``
     - ``nvflare recipe list --filter algorithm=fedavg_logistic_regression``
   * - ``fedeval``
     - ``pytorch``
     - ``nvflare recipe list --filter algorithm=fedeval``
   * - ``fedopt``
     - ``pytorch``, ``tensorflow``
     - ``nvflare recipe list --filter algorithm=fedopt``
   * - ``fedprox``
     - ``pytorch``, ``tensorflow``
     - ``nvflare recipe list --filter algorithm=fedprox``
   * - ``fedstats``
     - ``core``
     - ``nvflare recipe list --filter algorithm=fedstats``
   * - ``kmeans``
     - ``sklearn``
     - ``nvflare recipe list --filter algorithm=kmeans``
   * - ``scaffold``
     - ``pytorch``, ``tensorflow``
     - ``nvflare recipe list --filter algorithm=scaffold``
   * - ``svm``
     - ``sklearn``
     - ``nvflare recipe list --filter algorithm=svm``
   * - ``swarm``
     - ``pytorch``
     - ``nvflare recipe list --filter algorithm=swarm``
   * - ``xgboost_bagging``
     - ``xgboost``
     - ``nvflare recipe list --filter algorithm=xgboost_bagging``
   * - ``xgboost_horizontal``
     - ``xgboost``
     - ``nvflare recipe list --filter algorithm=xgboost_horizontal``
   * - ``xgboost_vertical``
     - ``xgboost``
     - ``nvflare recipe list --filter algorithm=xgboost_vertical``

Aggregation values:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Value
     - Frameworks
     - Example
   * - ``cluster_centers``
     - ``sklearn``
     - ``nvflare recipe list --filter framework=sklearn --filter aggregation=cluster_centers``
   * - ``server_optimizer``
     - ``pytorch``, ``tensorflow``
     - ``nvflare recipe list --filter aggregation=server_optimizer``
   * - ``support_vectors``
     - ``sklearn``
     - ``nvflare recipe list --filter framework=sklearn --filter aggregation=support_vectors``
   * - ``tree_ensemble``
     - ``xgboost``
     - ``nvflare recipe list --filter framework=xgboost --filter aggregation=tree_ensemble``
   * - ``weighted_average``
     - ``core``, ``numpy``, ``pytorch``, ``sklearn``, ``tensorflow``
     - ``nvflare recipe list --filter aggregation=weighted_average``

State exchange values:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Value
     - Frameworks
     - Example
   * - ``cluster_centers``
     - ``sklearn``
     - ``nvflare recipe list --filter state_exchange=cluster_centers``
   * - ``full_model``
     - ``core``, ``numpy``, ``pytorch``, ``sklearn``, ``tensorflow``
     - ``nvflare recipe list --filter state_exchange=full_model``
   * - ``model_weights``
     - ``numpy``
     - ``nvflare recipe list --filter state_exchange=model_weights``
   * - ``support_vectors``
     - ``sklearn``
     - ``nvflare recipe list --filter state_exchange=support_vectors``
   * - ``trees``
     - ``xgboost``
     - ``nvflare recipe list --filter state_exchange=trees``
   * - ``weight_diff``
     - ``pytorch``, ``tensorflow``
     - ``nvflare recipe list --filter state_exchange=weight_diff``

Privacy values:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Value
     - Frameworks
     - Example
   * - ``homomorphic_encryption``
     - ``pytorch``
     - ``nvflare recipe list --filter privacy=homomorphic_encryption``

The ``privacy`` filter matches privacy features declared by a recipe entry.
If a recipe does not declare a privacy value, that does not mean the underlying
algorithm is incompatible with privacy-enhancing technologies. For example,
FedAvg can be combined with multiple PETs through additional configuration or
components, but the generic FedAvg recipe does not enable one by default.

More examples:

.. code-block:: shell

   nvflare recipe list --filter framework=pytorch --filter algorithm=fedopt
   nvflare recipe list --filter framework=tensorflow --filter aggregation=server_optimizer
   nvflare recipe list --filter framework=xgboost --filter state_exchange=trees
   nvflare recipe list --filter framework=sklearn --filter aggregation=cluster_centers
   nvflare recipe list --filter privacy=homomorphic_encryption

Other options:

- Omitting ``--framework`` returns all available recipes.
- ``--schema``: print the command schema as JSON and exit.

Behavior notes:

- The command lists all documented built-in recipe variants, including recipe
  variants whose optional framework dependencies are not installed locally.
- ``optional_dependencies`` reports the packages needed to run a recipe whose
  framework is not currently installed.
- Valid metadata filters that match no available recipes return an empty list.
- The command combines a documented recipe manifest with dynamic recipe module
  discovery. When a recipe module cannot be imported because an optional
  dependency is missing, the CLI still returns the documented metadata and
  derives constructor parameters statically from source where possible.

The CLI prints a human-readable table in text mode and also emits the machine-
readable result envelope.

Example JSON response:

.. code-block:: json

   {
     "schema_version": "1",
     "status": "ok",
     "exit_code": 0,
     "data": [
       {
         "name": "fedavg-pt",
         "description": "FedAvg for PyTorch nn.Module models",
         "framework": "pytorch",
         "module": "nvflare.app_opt.pt.recipes.fedavg",
         "class": "FedAvgRecipe",
         "algorithm": "fedavg",
         "aggregation": "weighted_average",
         "state_exchange": "full_model",
         "privacy": []
       }
     ]
   }

****************
Show Recipe
****************

Use ``nvflare recipe show`` with a name returned by ``nvflare recipe list`` to
get one recipe's queryable metadata:

.. code-block:: shell

   nvflare recipe show fedavg-pt --format json

In text mode, the command prints ``Loading installed recipe metadata for
'<name>'...`` while it imports and inspects the selected recipe metadata. The
human output summarizes the main fields and points to the exact JSON command
for full constructor parameter details.

The JSON response includes list-time metadata plus framework support, privacy
compatibility, client requirements, constructor parameters, optional
dependencies, and template references. Parameter metadata is derived from the
recipe constructor signature or static source parsing; the command does not
instantiate the recipe.

For recipes with a configurable parameter transfer type, text output reports the
default state exchange and the transfer setting. For example, FedAvg reports
``state_exchange: full_model (default; params_transfer_type=FULL, supports FULL
or DIFF)`` because the default transfer is the full model, but the recipe can be
configured to send diffs.

Example JSON response:

.. code-block:: json

   {
     "schema_version": "1",
     "status": "ok",
     "exit_code": 0,
     "data": {
       "name": "fedavg-pt",
       "description": "FedAvg for PyTorch nn.Module models",
       "framework": "pytorch",
       "module": "nvflare.app_opt.pt.recipes.fedavg",
       "class": "FedAvgRecipe",
       "algorithm": "fedavg",
       "aggregation": "weighted_average",
       "state_exchange": "full_model",
       "privacy": [],
       "client_requirements": {
         "state_exchange": "full_model",
         "requires_training_script": true,
         "requires_per_site_config": true,
         "requires_site_list": false,
         "min_clients": {"required": true, "default": null}
       },
       "framework_support": ["pytorch"],
       "heterogeneity_support": ["horizontal"],
       "privacy_compatible": [],
       "parameters": [
         {
           "name": "min_clients",
           "type": "int",
           "required": true,
           "default": null,
           "kind": "keyword_only"
         }
       ],
       "optional_dependencies": ["pip install nvflare[PT]", "pip install torch"],
       "template_references": []
     }
   }

*****************
Typical Workflow
*****************

``nvflare recipe list`` is a discovery tool. A common workflow is:

.. code-block:: shell

   nvflare recipe list --filter framework=pytorch --filter algorithm=fedavg
   nvflare recipe show fedavg-pt --format json
   python job.py --export --export-dir /tmp/nvflare/hello-pt
   nvflare job submit -j /tmp/nvflare/hello-pt

``nvflare recipe list`` replaces the deprecated ``nvflare job list_templates``
discovery flow for new examples and recipes.

*********************
JSON Output and Help
*********************

Use ``--schema`` for machine-readable command discovery:

.. code-block:: shell

   nvflare recipe list --schema
   nvflare recipe show --schema

The top-level CLI also supports JSON output mode:

.. code-block:: shell

   nvflare recipe list --format json
   nvflare recipe list --framework sklearn --format json
   nvflare recipe list --filter framework=pytorch --filter state_exchange=full_model --format json
   nvflare recipe show fedavg-pt --format json

Human-readable argument errors print help first, followed by the specific
error. JSON mode prints only the JSON envelope.
