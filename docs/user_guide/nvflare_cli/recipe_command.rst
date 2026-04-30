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

The JSON response includes list-time metadata plus framework support, privacy
compatibility, client requirements, constructor parameters, optional
dependencies, and template references. Parameter metadata is derived from the
recipe constructor signature or static source parsing; the command does not
instantiate the recipe.

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
