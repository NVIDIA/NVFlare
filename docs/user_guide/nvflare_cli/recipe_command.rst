.. _recipe_command:

#########################
Recipe Command
#########################

``nvflare recipe`` lists built-in Job Recipe API recipes that can be used as
starting points for new FL workflows.

The current CLI surface provides one subcommand: ``nvflare recipe list``.

***********************
Command Usage
***********************

.. code-block:: none

   nvflare recipe -h

   usage: nvflare recipe [-h]  ...

   recipe subcommands:
     list      list available recipes (default)

****************
List Recipes
****************

Use ``nvflare recipe list`` to display the available built-in recipes:

.. code-block:: shell

   nvflare recipe list

Filter by framework:

.. code-block:: shell

   nvflare recipe list --framework pytorch

Supported framework filter values:

- ``any``
- ``pytorch``
- ``tensorflow``
- ``sklearn``
- ``xgboost``

Behavior notes:

- Recipes whose optional dependencies are not installed are skipped silently.
- The command checks recipe modules dynamically and returns only recipes that
  are actually available in the current environment.
- ``--schema`` prints the JSON schema for the command and exits.

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
         "class": "FedAvgRecipe"
       }
     ]
   }

*****************
Typical Workflow
*****************

``nvflare recipe list`` is a discovery tool. A common workflow is:

.. code-block:: shell

   nvflare recipe list --framework pytorch
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

The top-level CLI also supports JSON output mode:

.. code-block:: shell

   nvflare --out-format json recipe list
   nvflare --out-format json recipe list --framework sklearn

Human-readable argument errors print help first, followed by the specific
error. JSON mode prints only the JSON envelope.
