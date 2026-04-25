.. _nvflare_cli:

###########################
NVFlare CLI
###########################

``nvflare`` is the main command-line entry point for local development,
provisioning, distributed provisioning, job submission, and runtime operations.

Running ``nvflare -h`` shows the current command surface:

.. code-block:: none

   usage: nvflare [-h] [--version] [--format {txt,json}]
                  [--connect-timeout CONNECT_TIMEOUT] ...

Global options
==============

- ``--version`` / ``-V``: print the NVFlare version
- ``--format {txt,json}``: select human-readable output or a JSON envelope
- ``--connect-timeout``: control server connection timeout for remote commands

Human-readable argument errors print command help first, then the specific
message and hint. ``--format json`` prints only the JSON response or JSON
error envelope, which is intended for automation and agent-style callers.

Command groups
==============

- ``poc``: manage a local proof-of-concept deployment
- ``provision``: centralized provisioning workflow
- ``cert`` / ``package``: distributed (manual) provisioning workflow
- ``kit``: register local admin startup kits and choose the active startup kit
- ``job``: submit and manage jobs
- ``study``: manage multi-study lifecycle (register, enroll sites, manage users)
- ``system``: inspect and operate a running FL system
- ``config``: store non-startup-kit local CLI settings
- ``recipe``: list built-in recipe families for exported jobs
- ``preflight_check`` / ``preflight``: validate a provisioned startup kit
  before deployment (``preflight`` is the preferred alias)
- ``dashboard``: start the Dashboard service

Deprecated commands still exposed in help, such as ``simulator`` and
``authz_preview``, are documented only briefly and should not be used for new
workflows unless you are maintaining an older setup.

.. toctree::
   :maxdepth: 1

   fl_simulator
   poc_command
   kit_command
   provision_command
   distributed_provisioning
   job_cli
   study_command
   system_command
   config_command
   cert_command
   package_command
   recipe_command
   preflight_check
   dashboard_command
