.. _nvflare_cli:

###########################
NVFlare CLI
###########################

``nvflare`` is the main command-line entry point for local development,
provisioning, distributed provisioning, job submission, and runtime operations.

Running ``nvflare -h`` shows the current command surface:

.. code-block:: none

   usage: nvflare [-h] [--version] [--out-format {txt,json}]
                  [--connect-timeout CONNECT_TIMEOUT] ...

Global options
==============

- ``--version`` / ``-V``: print the NVFlare version
- ``--out-format {txt,json}``: select human-readable output or a JSON envelope
- ``--connect-timeout``: control server connection timeout for remote commands

Human-readable argument errors print command help first, then the specific
message and hint. ``--out-format json`` prints only the JSON response or JSON
error envelope, which is intended for automation and agent-style callers.

Command groups
==============

- ``poc``: manage a local proof-of-concept deployment
- ``provision``: centralized provisioning workflow
- ``cert`` / ``package``: distributed (manual) provisioning workflow
- ``job``: submit and manage jobs
- ``system``: inspect and operate a running FL system
- ``config``: store local CLI settings such as the startup kit path
- ``recipe``: list built-in recipe families for exported jobs
- ``preflight_check``: validate a provisioned startup kit before deployment
- ``dashboard``: start the Dashboard service

Deprecated commands still exposed in help, such as ``simulator`` and
``authz_preview``, are documented only briefly and should not be used for new
workflows unless you are maintaining an older setup.

.. toctree::
   :maxdepth: 1

   fl_simulator
   poc_command
   provision_command
   distributed_provisioning
   job_cli
   system_command
   config_command
   cert_command
   package_command
   recipe_command
   preflight_check
   dashboard_command
