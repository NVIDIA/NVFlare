.. _configurations:

###################
Configuration Files
###################

.. deprecated:: 2.9.1

   HOCON configuration through PyHOCON and YAML configuration through OmegaConf
   are deprecated in 2.9.1 and will be removed in 2.10.0. They remain supported
   in 2.9.1. Use JSON for new runtime and job configuration, and follow
   :ref:`configuration_format_migration_291` to migrate existing files.

This deprecation applies to the runtime and job configuration loaders,
including their ``.default`` variants. Provisioning YAML such as
``project.yml`` and ``cc_project.yml`` uses separate provisioning loaders
and is outside this deprecation.

**Supported Configuration File Formats**

- `JSON <https://www.json.org/json-en.html>`_ — recommended.
- HOCON through `PyHOCON <https://github.com/chimpler/pyhocon>`_ (``.conf``) —
  deprecated in 2.9.1; supports comments, variable substitution, and inheritance.
- YAML through `OmegaConf <https://omegaconf.readthedocs.io/en/2.3_branch/>`_
  (``.yml`` and ``.yaml``) — deprecated in 2.9.1.

In 2.9.1, configurations can still use a single format or combine formats, such
as ``config_fed_client.conf`` and ``config_fed_server.json``. When multiple
configuration formats coexist, lookup uses the following search order:

``.json -> .conf -> .yml -> .yaml``

See the sections below for more in-depth information about the different capabilities and types of configuration files:

.. toctree::
   :maxdepth: 1

   variable_resolution
   job_configuration
   server_port_consolidation
   communication_configuration
   logging_configuration
   site_config
