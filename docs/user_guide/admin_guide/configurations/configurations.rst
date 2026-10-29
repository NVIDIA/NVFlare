.. _configurations:

###################
Configuration Files
###################

**Supported Configuration File Formats**

- `JSON <https://www.json.org/json-en.html>`_ with ``//`` and ``/* ... */`` comments and trailing commas.
  Both ``.json`` and ``.conf`` files use JSON syntax.
- `YAML <https://yaml.org/>`_ parsed as data with ``yaml.safe_load``; OmegaConf expressions remain literal.

Users have the flexibility to use a single format or combine several formats, as exemplified by using config_fed_client.conf and config_fed_server.json together.
If multiple configuration formats coexist, then their usage will be prioritized based on the following search order: 

``.json -> .conf -> .yml -> .yaml``

Before upgrading, convert HOCON content to JSON or YAML; ``.conf`` filenames can stay
when converted to JSON. HOCON includes and substitutions are unsupported. Combine repeated
HOCON object definitions explicitly: duplicate JSON keys are rejected. Legacy
``nvflare job create`` resolves local HOCON templates before writing JSON.
YAML filenames and comments remain supported; replace OmegaConf expressions with literal
values or supported FLARE runtime references (:ref:`variable_resolution`).
Site overrides belong under ``startup/`` or ``local/``. Provisioning YAML and site-owned
``study_runtime.yaml`` are unchanged. Neither supported format needs an extra dependency.

See the sections below for more in-depth information about the different capabilities and types of configuration files:

.. toctree::
   :maxdepth: 1

   variable_resolution
   job_configuration
   server_port_consolidation
   communication_configuration
   logging_configuration
   site_config
