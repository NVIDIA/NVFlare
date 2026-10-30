.. _configurations:

###################
Configuration Files
###################

**Supported Configuration File Formats**

- `JSON <https://www.json.org/json-en.html>`_
- `YAML <https://yaml.org/>`_
- `Pyhocon <https://github.com/chimpler/pyhocon>`_ - a JSON variant and HOCON (Human-Optimized Config Object Notation) parser for python.
  Supports comments, variable substitution, and inheritance.
- `OmegaConf <https://omegaconf.readthedocs.io/en/2.3_branch/>`_ - a YAML based hierarchical configuration.

Users have the flexibility to use a single format or combine several formats, as exemplified by using config_fed_client.conf and config_fed_server.json together.
If multiple configuration formats coexist, then their usage will be prioritized based on the following search order:

``.json -> .conf -> .yml -> .yaml``

HOCON includes and substitutions, and OmegaConf interpolations, can read files, URLs, and environment variables.
Only configuration files in a site's ``startup/`` and ``local/`` folders may use them. Other configurations, including
job configs, must be plain data: ``.conf`` files must be valid JSON, and YAML files must not use aliases or ``${...}``.
Job submission and the simulator render HOCON and OmegaConf job configs to plain data on the submitting machine,
so environment variables a job config references are read from the submitter's environment.

See the sections below for more in-depth information about the different capabilities and types of configuration files:

.. toctree::
   :maxdepth: 1

   variable_resolution
   job_configuration
   server_port_consolidation
   communication_configuration
   logging_configuration
   site_config
