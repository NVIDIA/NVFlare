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

See the sections below for more in-depth information about the different capabilities and types of configuration files:

.. toctree::
   :maxdepth: 1

   configurations/variable_resolution
   configurations/job_configuration
   configurations/communication_configuration
   configurations/logging_configuration
   