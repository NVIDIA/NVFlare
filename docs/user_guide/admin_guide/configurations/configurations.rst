.. _configurations:

###################
Configuration Files
###################

**Configuration File Format**

FLARE job and site configuration files are `JSON <https://www.json.org/json-en.html>`_.
A ``.json.default`` file is used when the matching ``.json`` file does not exist.
HOCON (``.conf``) and OmegaConf (``.yml``/``.yaml``) configuration files are not supported.

See the sections below for more in-depth information about the different capabilities and types of configuration files:

.. toctree::
   :maxdepth: 1

   variable_resolution
   job_configuration
   server_port_consolidation
   communication_configuration
   logging_configuration
   site_config
