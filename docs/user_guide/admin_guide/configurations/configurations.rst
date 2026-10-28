.. _configurations:

###################
Configuration Files
###################

Job, startup, and resource configuration files use JSON syntax, with optional ``//`` line comments and
``/* ... */`` block comments and trailing commas. Existing ``.json`` and ``.conf``
filenames use the same parser. HOCON expressions, includes, and
YAML are no longer supported for these files. Provisioning project files and
site-owned ``study_runtime.yaml`` retain their YAML support.

FedJob/recipe exports and provisioned startup kits already contain JSON. Legacy
``nvflare job create`` renders bundled HOCON templates into JSON syntax while
keeping ``.conf`` names. Job metadata is data only;
runtime application references are described in :ref:`variable_resolution`.

Before upgrading, convert existing HOCON/YAML jobs and site configurations.
HOCON ``.conf`` filenames can stay unchanged; YAML files must become ``.json``
or ``.conf`` files containing JSON. YAML overrides produce a conversion error
instead of silently selecting a default configuration. Combine repeated HOCON
object definitions explicitly: JSON keeps only the last duplicate key.
Place site configuration overrides under ``startup/`` or ``local/``.

See the sections below for more in-depth information about the different capabilities and types of configuration files:

.. toctree::
   :maxdepth: 1

   variable_resolution
   job_configuration
   server_port_consolidation
   communication_configuration
   logging_configuration
   site_config
