.. _configurations:

###################
Configuration Files
###################

**Configuration File Format**

FLARE job and core site configuration files are `JSON <https://www.json.org/json-en.html>`_.
A ``.json.default`` file is used when the matching ``.json`` file does not exist.
The generic configuration loader no longer supports HOCON (``.conf``) or
OmegaConf (``.yml``/``.yaml``), including their ``.default`` variants.

YAML is still supported by tools that have their own YAML readers, using PyYAML.
The format depends on which tool reads the file:

.. list-table:: Configuration formats by purpose
   :header-rows: 1
   :widths: 25 10 65

   * - Purpose
     - Format
     - Examples
   * - Job definition and execution
     - JSON
     - ``meta.json``, ``config_fed_server.json``, ``config_fed_client.json``
   * - Core site settings
     - JSON
     - ``fed_server.json``, ``fed_client.json``, ``fed_admin.json``,
       ``resources.json``, ``comm_config.json``, ``log_config.json``
   * - CLI settings
     - JSON
     - ``~/.nvflare/config.json``
   * - Provisioning
     - YAML
     - ``project.yml`` and provisioning templates
   * - Deployment and study launch settings
     - YAML
     - Kubernetes/Helm manifests and ``local/study_runtime.yaml``
   * - Integration test orchestration
     - YAML
     - ``tests/integration_test/test_configs.yml`` and its test scenario files

A YAML test scenario describes which jobs to submit and which results to expect;
the job's metadata and FLARE execution configs still use JSON. Training code can
also read its own application-specific YAML files independently of FLARE's
configuration loader.

See the sections below for more in-depth information about the different capabilities and types of configuration files:

.. toctree::
   :maxdepth: 1

   variable_resolution
   job_configuration
   server_port_consolidation
   communication_configuration
   logging_configuration
   site_config
