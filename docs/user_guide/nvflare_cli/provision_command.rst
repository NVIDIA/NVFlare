*****************************
Provision command
*****************************

Running ``nvflare provision -h`` shows all available options.

.. code-block:: shell

    usage: nvflare provision [-h] [-p PROJECT_FILE] [-g] [-e] [-w WORKSPACE]
                             [-c CUSTOM_FOLDER] [--add_user ADD_USER]
                             [--add_client ADD_CLIENT] [-s] [--force] [--schema]

    options:
    -h, --help                                               show this help message and exit
    -p PROJECT_FILE, --project_file PROJECT_FILE                 file to describe FL project
    -g, --generate                                             generate a sample project.yml and exit
    -e, --gen_edge                                             generate a sample edge project.yml and exit
    -w WORKSPACE, --workspace WORKSPACE                          directory used by provision
    -c CUSTOM_FOLDER, --custom_folder CUSTOM_FOLDER    additional folder to load python code
    --add_user ADD_USER                                             yaml file for added user
    --add_client ADD_CLIENT                                       yaml file for added client
    -s, --gen_scripts                                            generate helper scripts such as start_all.sh
    --force                                                      skip Y/N confirmation prompts
    --schema                                                     print command schema as JSON and exit

Running ``provision`` without any options and without a project.yml file in the current working directory will prompt
to copy a default project.yml to the current working directory.

JSON mode
=========

``nvflare provision --format json`` returns only a JSON envelope on
stdout. When the command generates a sample project file, the JSON ``data``
section includes structured guidance such as:

- ``message``
- ``next_step``
- ``suggested_command``

This keeps JSON output machine-readable while still carrying follow-up
instructions.

Certificate Identity Overrides
==============================

For mTLS deployments, each CellNet endpoint is authenticated against the peer
certificate common name (CN). By default, the expected certificate identity is
derived from the participant name or FQCN. If a participant intentionally uses a
certificate CN that differs from its FLARE site name, set ``auth_identity`` in
``project.yml`` before provisioning.

For example, if the FLARE site is named ``site-1`` but its certificate CN is
``server.example.com``:

.. code-block:: yaml

   participants:
     - name: site-1
       type: client
       org: nvidia
       auth_identity: server.example.com

Provisioning uses this value to generate the corresponding startup-kit
configuration, including peer identity mappings needed by the server and by job
cells. A job cell such as ``site-1.<job_id>`` still authenticates with the
parent site's configured certificate identity.

.. important::

   Endpoint-to-CN binding is enforced when an mTLS transport exposes the
   authenticated peer CN. Some active-side gRPC connections do not expose the
   peer CN to the Python driver, so NVFlare accepts those active connections and
   relies on passive-side endpoint validation plus the normal application-layer
   authentication checks.

   Admin console cells use per-session endpoint names and authenticate the admin
   user by certificate/user identity on the admin listener. Keep admin
   application-layer authentication configured and protected.

   ``auth_identity`` is loaded when a FLARE process starts. After rotating site
   certificates or changing certificate CNs, regenerate the startup kits as
   needed and restart the affected FLARE processes so the in-memory identity
   resolver uses the new certificate identity.

.. warning::

   Do not edit ``startup/fed_client.json`` or ``startup/fed_server.json`` by
   hand in a signed startup kit. If ``startup/signature.json`` is present, the
   startup configuration is covered by the signature and manual edits invalidate
   that signature. Change ``project.yml`` and re-run provisioning so the startup
   configuration and signature are generated together.

.. _dynamic_provisioning_cli:

Dynamic Provisioning
====================

The options ``--add_user`` and ``--add_client`` allow for adding to an existing project. Both of these commands take a yaml
file to define the additional participant to provision.

Sample user.yaml for ``--add_user``:

.. code-block:: yaml

    name: new_user@nvidia.com
    org: nvidia
    role: project_admin


Sample client.yaml for ``--add_client``:

.. code-block:: yaml

    name: new-site
    org: nvidia
    components:
      resource_manager:    # This id is reserved by system.  Do not change it.
        path: nvflare.app_common.resource_managers.gpu_resource_manager.GPUResourceManager
        args:
          num_of_gpus: 0
          mem_per_gpu_in_GiB: 0
      resource_consumer:    # This id is reserved by system.  Do not change it.
        path: nvflare.app_common.resource_consumers.gpu_resource_consumer.GPUResourceConsumer
        args:
 

After running ``nvflare provision`` with ``--add_user`` or ``--add_client`` followed by the name of the yaml file (:mod:`nvflare.lighter.provision` will
look for the yaml file in the current directory), the new user or client will be included in the prod_NN folder.

To permanently include users or clients, please update the project.yml.

.. note::

   To use multi-study features, set ``api_version: 4`` in your ``project.yml`` and add a ``studies:``
   section. See :ref:`multi_study_guide` for details.
