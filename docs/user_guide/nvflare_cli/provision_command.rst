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

``nvflare provision --out-format json`` returns only a JSON envelope on
stdout. When the command generates a sample project file, the JSON ``data``
section includes structured guidance such as:

- ``message``
- ``next_step``
- ``suggested_command``

This keeps JSON output machine-readable while still carrying follow-up
instructions.

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
