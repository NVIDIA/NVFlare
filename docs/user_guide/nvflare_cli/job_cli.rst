.. _job_cli:

#########################
NVIDIA FLARE Job CLI
#########################

The NVIDIA FLARE :mod:`Job CLI<nvflare.tool.job.job_cli>` provides options to create and submit
jobs from a command line interface. See the :github_nvflare_link:`NVFlare Job CLI Notebook <examples/tutorials/setup_poc.ipynb>`
for a tutorial on how to use the Job CLI.

.. note::
    
    We have introduced a new Pythonic Job API experience, please
    check :ref:`fed_job_api`.

***********************
Command Usage
***********************

.. code-block::

    usage: nvflare job [-h] {list_templates,create,submit,show_variables} ...

    options:
    -h, --help            show this help message and exit

    job:
    {list_templates,create,submit,show_variables}
                        job subcommand
    list_templates      show available job templates
    create              create job
    submit              submit job
    show_variables      show template variable values in configuration


*****************
Command examples
*****************

Lists Job Templates
===================

The ``nvflare job list_templates`` command lists the available job templates. The option
``-d "<job_templates_dir>"`` or ``--job_template_dir "<job_templates_dir>"`` is the location of
the job_templates.

.. code-block:: shell

    nvflare job list_templates -d "<NVFlare location>/job_templates"

The output should be similar to the following:

.. code-block:: none

    The following job templates are available: 

    ----------------------------------------------------------------------------------------------------------------------
    name                 Description                                                  Controller Type   Execution API Type
    ----------------------------------------------------------------------------------------------------------------------
    cyclic_cc_pt         client-controlled cyclic workflow with PyTorch ClientAPI tra client            client_api
    cyclic_pt            server-controlled cyclic workflow with PyTorch ClientAPI tra server            client_api
    psi_csv              private-set intersection for csv data                        server            Executor
    sag_cross_np         scatter & gather and cross-site validation using numpy       server            client executor
    sag_cse_pt           scatter & gather workflow and cross-site evaluation with PyT server            client_api
    sag_gnn              scatter & gather workflow for gnn learning                   server            client_api
    sag_nemo             Scatter and Gather Workflow for NeMo                         server            client_api
    sag_np               scatter & gather workflow using numpy                        server            client_api
    sag_np_cell_pipe     scatter & gather workflow using numpy                        server            client_api
    sag_np_metrics       scatter & gather workflow using numpy                        server            client_api
    sag_pt               scatter & gather workflow using pytorch                      server            client_api
    sag_pt_deploy_map    SAG workflow with pytorch, deploy_map, site-specific configs server            client_api
    sag_pt_executor      scatter & gather workflow and cross-site evaluation with PyT server            Executor
    sag_pt_he            scatter & gather workflow using pytorch and homomorphic encr server            client_api
    sag_pt_mlflow        scatter & gather workflow using pytorch with MLflow tracking server            client_api
    sag_pt_model_learner scatter & gather workflow and cross-site evaluation with PyT server            ModelLearner
    sag_tf               scatter & gather workflow using TensorFlow                   server            client_api
    sklearn_kmeans       scikit-learn KMeans model                                    server            client_api
    sklearn_linear       scikit-learn linear model                                    server            client_api
    sklearn_svm          scikit-learn SVM model                                       server            client_api
    stats_df             FedStats: tabular data with pandas                           server            stats executor
    stats_image          FedStats: image intensity histogram                          server            stats executor
    swarm_cse_pt         Swarm Learning with Cross-Site Evaluation with PyTorch       client            client_api
    swarm_cse_pt_model_l Swarm Learning with Cross-Site Evaluation with PyTorch Model client            ModelLearner
    vertical_xgb         vertical federated xgboost                                   server            Executor
    xgboost_tree         xgboost horizontal tree-based collaboration model            server            client_api
    ----------------------------------------------------------------------------------------------------------------------

View all the available templates at the :github_nvflare_link:`FLARE Job Template Registry <job_templates>`.

Setting job_template path
-------------------------
You can also use the ``nvflare job list_templates`` command without the `-d` option. When the job templates directory
is not specified, the Job CLI will try to find the location with the following logic:

See if the ``NVFLARE_HOME`` environment variable is set. If ``NVFLARE_HOME`` is not empty, the Job CLI will look for the job templates at
``${NVFLARE_HOME}/job_templates``.
 
If the ``NVFLARE_HOME`` environment variable is not set, the Job CLI will look for the ``job_template`` path in the config in the nvflare
hidden directory (located at ``~/.nvflare/config.conf``). Once the ``-d <job_template_dir>`` option is used, the ``job_template`` value
in ``~/.nvflare/config.conf`` will be updated so you don't need to specify ``-d`` again. 

If you want to change the ``job_template`` path, you can directly edit this config file or use the ``nvflare config`` command with the
``-jt`` or ``--job_templates_dir`` option:

.. code-block:: shell

    nvflare config -jt ../../job_templates


Create new job
===================

The ``nvflare job create`` command will allow you to create a new job based on a template, with options to replace variables in config files.
The options for usage are as follows:

.. code-block::

    usage: nvflare job create [-h] [-j [JOB_FOLDER]] [-w [TEMPLATE]] [-sd [SCRIPT_DIR]] [-f [CONFIG_FILE [CONFIG_FILE ...]]] [-debug] [-force]

    optional arguments:
    -h, --help            show this help message and exit
    -j [JOB_FOLDER], --job_folder [JOB_FOLDER]
                            job_folder path, default to ./current_job directory
    -w [TEMPLATE], --template [TEMPLATE]
                            template name or template folder. You can use list_templates to see available jobs from job templates, pick name such as 'sag_pt' as template name. Alternatively, you can use the path to the job
                            template folder, such as job_templates/sag_pt
    -sd [SCRIPT_DIR], --script_dir [SCRIPT_DIR]
                            script directory contains additional related files. All files or directories under this directory will be copied over to the custom directory.
    -f [CONFIG_FILE [CONFIG_FILE ...]], --config_file [CONFIG_FILE [CONFIG_FILE ...]]
                            Training config file with corresponding optional key=value pairs. If key presents in the preceding config file, the value in the config file will be overwritten by the new value
    -debug, --debug       debug is on
    -force, --force       force create is on, if -force, overwrite existing configuration with newly created configurations

The ``-j`` option or ``--job_folder`` option is the path to the job folder to be created. If the job folder is not specified, the Job CLI will create a
``current_job`` folder in the current directory.

The ``-w`` option or ``--template`` option is the name of the template that the new job will be created from.


Show variables
===============
The ``nvflare job show_variables`` command can be used to show the variables in a job. The options for usage are as follows:

.. code-block:: shell

    nvflare job show_variables -j <path/to/my_job>


Submit job with CLI
===================

The ``nvflare job submit`` command can be used to submit jobs:

.. code-block::

    usage: nvflare job submit [-h] [-j [JOB_FOLDER]] [-f [CONFIG_FILE ...]] [-debug]

    options:
    -h, --help            show this help message and exit
    -j [JOB_FOLDER], --job_folder [JOB_FOLDER]
                            job_folder path, default to ./current_job directory
    -f [CONFIG_FILE ...], --config_file [CONFIG_FILE ...]
                            Training config file with corresponding optional key=value pairs. If key presents in the preceding config file, the value in the config file will be overwritten by the new value
    -debug, --debug       debug is on

In order to do this, it will need to know the location of the admin console
startup kit directory. In POC mode, this is set for the user automatically. For a provisioned setup, the user will need to set the path to
the startup kit for the Job CLI. The startup kit path is stored in the ``~/.nvflare/config.conf`` file in the nvflare hidden directory at
the user's home directory. You can edit this path in the file and set it directly for example:

.. code-block:: shell

    startup_kit {
        path = /tmp/nvflare/poc/example_project/prod_00
    }

Alternatively, you can use the ``nvflare config`` command with the ``-d`` or ``--startup_kit_dir`` option to set the startup kit path:

.. code-block:: shell

    nvflare config --startup_kit_dir /tmp/nvflare/poc/example_project/prod_00

With the startup kit directory path set, you can submit the job (this following example is from the
:github_nvflare_link:`NVFlare Job CLI Notebook <examples/tutorials/setup_poc.ipynb>` and
replaces several variables in the ``config_fed_server.conf`` config file):

.. code-block:: shell

    nvflare job submit -j /tmp/nvflare/my_job -f config_fed_server.conf num_rounds=1 app_config="--dataset_path /tmp/nvflare/data/cifar10"

Troubleshooting with the -debug flag
------------------------------------

Since the ``nvflare job submit`` command does not overwrite the job folder configuration during submission, it has to use a temp job folder. 
If you want to check the final configs submited to the server or simply want to see the stack trace of the exception, you can use the ``-debug`` flag. 

With the ``-debug`` flag, the ``nvflare job submit`` command will not delete the temp job folder once it has finished job submission,
and it will also print the exception stack trace in case of failure.

When you submit a job with the ``-debug`` flag, you should see a statement like the following after the message that the job was
submitted (the actual random folder name will vary): 

.. code-block:: shell

    in debug mode, job configurations can be examined in temp job directory '/tmp/tmpdnusoyzj'

You can look at the contents of the temp job folder for more information about the job submission. For example, you can look at the
``config_fed_server.conf`` file in the temp job folder to see if the final configuration is what you intended.

***************************
Advanced Job Configurations
***************************

For different configurations for different client sites, you can use the ``-f`` option to specify the variables to change for each
config file for each client site.

For example, to change number of training rounds to 2, change default app_script from "cifar10.py" to "train.py" for both app_1 and app_2,
and change the app_1 batch_size to 4, app_2 batch_size to 6 for sag_pt_deploy_map as in the
:github_nvflare_link:`NVFlare Job CLI Notebook <examples/tutorials/setup_poc.ipynb>`:

.. code-block:: shell

    nvflare job create \
    -j /tmp/nvflare/my_job -w sag_pt_deploy_map \
    -f app_server/config_fed_server.conf num_rounds=2 \
    -f app_1/config_fed_client.conf app_script=train.py app_config="--batch_size 4" \
    -f app_2/config_fed_client.conf app_script=train.py app_config="--batch_size 6" \
    -sd ../hello-world/step-by-step/cifar10/code/fl

.. note::

    The app names must be defined in the job template being used: in this case ``app_1``, ``app_2``, and ``app_server``,
    are in ``sag_pt_deploy_map``.
