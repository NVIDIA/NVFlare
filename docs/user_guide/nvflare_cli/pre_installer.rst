.. _pre_installer:

##########################
NVFLARE Code Pre-Installer
##########################

This tool helps install NVFLARE application code and libraries before running federated learning jobs.

Overview
========

The code pre-installer handles:
- Installation of application code
- Installation of shared libraries
- Site-specific customizations
- Python package dependencies

The tool provides two main commands:
- `prepare`: Package application code for installation
- `install`: Install packaged code to target sites

Directory Structure
===================

Expected application code zip structure:

.. code-block:: text

   application.zip
   ├── application/<job_name>/
   │               ├── meta.json       # Job metadata
   │               ├── app_<site>/     # Site custom code
   │                  └── custom/      # Site custom code
   ├── application-share/              # Shared resources
   │   └── shared.py
   └── requirements.txt       # Python dependencies (optional)

or

.. code-block:: text

   application.zip
   ├── application/<job_name>/
   │               ├── meta.json       # Application metadata
   │               ├── app/            # Site custom code
   │                  └── custom/      # Site custom code

Here is an example of creating a folder structure for pre-installation:

.. code-block:: bash

   mkdir -p /tmp/nvflare/pre-install/application
   mkdir -p /tmp/nvflare/pre-install/application-share

For example, if the app name is `fedavg`, the directory structure would look like this:

Tree structure of the job configuration:

.. code-block:: text

   /tmp/nvflare/pre-install/
   ├── application
   │   └── fedavg
   │       ├── app_server
   │       │   ├── config
   │       │   └── custom
   │       ├── app_site-1
   │       │   ├── config
   │       │   └── custom
   │       ├── app_site-2
   │       │   ├── config
   │       │   └── custom
   │       ├── app_site-3
   │       │   ├── config
   │       │   └── custom
   │       ├── app_site-4
   │       │   ├── config
   │       │   └── custom
   │       ├── app_site-5
   │       │   ├── config
   │       │   └── custom
   │       └── meta.json
   └── application-share
       └── pt
           ├── learner_with_mlflow.py
           ├── learner_with_tb.py
           ├── learner_with_wandb.py
           ├── pt_constants.py
           ├── simple_network.py
           └── test_custom.py

Then we can simply copy the `fedavg` folder to the pre-install folder:

.. code-block:: bash

   cp -r /tmp/nvflare/jobs/workdir/fedavg /tmp/nvflare/pre-install/application/.

If you have shared code (such as Python modules with nested folders and files) in "/tmp/nvflare/jobs/workdir/pt", copy it to the application-share directory:

.. code-block:: bash

   cp -r /tmp/nvflare/jobs/workdir/pt /tmp/nvflare/pre-install/application-share/.

You should have something like the following:

.. code-block:: text

   tree /tmp/nvflare/pre-install/ -L 3
   /tmp/nvflare/pre-install/
   ├── application
   │   └── fedavg
   │       ├── app_server
   │       ├── app_site-1
   │       ├── app_site-2
   │       ├── app_site-3
   │       ├── app_site-4
   │       ├── app_site-5
   │       └── meta.json
   └── application-share
       └── pt
           ├── learner_with_mlflow.py
           ├── learner_with_tb.py
           ├── learner_with_wandb.py
           ├── pt_constants.py
           ├── simple_network.py
           └── test_custom.py

Finally, create the app-code.zip file from the pre-install folder:

.. code-block:: bash

   cd /tmp/nvflare/pre-install/
   zip -r ../application.zip *

The application.zip file will be created in the `/tmp/nvflare/` directory.

This can also be done with `nvflare pre-install prepare` command

Usage
=====

Command Line Interface
----------------------

Prepare Application Code
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   nvflare pre-install prepare [-h] -j JOB [-o OUTPUT] [-s SHARED] [-r REQUIREMENTS] [-debug]

Arguments:
  -j, --job            Job folder path (e.g., jobs/fedavg)
  -o, --output        Output directory for application.zip (default: /tmp/application/prepare)
  -s, --shared        Optional shared library folder
  -r, --requirements  Optional requirements.txt file
  -debug              Enable debug output

Install Application Code
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   nvflare pre-install install [-h] -a APPLICATION [-p INSTALL_PREFIX] -s SITE_NAME
                             [-ts TARGET_SHARED_DIR] [-debug] [-d]
   Arguments:
      -a, --application    Path to application code zip file
      -p, --install-prefix Installation prefix (default: /opt/nvflare/apps)
      -s, --site-name      Target site name (e.g., site-1, server)
      -ts, --target_shared_dir Target share path (default: /local/custom)
      -debug               Enable debug output
      -d, --delete        Delete the zip file after installation

Example
-------

.. code-block:: bash

   # 1. Package application code
   nvflare pre-install prepare -j jobs/fedavg -o /tmp/prepare

   # Package with requirements.txt
   nvflare pre-install prepare -j jobs/fedavg -o /tmp/prepare -r requirements.txt

   # 2. Install on server
   nvflare pre-install install -a /tmp/prepare/application.zip -s server

   # 3. Install on clients
   nvflare pre-install install -a /tmp/prepare/application.zip -s site-1

Application Code Structure
==========================

The application zip file should have the following structure:

.. code-block:: text

   application/
   ├── job_name/
   │   ├── meta.json
   │   ├── app_site-1/
   │   │   └── custom/
   │   │       └── site_specific_code.py
   │   └── app_site-2/
   │       └── custom/
   │           └── site_specific_code.py
   └── application-share/
       └── shared_code.py

- `job_name/`: job directory containing site-specific code
- `meta.json`: job metadata file
- `app_site-*/custom/`: Site-specific custom code directories
- `application-share/`: Shared code directory

Installation Paths
==================

- Application code: `<install-prefix>/<job-name>/`
- Shared resources: `/local/custom/`

Error Handling
==============

The installer will fail if:
- Job structure zip is invalid or missing required directories
- meta.json is missing or invalid
- Site directory not found and no default apps available
- Installation directories cannot be created
- File operations fail
- Package installation fails (if requirements.txt present)

Notes
=====

- Existing files may be overwritten
- Python path is automatically configured for shared packages
- All file permissions are preserved during installation
- Network access needed if requirements.txt present
- Can use private PyPI server by configuring pip
- The tool will extract site-specific code to the installation prefix
- Shared code will be installed to the target shared directory
- The application zip file will be cleaned up after installation
- Installation paths must be writable by the current user

Using Pre-installed Code when submit job
========================================

Here is the configuration change, in development, if there is "custom" directory,
i.e. the python training code is not **pre-installed**, the config_fed_client.json

.. code-block:: json

   {
       "format_version": 2,
       "executors": [
           {
               "tasks": [
                   "*"
               ],
               "executor": {
                   "path": "nvflare.app_opt.pt.in_process_client_api_executor.PTInProcessClientAPIExecutor",
                   "args": {
                       "task_script_path": "src/client.py",
                       "task_script_args": "--learning_rate 0.01 --batch_size 12",
                       "params_exchange_format": "numpy"
                   }
               }
           }
       ],
       "components": [],
       "task_data_filters": [],
       "task_result_filters": []
   }

Now the **pre-installed** training code, the config_fed_client.json will need to be changed

.. code-block:: json

   {
       "format_version": 2,
       "executors": [
           {
               "tasks": [
                   "*"
               ],
               "executor": {
                   "path": "nvflare.app_opt.pt.in_process_client_api_executor.PTInProcessClientAPIExecutor",
                   "args": {
                       "task_script_path": "/tmp/opt/nvflare/site-1/fedavg/src/client.py",
                       "task_script_args": "--learning_rate 0.01 --batch_size 12",
                       "params_exchange_format": "numpy"
                   }
               }
           }
       ],
       "components": [],
       "task_data_filters": [],
       "task_result_filters": []
   }

Notice that

.. code-block:: json

   "task_script_path": "/tmp/opt/nvflare/site-1/fedavg/src/client.py",

**"src/client.py"** --> **"/tmp/opt/nvflare/site-1/fedavg/src/client.py"**

**<install-prefix>/fedavg/** is the prefix
