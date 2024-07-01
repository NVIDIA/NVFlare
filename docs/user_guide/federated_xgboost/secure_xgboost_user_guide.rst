##########################
NVFlare XGBoost User Guide
##########################

Overview
========
NVFlare supports federated training with XGBoost. It provides the following advantages over doing the training natively with XGBoost:

- Secure training with Homomorphic Encryption (HE)
- Lifecycle management of XGBoost processes.
- Reliable messaging which can overcome network glitches
- Training over complicated networks with relays.

It supports federated training in the following 4 modes:

1. **horizontal(h)**: Row split without encryption
2. **vertical(v)**: Column split without encryption
3. **horizontal_secure(hs)**: Row split with HE (Requires at least 3 clients. With 2 clients, the other client's histogram can be deduced.)
4. **vertical_secure(vs)**: Column split with HE

When running with NVFlare, all the GRPC connections in XGBoost are local and the messages are forwarded to other clients through NVFlare's CellNet communication. The local GRPC ports are selected automatically by NVFlare.

The encryption, decryption, and secure aggregation in XGBoost are handled by processor plugins, which are external components that can be installed at runtime. The plugins are bundled with NVFlare.

Prerequisites
=============
Required Python Packages
------------------------

NVFlare 2.4.2 or above,

.. code-block:: bash

    pip install nvflare~=2.4.2

A version of XGBoost that supports secure mode is required, which can be installed using this command,

.. code-block:: bash

	pip install https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/vertical-federated-learning/xgboost-2.1.0.dev0%2Bde4013fc733648dfe5c2c803a13e2782056e00a2-py3-none-manylinux_2_28_x86_64.whl

``TenSEAL`` package is needed for horizontal secure training,

.. code-block:: bash

    pip install tenseal

``ipcl_python`` package is required for vertical secure training if **nvflare** plugin is used. This package is not needed if **cuda_paillier** plugin is used.

.. code-block:: bash

    pip install ipcl-python

This package is only available for Python 3.8 on PyPI. For other versions of python, it needs to be installed from github,

.. code-block:: bash

    pip install git+https://github.com/intel/pailliercryptolib_python.git@development

System Environments
-------------------
To support secure training, several homomorphic encryption libraries are used. Those libraries require Intel CPU or Nvidia GPU.

Linux is the preferred OS. It's tested extensively under Ubuntu 22.4.

The following docker image is recommended for GPU training:

::

    nvcr.io/nvidia/pytorch:24.03-py3

Most Linux distributions are supported, as long as they have a recent glibc. The oldest glibc version tested is 2.35. Systems with older glibc may run into issues.

NVFlare Provisioning
--------------------
For horizontal secure training, the NVFlare system must be provisioned with homomorphic encryption context. The HEBuilder in ``project.yml`` is used to achieve this.
An example configuration can be found at :github_nvflare_link:`secure_project.yml <examples/advanced/cifar10/cifar10-real-world/workspaces/secure_project.yml#L64>`.

This is a snippet of the ``secure_project.yml`` file with the HEBuilder:

.. code-block:: yaml

    api_version: 3
    name: secure_project
    description: NVIDIA FLARE sample project yaml file for CIFAR-10 example

    participants:

    ...

    builders:
    - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
        args:
        template_file: master_template.yml
    - path: nvflare.lighter.impl.template.TemplateBuilder
    - path: nvflare.lighter.impl.static_file.StaticFileBuilder
        args:
        config_folder: config
        overseer_agent:
            path: nvflare.ha.dummy_overseer_agent.DummyOverseerAgent
            overseer_exists: false
            args:
            sp_end_point: localhost:8102:8103
            heartbeat_interval: 6
    - path: nvflare.lighter.impl.he.HEBuilder
        args:
        poly_modulus_degree: 8192
        coeff_mod_bit_sizes: [60, 40, 40]
        scale_bits: 40
        scheme: CKKS
    - path: nvflare.lighter.impl.cert.CertBuilder
    - path: nvflare.lighter.impl.signature.SignatureBuilder


Data Preparation
================
Data must be properly formatted for federated XGBoost training based on split mode (horizontal or vertical).

For horizontal training, the datasets on all clients must share the same columns.

For vertical training, the datasets on all clients contain different columns, but must share overlapping rows. For more details on vertical split preprocessing, refer to the :github_nvflare_link:`Vertical XGBoost Example <examples/advanced/vertical_xgboost>`.

XGBoost Plugin Configuration
============================
XGBoost requires a plugin to handle secure training. 
Two plugins are initially shipped with NVFlare,

- **cuda_paillier**: The default plugin. This plugin uses GPU for cryptographic operations.
- **nvflare**: This plugin forwards data locally to NVFlare process for encryption.

Vertical (Non-secure)
---------------------
Any plugin can be used for vertical training. No configuration change is needed.

Horizontal (Non-secure)
-----------------------
Any plugin can be used for horizontal training. No configuration change is needed.

Vertical Secure
---------------
Both plugins can be used for vertical secure training.

The default cuda_paillier plugin is preferred because it uses GPU for faster cryptographic operations.

.. note::

    **cuda_paillier** plugin requires NVIDIA GPUs that support compute capability 7.0 or higher. Please refer to https://developer.nvidia.com/cuda-gpus for more information.

If you see the following errors in the log, it means either no GPU is available or the GPU does not meet the requirements:

::

    CUDA runtime API error no kernel image is available for execution on the device at line 241 in file /my_home/nvflare-internal/processor/src/cuda-plugin/paillier.h
    2024-07-01 12:19:15,683 - SimulatorClientRunner - ERROR - run_client_thread error: EOFError:


In this case, the nvflare plugin can be used to perform encryption on CPUs, which requires the ipcl-python package.
The plugin can be configured in the ``local/resources.json`` file on clients:

.. code-block:: json

    {
        "xgb_plugin_name": "nvflare"
    }

or by setting this environment variable,

.. code-block:: bash

    export NVFLARE_XGB_PLUGIN_NAME=nvflare

Horizontal Secure
-----------------
The plugin setup is the same as vertical secure.

This mode requires the tenseal package for all plugins.
The provisioning of NVFlare systems must include tenseal context.
See :ref:`provisioning` for details.


Job Configuration
=================
Controller
----------

On the server side, following controller must be configured in workflows,

``nvflare.app_opt.xgboost.histogram_based_v2.fed_controller.XGBFedController``

Even though the XGBoost training is performed on clients, the parameters are configured on the server so all clients share the same configuration. 
XGBoost parameters are defined here, https://xgboost.readthedocs.io/en/stable/python/python_intro.html#setting-parameters

- **num_rounds**: Number of training rounds
- **training_mode**: Training mode, must be one of the following: horizontal, vertical, horizontal_secure, vertical_secure.
- **xgb_params**: The training parameters defined in this dict are passed to XGBoost as params.
- **xgb_options**: This dict contains other optional parameters passed to XGBoost. Currently, only early_stopping_rounds is supported.
- **client_ranks**: A dict that maps client name to rank.

Executor
--------

On the client side, following executor must be configured in executors,

``nvflare.app_opt.xgboost.histogram_based_v2.fed_executor.FedXGBHistogramExecutor``

Only one parameter is required for executor,

- **data_loader_id**: The component ID of Data Loader

Data Loader
-----------

On the client side, a data loader must be configured in the components. The SecureDataLoader can be used if the data is pre-processed. For example,

.. code-block:: json

    {
        "id": "dataloader",
        "path": "nvflare.app_opt.xgboost.histogram_based_v2.secure_data_loader.SecureDataLoader",
        "args": {
            "rank": 0,
            "folder": "/opt/dataset/vertical_xgb_data"
        }
    }


If the data requires any special processing, a custom loader can be implemented. The loader must implement the XGBDataLoader interface.


Job Example
===========

Vertical Training
-----------------

Here are the configuration files for a vertical secure training job. If encryption is not needed, just change the training_mode to vertical.

config_fed_server.json

.. code-block:: json

    {
        "format_version": 2,
        "num_rounds": 3,
        "workflows": [
            {
                "id": "xgb_controller",
                "path": "nvflare.app_opt.xgboost.histogram_based_v2.fed_controller.XGBFedController",
                "args": {
                    "num_rounds": "{num_rounds}",
                    "training_mode": "vertical_secure",
                    "xgb_options": {
                        "early_stopping_rounds": 2
                    },
                    "xgb_params": {
                        "max_depth": 3,
                        "eta": 0.1,
                        "objective": "binary:logistic",
                        "eval_metric": "auc",
                        "tree_method": "hist",
                        "nthread": 1
                    },
                    "client_ranks": {
                        "site-1": 0,
                        "site-2": 1
                    }
                }
            }
        ]
    }


config_fed_client.json

.. code-block:: json

    {
        "format_version": 2,
        "executors": [
            {
                "tasks": [
                    "config",
                    "start"
                ],
                "executor": {
                    "id": "Executor",
                    "path": "nvflare.app_opt.xgboost.histogram_based_v2.fed_executor.FedXGBHistogramExecutor",
                    "args": {
                        "data_loader_id": "dataloader"
                    }
                }
            }
        ],
        "components": [
            {
                "id": "dataloader",
                "path": "nvflare.app_opt.xgboost.histogram_based_v2.secure_data_loader.SecureDataLoader",
                "args": {
                    "rank": 0,
                    "folder": "/opt/dataset/vertical_xgb_data"
                }
            }
        ]
    }


Horizontal Training
-------------------

The configuration for horizontal training is the same as vertical except training_mode and the data loader must point to horizontal split data.

config_fed_server.json

.. code-block:: json

    {
        "format_version": 2,
        "num_rounds": 3,
        "workflows": [
            {
                "id": "xgb_controller",
                "path": "nvflare.app_opt.xgboost.histogram_based_v2.fed_controller.XGBFedController",
                "args": {
                    "num_rounds": "{num_rounds}",
                    "training_mode": "horizontal_secure",
                    "xgb_options": {
                        "early_stopping_rounds": 2
                    },
                    "xgb_params": {
                        "max_depth": 3,
                        "eta": 0.1,
                        "objective": "binary:logistic",
                        "eval_metric": "auc",
                        "tree_method": "hist",
                        "nthread": 1
                    },
                    "client_ranks": {
                        "site-1": 0,
                        "site-2": 1
                    },
                    "in_process": true
                }
            }
        ]
    }




config_fed_client.json

.. code-block:: json

    {
        "format_version": 2,
        "executors": [
            {
                "tasks": [
                    "config",
                    "start"
                ],
                "executor": {
                    "id": "Executor",
                    "path": "nvflare.app_opt.xgboost.histogram_based_v2.fed_executor.FedXGBHistogramExecutor",
                    "args": {
                        "data_loader_id": "dataloader",
                        "in_process": true
                    }
                }
            }
        ],
        "components": [
            {
                "id": "dataloader",
                "path": "nvflare.app_opt.xgboost.histogram_based_v2.secure_data_loader.SecureDataLoader",
                "args": {
                    "rank": 0,
                    "folder": "/data/xgboost_secure/dataset/horizontal_xgb_data"
                }
            }
        ]
    }
