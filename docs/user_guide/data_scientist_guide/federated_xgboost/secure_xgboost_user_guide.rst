######################################
Federated Learning for XGBoost
######################################

Overview
========
This guide demonstrates how to use NVIDIA FLARE (NVFlare) to train XGBoost models in a federated learning environment. It showcases multiple collaboration strategies with varying levels of security.

NVFlare provides the following advantages:

- Secure training with Homomorphic Encryption (HE), protecting local histograms and gradients from the federated server and passive parties.
- Lifecycle management of XGBoost processes
- Reliable messaging that can overcome network glitches
- Training over complex networks with relays

This guide covers several federated XGBoost configurations:

- **Horizontal Collaboration**: Histogram-based and tree-based approaches (non-secure and secure)
- **Vertical Collaboration**: Histogram-based approach (non-secure and secure with Homomorphic Encryption)

What is XGBoost?
----------------
XGBoost (eXtreme Gradient Boosting) is a powerful machine learning algorithm that uses decision/regression trees for classification and regression tasks. It excels particularly with tabular data and remains widely used due to its:

- **High performance** on structured data
- **Explainability** of predictions
- **Computational efficiency**

These examples use `DMLC XGBoost <https://github.com/dmlc/xgboost>`_, which provides:

- GPU acceleration capabilities
- Distributed and federated learning support
- Optimized gradient boosting implementations

Federated Learning Modes
=========================

Horizontal Federated Learning
------------------------------
In horizontal collaboration, each participant has:

- **Same features** (columns) across all sites
- **Different data samples** (rows) at each site
- **Equal status** as label owners

**Example**: Multiple hospitals each have complete patient records (all features), but different patients.

Vertical Federated Learning
----------------------------
In vertical collaboration, each participant has:

- **Different features** (columns) at each site
- **Same data samples** (rows) across all sites
- **One "active party"** (label owner) and multiple "passive parties"

**Example**: A bank and a retailer have data about the same customers, but different attributes (financial vs. shopping behavior).

Supported Training Modes
-------------------------
When running with NVFlare, all XGBoost communications are local and messages are forwarded through NVFlare's communication infrastructure. The encryption is handled in XGBoost by encryption plugins, which are external components that can be installed at runtime.

NVFlare supports federated training in the following 4 modes:

1. **Horizontal without HE-based security protection** - Histogram-based or tree-based (tree-based is secured by removing "sum_hessian" values before transmission)
2. **Vertical without HE-based security protection** - Histogram-based
3. **Horizontal with HE** - Histogram-based (histograms secured against federated server)
4. **Vertical with HE** - Histogram-based (gradients secured against passive parties)

Security Risks and Mitigations
=======================

Risks
--------------

Federated XGBoost faces three main security risks:

1. **Model Statistics Leakage**: The default XGBoost JSON model contains "sum_hessian" statistics that enable model inversion attacks to recover data distributions. (Reference: `TimberStrike <https://arxiv.org/abs/2506.07605>`_)

2. **Histogram Leakage**: Gradient histograms can be exploited to reconstruct data distributions. The same model statistics of "sum_hessian" can be derived from histograms. (Reference: `TimberStrike <https://arxiv.org/abs/2506.07605>`_)

3. **Gradient Leakage**: Sample-wise gradients may reveal label information. (Reference: `SecureBoost <https://arxiv.org/abs/1901.08755>`_)

Attack Surface
--------------

The attack surface varies by collaboration mode and party role:

**Server**: Depending on the collaboration mode, the server may have access to 
1. The local model:
   - Horizontal tree-based:
      - **Model Statistics Leakage** over each client's data distribution
2. Local histograms:
   - Horizontal histogram-based / vertical histogram-based:
      - **Histogram Leakage** over each client / passive party's data distribution
3. Sample-wise gradients:
   - Vertical histogram-based:
      - **Gradient Leakage** over active party's label information

**Clients**: Depending on the collaboration mode, the clients may have access to
1. The aggregated global model:
   - Horizontal tree-based:
      - **Model Statistics Leakage** over global data distribution 
2. Global histograms:
   - Horizontal histogram-based:
      - **Histogram Leakage** over global data distribution 
3. Local histograms:
   - Vertical histogram-based: 
      - **Histogram Leakage** over each passive party's data distribution on active party
3. Sample-wise gradients:
   - **Gradient Leakage** over active party's label information on passive parties

Mitigations
------------------

The following table summarizes the available mitigations for different collaboration scenarios:

.. list-table:: Mitigations by Collaboration Mode
   :widths: 15 12 28 18 20 20
   :header-rows: 1

   * - Collaboration Mode
     - Algorithm
     - Data Exchange
     - Risk Mitigated
     - Security Measure
     - Implementation
   * - **Horizontal**
     - Tree-based
     - Clients send locally boosted trees to server; server combines and distributes trees back to clients
     - **Model statistics leakage** on both server and clients
     - Remove "sum_hessian" values from JSON model
     - Removed before clients send local trees to server
   * - **Horizontal**
     - Histogram-based
     - Clients send local histograms to server; server aggregates to global histogram and distributes it back to clients
     - **Histogram leakage** on server (client-side remain)
     - Encrypt histograms
     - Local histograms encrypted before transmission
   * - **Vertical**
     - Histogram-based
     - Active party computes gradients; routed by server, passive parties receive gradients, compute histograms, and send them back to active party through server
     - **Histogram leakage** on server (active party-side remain), **Gradient leakage** on both server and passive parties
     - **Primary**: Encrypt gradients; **Secondary**: Mask feature ownership in split values
     - Gradients encrypted before sending out to passive parties

**Notes:**

- **Vertical histogram-based**: 
  
  - **Primary goal**: Protect sample gradients from passive parties (critical)
  - **Secondary goal**: Hide split values from non-feature owners (desirable but lower risk)

- **The remaining two risks** will be discussed in the `Advanced Topics: Future Security Scenarios`_ section.

GPU Acceleration
================

Federated XGBoost supports two levels of GPU acceleration:

1. XGBoost GPU Training
-----------------------
Enable GPU-accelerated training by setting ``tree_method='gpu_hist'`` when initializing the XGBoost model.

- **Performance**: Up to **4.15x speedup** vs. CPU training (`GPU XGBoost Blog <https://developer.nvidia.com/blog/gradient-boosting-decision-trees-xgboost-cuda/>`_)

2. GPU-Accelerated Homomorphic Encryption (HE)
-----------------------------------------------
NVFlare provides GPU acceleration for HE operations using specialized encryption plugins.

- **Performance**: Up to **36.5x speedup** vs. CPU encryption (`NVFlare Secure XGBoost Blog <https://developer.nvidia.com/blog/security-for-data-privacy-in-federated-learning-with-cuda-accelerated-homomorphic-encryption-in-xgboost/>`_)

We will refer to these as "CPU/GPU XGBoost" and "CPU/GPU Encryption".

Security Implementation Matrix
==============================

The following table shows which security measures are supported across different hardware configurations:

.. list-table:: Security Implementation Matrix
   :widths: 18 30 13 13 13 13
   :header-rows: 1

   * - Collaboration Mode
     - Security Goal
     - CPU XGBoost + CPU Encryption
     - CPU XGBoost + GPU Encryption
     - GPU XGBoost + CPU Encryption
     - GPU XGBoost + GPU Encryption
   * - **Horizontal**
     - Histogram protection against server
     - ✅
     - N/A\*
     - ✅
     - N/A\*
   * - **Vertical**
     - **Primary**: Gradient protection
     - ✅
     - ✅
     - ✅
     - ✅
   * - **Vertical**
     - **Secondary**: Split value masking
     - ✅
     - ✅
     - ❌
     - ❌

**\*Note**: Horizontal histogram encryption is not computationally intensive (encrypting histogram vectors), so GPU encryption is not needed.

**Implementation Notes**:

- **Vertical mode primary goal** (gradient protection): Fully supported across all configurations
- **Vertical mode secondary goal** (split value masking): Only supported with CPU XGBoost

Advanced Topics: Future Security Scenarios
===========================================

The following security scenarios are not currently implemented in our solution. Users should be aware that **plaintext histogram communication** can reveal data distribution information, which may enable data reconstruction attacks as stated above. On the other hand, similar statistics can also be derived from common practices such as `federated statistics <https://nvflare.readthedocs.io/en/main/examples/federated_statistics_overview.html>`_. As the attack potency depends on multiple factors including data complexity, model hyperparameters, and the data distribution information that can be utilized, the corresponding indications of a certain type of attack can vary significantly. This is still an open and active research area.

Potential Future Enhancements to Protect Against All Parties
-------------------------------------------------------------

.. list-table:: Future Security Scenarios
   :widths: 15 12 20 25 28
   :header-rows: 1

   * - Collaboration Mode
     - Algorithm
     - Remaining Security Risk
     - Possible Approach
     - Challenges
   * - **Horizontal**
     - Histogram-based
     - Histogram leakage over global data distribution on clients (in addition to server as addressed above)
     - Confidential computing, advanced HE
     - HE compatibility issue [*]_ with server performing calculations and distributing only final splits
   * - **Vertical**
     - Histogram-based
     - Histogram leakage over each passive party's data distribution on active party (in addition to Histogram leakage on server, and Gradient leakage on server and passive parties as addressed above)
     - Local data preprocessing and anonymization, confidential computing, advanced HE
     - HE compatibility issue [*]_ with passive parties performing calculations and sending only final splits

.. [*] **HE Compatibility Challenge**: Current Homomorphic Encryption schemes do not efficiently support operations like ciphertext division and argmax, which are required for performing split calculations on encrypted data. Advanced HE features are needed to support approaches that "perform calculations until splits on the server/passive parties."

Prerequisites
=============

Required Python Packages
------------------------

NVFlare 2.7.2 or above,

.. code-block:: bash

    pip install nvflare~=2.7.2

Federated Secure XGBoost, which can be installed from the binary build using this command,

.. code-block:: bash

    pip install https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/federated-secure/xgboost-2.2.0.dev0%2B4601688195708f7c31fcceeb0e0ac735e7311e61-py3-none-manylinux_2_28_x86_64.whl

.. note::

   The xgboost build environment may depend on specific numpy versions that require Python < 3.12.

or in case you need to get the most current build of XGBoost,

.. code-block:: bash

    pip install https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/federated-secure/`curl -s https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/federated-secure/meta.json | grep -o 'xgboost-2\.2.*whl'|sed -e 's/+/%2B/'`

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

Building Encryption Plugins
---------------------------

The secure training requires encryption plugins, which need to be built from the source code
for your specific environment.

To build the plugins, check out the NVFlare source code from https://github.com/NVIDIA/NVFlare and follow the
instructions in :github_nvflare_link:`this document <integration/xgboost/encryption_plugins/README.md>`.

.. _xgb_provisioning:

NVFlare Provisioning
--------------------
For horizontal secure training, the NVFlare system must be provisioned with a homomorphic encryption context. The HEBuilder in ``project.yml`` is used to achieve this.
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
Data must be properly formatted for federated XGBoost training based on the collaboration mode.

Horizontal Training
-------------------
For horizontal training, the datasets on all clients must share the same columns (features). Each client has different data samples (rows).

Vertical Training
-----------------
For vertical training, the datasets on all clients contain different columns (features), but must share overlapping rows (data samples). The label column is typically assigned to site-1 (the "active party") by default.

For more details on vertical split preprocessing, refer to the :github_nvflare_link:`Vertical XGBoost Example <examples/advanced/vertical_xgboost>`.

XGBoost Plugin Configuration
============================
XGBoost requires an encryption plugin to handle secure training. Two plugins are available:

- **cuda_paillier**: The default plugin. This plugin uses GPU for cryptographic operations.
- **nvflare**: This plugin forwards data locally to NVFlare process for encryption.

.. note::

   All clients must use the same plugin. When different plugins are used in different clients,
   the behavior of federated XGBoost is undetermined, which can cause the job to crash.

The **cuda_paillier** plugin requires NVIDIA GPUs that support compute capability 7.0 or higher. Also, CUDA
12.2 or 12.4 must be installed. Please refer to https://developer.nvidia.com/cuda-gpus for more information.

The two included plugins are only different in vertical secure training. For horizontal secure training, both
plugins work exactly the same by forwarding the data to NVFlare for encryption.

Plugin Configuration by Training Mode
--------------------------------------

Vertical (Non-secure)
~~~~~~~~~~~~~~~~~~~~~
No plugin is needed.

Horizontal (Non-secure)
~~~~~~~~~~~~~~~~~~~~~~~
No plugin is needed.

Vertical Secure
~~~~~~~~~~~~~~~
Both plugins can be used for vertical secure training.

The default cuda_paillier plugin is preferred because it uses GPU for faster cryptographic operations.

.. note::

    **cuda_paillier** plugin requires NVIDIA GPUs that support compute capability 7.0 or higher. Please refer to https://developer.nvidia.com/cuda-gpus for more information.

If you see the following errors in the log, it means either no GPU is detected or the GPU does not meet the requirements:

::

    CUDA runtime API error no kernel image is available for execution on the device at line 241 in file /my_home/nvflare-internal/processor/src/cuda-plugin/paillier.h
    2024-07-01 12:19:15,683 - SimulatorClientRunner - ERROR - run_client_thread error: EOFError:


In this case, the nvflare plugin can be used to perform encryption on CPUs, which requires the ipcl-python package.
The plugin can be configured in the ``local/resources.json`` file on clients:

.. code-block:: json

    {
        "federated_plugin": {
            "name": "nvflare",
            "path": "/opt/libs/libnvflare.so"
        }
    }

Where **name** is the plugin name and **path** is the full path of the plugin including the library file name.
The **path** is optional, the default value is the library distributed with NVFlare for the plugin.

The following environment variables can be used to override the values in the JSON,

.. code-block:: bash

    export NVFLARE_XGB_PLUGIN_NAME=nvflare
    export NVFLARE_XGB_PLUGIN_PATH=/opt/libs/libnvflare.so

.. note::

   When running with the NVFlare simulator, the plugin must be configured using environment variables,
   as it does not support resources.json.

Horizontal Secure
~~~~~~~~~~~~~~~~~
The plugin setup is the same as vertical secure.

This mode requires the tenseal package for all plugins.
The provisioning of NVFlare systems must include tenseal context.
See :ref:`xgb_provisioning` for details.

For simulator, the tenseal context generated by provisioning needs to be copied to the startup folder,

``simulator_workspace/startup/client_context.tenseal``

For example,

.. code-block:: bash

    nvflare provision -p secure_project.yml -w /tmp/poc_workspace
    mkdir -p /tmp/simulator_workspace/startup
    cp /tmp/poc_workspace/example_project/prod_00/site-1/startup/client_context.tenseal /tmp/simulator_workspace/startup

The server_context.tenseal file is not needed.

Job Configuration
=================
.. _secure_xgboost_controller:

Controller
----------

On the server side, the following controller must be configured in workflows,

``nvflare.app_opt.xgboost.histogram_based_v2.fed_controller.XGBFedController``

Even though the XGBoost training is performed on clients, the parameters are configured on the server so all clients share the same configuration. 
XGBoost parameters are defined here, https://xgboost.readthedocs.io/en/stable/python/python_intro.html#setting-parameters

- **num_rounds**: Number of training rounds.
- **data_split_mode**: Same as XGBoost data_split_mode parameter, 0 for horizontal, 1 for vertical.
- **secure_training**: If true, XGBoost will train in secure mode using the plugin.
- **xgb_params**: The training parameters defined in this dict are passed to XGBoost as **params**, the boost parameter.
- **xgb_options**: This dict contains other optional parameters passed to XGBoost. Currently, only **early_stopping_rounds** is supported.
- **client_ranks**: A dict that maps client name to rank.

Executor
--------

On the client side, the following executor must be configured in executors,

``nvflare.app_opt.xgboost.histogram_based_v2.fed_executor.FedXGBHistogramExecutor``

Only one parameter is required for executor,

- **data_loader_id**: The component ID of Data Loader

Data Loader
-----------

On the client side, a data loader must be configured in the components. The CSVDataLoader can be used if the data is pre-processed. For example,

.. code-block:: json

    {
        "id": "dataloader",
        "path": "nvflare.app_opt.xgboost.histogram_based_v2.csv_data_loader.CSVDataLoader",
        "args": {
            "folder": "/opt/dataset/vertical_xgb_data"
        }
    }


If the data requires any special processing, a custom loader can be implemented. The loader must implement the XGBDataLoader interface.


Job Examples
============

Vertical Training
-----------------

Here are the configuration files for a vertical secure training job. If encryption is not needed, just change the ``secure_training`` arg to false.

.. code-block::

    :caption: config_fed_server.json

    {
        "format_version": 2,
        "num_rounds": 3,
        "workflows": [
            {
                "id": "xgb_controller",
                "path": "nvflare.app_opt.xgboost.histogram_based_v2.fed_controller.XGBFedController",
                "args": {
                    "num_rounds": "{num_rounds}",
                    "data_split_mode": 1,
                    "secure_training": true,
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



.. code-block::

    :caption: config_fed_client.json

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
                "path": "nvflare.app_opt.xgboost.histogram_based_v2.csv_data_loader.CSVDataLoader",
                "args": {
                    "folder": "/opt/dataset/vertical_xgb_data"
                }
            }
        ]
    }


Horizontal Training
-------------------

The configuration for horizontal training is the same as vertical except ``data_split_mode`` is 0 and the data loader must point to horizontal split data.

.. code-block:: json
   :caption: config_fed_server.json

    {
        "format_version": 2,
        "num_rounds": 3,
        "workflows": [
            {
                "id": "xgb_controller",
                "path": "nvflare.app_opt.xgboost.histogram_based_v2.fed_controller.XGBFedController",
                "args": {
                    "num_rounds": "{num_rounds}",
                    "data_split_mode": 0,
                    "secure_training": true,
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




.. code-block:: json
   :caption: config_fed_client.json

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
                "path": "nvflare.app_opt.xgboost.histogram_based_v2.csv_data_loader.CSVDataLoader",
                "args": {
                    "folder": "/data/xgboost_secure/dataset/horizontal_xgb_data"
                }
            }
        ]
    }

Pre-Trained Models
==================
To continue training using a pre-trained model, the model can be placed in the job folder with the path and name
of ``custom/model.json``.

Every site should share the same ``model.json``. The result of previous training with the same dataset can be used as the input model.

When a pre-trained model is detected, NVFlare prints following line in the log:

::

    INFO - Pre-trained model is used: /tmp/nvflare/poc/example_project/prod_00/site-1/startup/../996ac44f-e784-4117-b365-24548f1c490d/app_site-1/custom/model.json


Performance Tuning
==================
Timeouts
--------
For secure training, the HE operations are very slow. If a large dataset is used, several timeout values need
to be adjusted.

The XGBoost messages are transferred between client and server using
Reliable Messages (:class:`ReliableMessage<nvflare.apis.utils.reliable_message.ReliableMessage>`). The following parameters
in executor arguments control the timeout behavior:

    - **per_msg_timeout**: Timeout in seconds for each message.
    - **tx_timeout**: Timeout for the whole transaction in seconds. This is the total time to wait for a response, accounting for all retry attempts.

.. code-block::
   :caption: config_fed_client.json

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
                        "per_msg_timeout": 300.0,
                        "tx_timeout": 900.0,
                        "in_process": true
                    }
                }
            }
        ],
        ...
    }

Number of Clients
-----------------
The default configuration can only handle 20 clients. This parameter needs to be adjusted if more clients are involved in the training:

.. code-block::
   :caption: config_fed_client.json

    {
        "format_version": 2,
        "num_rounds": 3,
        "rm_max_request_workers": 100,
        ...
    }


Additional Resources
====================

- `NVIDIA FLARE Documentation <https://nvflare.readthedocs.io/>`_
- `XGBoost Documentation <https://xgboost.readthedocs.io/>`_
- `GPU XGBoost Blog <https://developer.nvidia.com/blog/gradient-boosting-decision-trees-xgboost-cuda/>`_
- `NVFlare Secure XGBoost Blog <https://developer.nvidia.com/blog/security-for-data-privacy-in-federated-learning-with-cuda-accelerated-homomorphic-encryption-in-xgboost/>`_
