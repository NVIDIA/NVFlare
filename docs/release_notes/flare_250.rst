**************************
What's New in FLARE v2.5.0
**************************

User Experience Improvements
============================
NVFlare 2.5.0 offers several new sets of APIs that allows for end-to-end ease of use that can greatly improve researcher and data
scientists' experience working with FLARE. The new API covers client, server and job construction with end-to-end pythonic user experience.

Model Controller API
--------------------
The new Model Controller API greatly simplifies the experience of developing new federated learning workflows. Users can simply subclass
the ModelController to develop new workflows. The new API doesn't require users to know the details of NVFlare constructs except for FLModel
class, where it is simply a data structure that contains model weights, optimization parameters and metadata. 

You can easily construct a new workflow with basic python code, and when ready, the send_and_wait() communication function is all you need for
communication between clients and server. 

Client API
----------
We introduced another :ref:`client_api` implementation,
:class:`InProcessClientAPIExecutor<nvflare.app_common.executors.in_process_client_api_executor.InProcessClientAPIExecutor>`.
This has the same interface and syntax of the previous Client API using
:class:`SubprocessLauncher<nvflare.app_common.launchers.subprocess_launcher.SubprocessLauncher>`, except all communication is in memory. 

Using this in-process client API, we build a :class:`ScriptExecutor<nvflare.app_common.executors.script_executor.ScriptExecutor>`,
which is directly used in the new Job API.

Compared with SubProcessLauncherClientAPI, the in-process client API offers better efficiency and is easier to configure. All
the operations will be carried out within the memory space of the executor.  

SubProcessLauncherClientAPI can be used for cases where a separate training process is required.

Job API
-------
The new Job API, or :ref:`fed_job_api`, combined with Client API and Model Controller API, will give users an end-to-end pythonic
user experience. The Job configuration, required prior to the current release, can now be directly generated automatically, so the
user doesn't need to edit the configuration files manually. 

We provide many examples to demonstrate the power of the new Job APIs making it very easy to experiment with new federated
learning algorithms or create new applications. 

Flower Integration
==================
Integration between NVFlare and the `Flower <https://flower.ai/>`_ framework aims to provide researchers the ability to leverage
the strengths of both frameworks by enabling Flower projects to seamlessly run on top of NVFlare. Through the seamless
integration of Flower and FLARE, applications crafted within the Flower framework can effortlessly operate within the FLARE runtime
environment without necessitating any modifications. This initial integration streamlines the process, eliminating complexities and
ensuring smooth interoperability between the two platforms, thus enhancing the overall efficiency and accessibility of FL applications.
Please find details `here <https://arxiv.org/abs/2407.00031>`__. A hello-world example is available
:github_nvflare_link:`here <examples/hello-world/hello-flower>`.

Secure XGBoost
==============
The latest features from XGBoost introduced the support for secure federated learning via homomorphic encryption. For vertical federated
XGBoost learning, the gradients of each sample are protected by encryption such that the label information
will not be leaked to unintended parties; while for horizontal federated XGBoost learning, the local gradient histograms will not be
learnt by the central aggregation server. 

With our encryption plugins working with XGBoost, NVFlare now supports all secure federated schemes for XGBoost model training, with
both CPU and GPU.

Tensorflow support
==================
With community contributions, we add FedOpt, FedProx and Scaffold algorithms using Tensorflow to create parity with Pytorch. You
can them :github_nvflare_link:`here <nvflare/app_opt/tf>`.

FOBS Auto register 
==================


New Examples
============
Secure Federated Kaplan-Meier Analysis
--------------------------------------
The :github_nvflare_link:`Secure Federated Kaplan-Meier Analysis via Time-Binning and Homomorphic Encryption example <examples/advanced/kaplan-meier-he>`
illustrates two features:

  - How to perform Kaplan-Meier survival analysis in a federated setting without and with secure features via time-binning and Homomorphic Encryption (HE).
  - How to use the Flare ModelController API to contract a workflow to facilitate HE under simulator mode.


Federated Logistic Regression with NR optimization
--------------------------------------------------
The :github_nvflare_link:`Federated Logistic Regression with Second-Order Newton-Raphson optimization example <examples/advanced/lr-newton-raphson>`
shows how to implement a federated binary classification via logistic regression with second-order Newton-Raphson optimization.

BioNemo example for Drug Discovery
----------------------------------
`BioNeMo <https://www.nvidia.com/en-us/clara/bionemo/>`_ is NVIDIA's generative AI platform for drug discovery.
We included several examples of running BioNeMo in a federated learning environment using NVFlare:

  - The :github_nvflare_link:`task fitting example <examples/advanced/bionemo/task_fitting/README.md>` includes a notebook that
  shows how to obtain protein-learned representations in the form of embeddings using the ESM-1nv pre-trained model. The
  model is trained with NVIDIA's BioNeMo framework for Large Language Model training and inference.
  - The :github_nvflare_link:`downstream example <examples/advanced/bionemo/downstream/README.md>` shows three different downstream
  tasks for fine-tuning a BioNeMo ESM-style model.

Hearchical Federated Statistics
--------------------------------
:github_nvflare_link:`Hierarchical Federated Statistics <examples/advanced/federated-statistics/hierarchical_stats>` is helpful when there
are multiple organizations involved.  For example, in the medical device applications, the medical devices usage statistics can be
viewed from both device,device-hosting site and hospital, manufacturing point of views.
Manufacturers would like to see the usage stats of their product (device) in different sites and hospitals. While hospitals
like to see overall stats of devices including different products from different manufacturers. In such a case, the hierarchical
federated stats will be very helpful. 

FedAvg Early Stopping Example
------------------------------
The FedAvg Early Stopping example tries to demonstrate that with the new server-side model controller API, its very easy to change the control conditions
and adjust workflows with a few lines of python code. 
Tensorflow Algorithms & Examples
FedOpt, FedProx, Scaffold implementation for Tensorflow

End-to-end Federated XGBoost examples
-------------------------------------
In this example, we try to show that end-to-end process of feature engineering, pre-processing and training in federated settings. You
can use FLARE to perform federated ETL and then training. 

Developer Tutorial Page
=======================
We made the develop FL algorithm really simple via the newly developed APIs. To let users quickly learn Federated Learning with FLARE,
we developed a tutorial web page with both code and video to interactively learn how to convert and run FL in a few minutes. We also
created a tutorial catalog to help user search and find the interested examples

