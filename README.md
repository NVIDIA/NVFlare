**NV**IDIA **F**ederated **L**earning **A**pplication **R**untime **E**nvironment


[NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) enables researchers to collaborate and build AI models without sharing private data. 

NVIDIA FLARE is a standalone python library designed to enable federated learning amongst different parties using their local secure protected data for client-side training, at the same time it includes capabilities to coordinate and exchange progressing of results across all sites to achieve better global model while preserving data privacy. The participating clients can be in any part of the world. 

NVIDIA FLARE builds on a flexible and modular architecture and is abstracted through APIs allowing developers & researchers to customize their implementation of functional learning components in a Federated Learning paradigm. 

Learn more - [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html).


## Installation

To install [the current release](https://pypi.org/project/nvflare), you can simply run:

```bash
pip install nvflare
```

## Quick Start

 * [Getting Started](https://nvflare.readthedocs.io/en/main/getting_started.html)
 * [Examples](https://github.com/NVIDIA/NVFlare/tree/main/examples/)

## Release Highlights

### Release 2.2.1

* [FL Simulator]( https://nvflare.readthedocs.io/en/main/user_guide/fl_simulator.html) -- 
  A lightweight simulator of a running NVFLARE FL deployment. It allows researchers to test and debug their application without provisioning 
 a real project. The FL jobs run on a server and multiple clients in the same process but 
 in a similar way to how it would run in a real deployment. Researchers can quickly 
 build out new components and jobs that can then be directly used in a real production deployment.

* [FLARE Dashboard](https://nvflare.readthedocs.io/en/main/user_guide/dashboard_api.html) --
  NVFLARE's web UI. In its initial incarnation, the Flare Dashboard is used to help
  project setup, user registration, startup kits distribution and dynamic provisions.  

* [Site-policy management](https://nvflare.readthedocs.io/en/main/user_guide/site_policy_management.html) -- 
  Prior to NVFLARE 2.2, all policies (resource management, authorization and privacy protection, logging configurations) 
  can only be defined by the Project Admin during provision time; and authorization policies are centrally enforced by the FL Server.
  NVFLARE 2.2 makes it possible for each site to define its own policies in the following areas:
  * Resource Management: the configuration of system resources that are solely the decisions of local IT.
  * Authorization Policy: local authorization policy that determines what a user can or cannot do on the local site. see related [Federated Authorization](https://nvflare.readthedocs.io/en/main/user_guide/federated_authorization.html)
  * Privacy Policy: local policy that specifies what types of studies are allowed and how to add privacy protection to the learning results produced by the FL client on the local site.
  * Logging Configuration: each site can now define its own logging configuration for system generated log messages.
  
* [Federated XGBoost](<https://github.com/NVIDIA/NVFlare/tree/main/examples/xgboost>) --
  We developed federated XGBoost for data scientists to perform machine learning on tabular data with popular tree-based method. In this release, we provide several 
  approaches for the horizontal federated XGBoost algorithms. 
  * Histogram-based Collaboration -- leverages recently released (XGBoost 1.7.0) federated versions of open-source XGBoost histogram-based distributed training algorithms that operates in a federated manner achieving identical results as centralized training (trees trained on global data information).
  * Tree-based Collaboration -- individual trees are independently trained on each client's local data without aggregating the global sample gradient histogram information. 
  Trained trees are collected and passed to the server / other clients for aggregation and further boosting rounds.
  
* [Federated Statistics](<https://github.com/NVIDIA/NVFlare/tree/main/examples/federated_statistics>) -- 
  built-in federated statistics operators that can generate global statistics based on local client side statistics. 
  The results, for all features of all datasets at all sites as well as global aggregates, can be visualized via the visualization utility in the notebook.  

* [MONAI Integration](<https://github.com/NVIDIA/NVFlare/tree/main/integration/monai/README.md>)
  In 2.2 release, we provided two implementations by leveraging Monai "bundle"
  * Monai [ClientAlgo](https://docs.monai.io/en/latest/fl.html#monai.fl.client.ClientAlgo) Integration -- enable running MONAI bundles directly in a federated setting using NVFLARE
  * Monai [Statistics](https://docs.monai.io/en/latest/fl.html#monai.fl.client.ClientAlgoStats) Integration -- through NVFLARE Federated Statistics we can generate, compare and visualize all client's data statistics generated from Monai summary statistics

* Tools and Production Support
  * [Improved POC command](https://nvflare.readthedocs.io/en/main/user_guide/poc_command.html) 
  * [Dynamic Provision](https://nvflare.readthedocs.io/en/main/user_guide/dynamic_provisioning.html)
  * [Docker Compose](https://nvflare.readthedocs.io/en/main/user_guide/docker_compose.html)
  * [Preflight Check](https://nvflare.readthedocs.io/en/main/user_guide/preflight_check.html#nvidia-flare-preflight-check)

    
### Migrations tips 

   To migrate from releases prior to 2.2.1, here are few notes that might help
   [migrate to 2.2.1](docs/release_notes/2.2.1/migration_notes.md)
   

## Third party license

See 3rdParty folder for their license files.


