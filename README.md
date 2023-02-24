**NV**IDIA **F**ederated **L**earning **A**pplication **R**untime **E**nvironment


[NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) enables researchers to collaborate and build AI models without sharing private data. 

NVIDIA FLARE is a standalone python library designed to enable federated learning amongst different parties using their local secure protected data for client-side training, at the same time it includes capabilities to coordinate and exchange progressing of results across all sites to achieve better global model while preserving data privacy. The participating clients can be in any part of the world. 

NVIDIA FLARE builds on a flexible and modular architecture and is abstracted through APIs allowing developers & researchers to customize their implementation of functional learning components in a Federated Learning paradigm. 

Learn more - [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html).


## Mission

Bring privacy preserved compute and Machine learning to data in a federated setting, keep it simple and production ready

## Installation

To install [the current release](https://pypi.org/project/nvflare), you can simply run:

```bash
pip install nvflare
```

## Quick Start
The quick start guide means to help the user get FLARE up & running quickly without introducing any advanced concepts. For more details, refer to [Getting Started](https://nvflare.readthedocs.io/en/main/getting_started.html). 
Since FLARE offers different modes of running the system, we only cover the simplest approaches here. This quick start guide uses the examples/hello-world/hello-numpy-sag as an example. You will find the details in the example's README.md file.
We also assume you have worked with Python, already set up the virtual env. If you are new to this, please refer [Getting Started](https://nvflare.readthedocs.io/en/main/getting_started.html).

#### Install NVFLARE
```
$ python3 -m pip install nvflare
```
Clone NVFLARE repo to get examples, switch main branch (latest stable branch)
```
$ git clone https://github.com/NVIDIA/NVFlare.git
$ cd NVFlare
$ git switch main
```

#### **Quick Start with Simulator**

```
nvflare simulator -w /tmp/nvflare/hello-numpy-sag -n 2 -t 2 examples/hello-world/hello-numpy-sag
```
Now you can watch the simulator run two clients (n=2) with two threads (t=2) and logs are saved in the /tmp/nvflare/hello-numpy-sag workspace.

#### **Quick start with POC mode**

Instead of using the simulator, you can simulate the real deployment with multiple processes via POC mode:
```shell
$ nvflare poc --prepare -n 2
$ nvflare poc --start -ex admin
```
From another terminal, start FLARE console:
```console
$ nvflare poc --start -p admin
```
Once FLARE Console started, you can check the status of the server.
```console
$ check_status server
$ submit_job hello-world/hello-numpy-sag
$ list_jobs
```
You can find out the other commands by using "?", you can download job results. use "bye" to exit.
```console
$ bye
```
You can use poc command to shut down clients and server

```shell
$ nvflare poc --stop  
``` 

#### **Quick Start Advanced Topics**
<details><summary>Quick start with production mode</summary>

Before you work in production mode, you need to first **provision**: a process to generate **startup kit**.
Startup kits are set of start scripts, configuration and certificates associated with different user, sites, server.
In this quick guide, we only show None-HA (non high availability mode), we will only have one FL server.

#### **provision**
<details><summary>Provision via CLI</summary>
```shell
$ cd /tmp
$ nvflare provision
```

select 2 for non-HA mode.  If you will generate a project.yml in the current directory. This will be the base configuration
files for provision. By default, the project.yml will have one server and two clients pre-defined

* server1
* site-1
* site-2

Now we are ready to provision,

```shell
$ cd /tmp
$ nvflare provision -p project.yml
```

it will generate startup kits in the following directory

```
/tmp/workspace/example_project/prod_00
```
</details>

#### **start Flare Server, Clients, Flare Console**
<details><summary>Starting different sub-systems</summary>
First start FL Server, open a new **terminal** for server

```shell
$ cd /tmp/workspace/example_project/prod_00
$ ./server1/startup/start.sh
```

Next start Site-1 and Site-2, open a new **terminal** for each site
in site-1 terminal:

```shell
$ cd /tmp/workspace/example_project/prod_00
$ ./site-1/startup/start.sh
```

in site-2 terminal:

```shell
$ cd /tmp/workspace/example_project/prod_00
$ ./site-2/startup/start.sh
```

Next finally for Flare console, open a new **terminal**

```shell
$ cd /tmp/workspace/example_project/prod_00
$ ./admin@nvidia.com/startup/fl_admin.sh
``` 
Once console started, you can use check-status command just like POC mode
</details>

#### **Provision and distributing startup kits via Flare Dashboard UI**
<details><summary>Starting FLARE Dashboard</summary>

Start the dashboard, then following the instructions. Once Dashboard started, you can setup project, invite users 
to participate, once user add the sites, you can approve the user and sites, then freeze the project. The user can download
the startup kits from the UI. 

```shell
 nvflare dashboard --start
```
</details>

</details>

## Getting Started
 For understand the concepts, details of above commands, examples, you can look into the following topics
 * [Getting Started](https://nvflare.readthedocs.io/en/main/getting_started.html)
 * [Examples](https://github.com/NVIDIA/NVFlare/tree/main/examples/)

## Release Highlights

### Release 2.3.0

* **Cloud Deployment Support**

  we expand the user support for cloud deployments in both Azure and AWS. 

* **Python version Support**

  Flare expand the python version support and dropped the Python 3.7 support, FLARE 2.3.0 will support 3.8, 3.9, 3.10
  
* **Flare API** 
  
  To add an improved version of Admin API to make it easier to use in notebook env. The current admin APIs still work.
  we potentially could deprecate the old Admin API in the future. FLARE API currently support selected commands of admin APIs
  For details of the Flare API, you can check [this notebook](https://github.com/NVIDIA/NVFlare/blob/dev/examples/tutorial/flare_api.ipynb)
  If you consider to migrating the existing Admin API to this new API, there is the [migration guide](https://nvflare.readthedocs.io/en/dev/real_world_fl/migrating_to_flare_api.html)

* **Sign custom code**
  Before a job is submitted to the server, the submitter's private key is used to sign
  each file's digest.  Each folder has one signature file, which maps file names to signatures
  of all files inside that folder.  The verification is performed at deployment time. 

* **Support client-site model initialization**
  Prior to FLARE 2.3.0, the model initialization is performed on the server-side.
  The model is either initialized via model file or custom model initiation code. Pre-defining a model file means to pre-generate and save the model file and then send over to the server.  
  Many users choose to run a custom model initialization code on server. But this, to some customers, could be a security risk.
  
  FLARE 2.3.0, introduce another way to initialize model on the client side, the Flare server will either can select 
  the initial model based on user-choose strategy. Here is the [example](https://github.com/NVIDIA/NVFlare/tree/dev/examples/hello-world/hello-pt) using client-side model. 
  You can read more about this feature in [FLARE documentation](TODO)

* **Tradition Machine Learning Examples** 
  We add several examples to support federated learning using traditional machine learning algorithms. 
  In particular, 
  * scikit-learn linear model ( linea and logistics regression )
  * scikit-learn SVM 
  * scikit-learn K-Means
  * XGBoost Random Forest

* **Vertical Learning**
  * **Private-Set Intersection** -- support multi-party private set intersection 
  * **Split-Learning** 

* **Research Area**
  * FedSM 
  * Data privacy detection tool

* **FLARE Communication**
  * we laid some ground work in FLARE communication layer. The new communication will be ready in next release.   
  
    
### Migrations tips
   [migrate to 2.2.1](docs/release_notes/2.2.1/migration_notes.md).
   [migrate to 2.3.0](docs/release_notes/2.3.0/migration_notes.md).


### Related talks and publications

For a list of talks, blogs, and publications related to NVIDIA FLARE, see [here](docs/publications_and_talks.md).

## Third party license

See 3rdParty folder for their license files.


