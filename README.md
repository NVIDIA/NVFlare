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
The quick start guide means to help the user get FLARE up & running quickly without introducing any advanced concepts. For more details, refer to [Getting Started](https://nvflare.readthedocs.io/en/main/getting_started.html)
Since FLARE offers different modes of running the system, we only cover the simplest approaches here. This quick start guide uses the examples/hello-world/hello-numpy-sag as an example. You will find the details in the example's README.md file.
We also assume you have worked with Python, already set up the virtual env. If you are new to this, please refer :ref:`getting_started`.

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
#### Quick start with CLI
Create a temp directory as workspace and install requirements/dependencies:
```
$ mkdir -p /tmp/nvflare
$ python3 -m pip install -r examples/hello-world/hello-numpy-sag/requirements.txt
```
* **Quick Start with Simulator**

```
nvflare simulator -w /tmp/nvflare/ -n 2 -t 2 examples/hello-world/hello-numpy-sag
```
Now you can watch the simulator run two clients (n=2) with two threads (t=2) and logs are saved in the /tmp/nvflare workspace.

* **Quick start with POC mode**

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
You can use poc command to shutdown clients and server

```shell
$ nvflare poc --stop  
``` 

## Getting Started
 For understand the concepts, details of above commands, examples, you can look into the following topics
 * [Getting Started](https://nvflare.readthedocs.io/en/main/getting_started.html)
 * [Examples](https://github.com/NVIDIA/NVFlare/tree/main/examples/)

## Release Highlights

### Release 2.3.0

* Cloud Deployment Support

  we expand the user support for cloud deployments in both Azure and AWS. 

* Python version Support

  Flare expand the python version support to include 3.9 and 3.10. We also dropped the Python 3.7 support
  
* Flare API 
  
  To add an improved version of Admin API to make it easier to use in notebook env. The current admin APIs still work.
  we potentially could deprecate the old Admin API in the future. FLARE API currently support selected commands of admin APIs 

* sign custom code
  Before a job is submitted to the server, the submitter's private key is used to sign
  each file's digest.  Each folder has one signature file, which maps file names to signatures
  of all files inside that folder.  The verification is performed at deployment time. 

* Support client-site model initialization
  In most of federated deep learning Scatter and Gather pattern (SAG) workflow pattern, 
  we requires all the clients starts with global initial mode. Prior to FLARE 2.3.0, the model initialization is performed 
  on the server-side, where the model file or custom model initiation code is performed on FL server.
  Pre-defining a model file seems to be a hassle. Many choose to run a custom model initialization code on server. 
  But this, to some customers, could be a security risk.
  
  FLARE 2.3.0, allows user to initialize model on the client side, the Flare server will either can select the initial model based
  on user-choose strategy (random one or some aggregation)

* Tradition Machine Learning Examples 
  We add several examples to support federated learning using traditional machine learning algorithms. 
  In particular, 
  * scikit-learn linear model ( linea and logistics regression )
  * scikit-learn SVM 
  * scikit-learn K-Means
  * XGBoost Random Forest

* Vertical Learning
  * Private-Set Intersection -- support multi-party private set intersection 
  * Split-Learning 

* Research Area
  * FedSM 
  * Data privacy detection tool

* FLARE Communication
  * we laid some ground work in FLARE communication layer. The new communication will be ready in next release.   
  
    
### Migrations tips 

   To migrate from releases prior to 2.2.1, here are few notes that might help
   [migrate to 2.2.1](docs/release_notes/2.2.1/migration_notes.md).

### Related talks and publications

For a list of talks, blogs, and publications related to NVIDIA FLARE, see [here](docs/publications_and_talks.md).

## Third party license

See 3rdParty folder for their license files.


