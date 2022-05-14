**NV**IDIA **F**ederated **L**earning **A**pplication **R**untime **E**nvironment


[NVIDIA FLARE](https://nvidia.github.io/NVFlare) enables researchers to collaborate and build AI models without sharing private data. 

NVIDIA FLARE is a standalone python library designed to enable federated learning amongst different parties using their local secure protected data for client-side training, at the same time it includes capabilities to coordinate and exchange progressing of results across all sites to achieve better global model while preserving data privacy. The participating clients can be in any part of the world. 

NVIDIA FLARE builds on a flexible, event-driven and modular architecture and is abstracted through APIs allowing developers & researchers to customize their implementation of functional learning components in a Federated Learning paradigm. 

Learn more - [NVIDIA FLARE](https://nvidia.github.io/NVFlare).


## Installation

To install [the current release](https://pypi.org/project/nvflare), you can simply run:

```bash
pip install nvflare
```

## Quick Start

This section provides a starting point for new users to start NVIDIA FLARE.
Users can go through the [Example Apps for NVIDIA FLARE](#https://nvflare.readthedocs.io/en/dev-2.1/example_applications.html#example-apps) and get familiar with how NVIDIA FLARE is designed,
operates and works.

Each example introduces concepts about NVIDIA FLARE, while showcasing how some popular libraries and frameworks can
easily be integrated into the Federated learning (FL) process.

To start to experiment with FL without setting up the many sites/servers, we provided a POC mode to simulate the FL
environments with local host that can simulate different sites/clients and server.

### Setting Up the Application Environment in POC Mode
> **warning**:
> POC mode is not intended to be secure and should not be run in any type of production environment or any environment
    where the server's ports are exposed. For actual deployment and even development, it is recommended to use a
    [secure provisioned setup](https://nvflare.readthedocs.io/en/dev-2.1/user_guide/overview.html#provisioned-setup)

To get started with a proof of concept (POC) setup after [installation](https://nvflare.readthedocs.io/en/dev-2.1/installation.html#installation), run this command to generate a poc folder
with a server, two clients, and one admin:

```shell
    $ poc -n 2
```
Copy necessary files (the exercise code in the examples directory of the NVFlare repository) to a working folder (upload
folder for the admin):

```bash
$ mkdir -p poc/admin/transfer
$ cp -rf NVFlare/examples/* poc/admin/transfer

```

### Starting the Application Environment in POC Mode

Once you are ready to start the FL system, you can run the following commands to start all the different parties (it is
recommended that you read into the specific [Example Apps for NVIDIA FLARE](#https://nvflare.readthedocs.io/en/dev-2.1/example_applications.html#example-apps) first, then start the FL
system to follow along at the parts with admin commands for you to run the example app).

FL systems usually have a server, and multiple clients. In case of high availability (HA), there is also a sub-system called overseer.
If you don't understand, that's ok, we will explain more in details. But for now, just follow the steps. 

We start the overseer first:
```shell

    $ ./poc/overseer/startup/start.sh

```

Once the overseer is running, you can start the server and clients in different terminals (make sure your terminals are
using the environment with NVIDIA FLARE [installed](#installation).

Open a new terminal and start the server:
```shell

    $ ./poc/server/startup/start.sh

```

Once the server is running, open a new terminal and start the first client:

```shell

    $ ./poc/site-1/startup/start.sh

```

Open another terminal and start the second client:

```shell

    $ ./poc/site-2/startup/start.sh

```

In one last terminal, start the admin:

```shell

$ ./poc/admin/startup/fl_admin.sh localhost

```

This will launch a command prompt where you can input admin commands to control and monitor many aspects of
the FL process.

For anything more than the most basic proof of concept examples, it is recommended that you use a
[secure provisioned setup](https://nvflare.readthedocs.io/en/dev-2.1/user_guide/overview.html#provisioned-setup).

### Submit Jobs

to run some example, we need additional dependencies. Here we will use hello-numpy-sag as example, we need to install numpy. 
assuming the venv is nvflare-env, you can do this by

```shell
 (nvflare-env) $ python3 -m pip install numpy
```
From admin console, you can find out the commands available by typing
```Admin Console
    ? 
```
to submit job,
```Admin Console
   submit_job hello-numpy-sag
```
you can check the status of server and jobs with various commands available from Admin Console, such as
```Admin Console
   check_status server
```
```Admin Console
   check_status client
```
```Admin Console
   list_jobs 
```
Once the job completed, you are done with the first example. If you experience errors, 
you can check the log.txt under job number directories for server and clients respectively.   

To learn more, please check [NVIDIA FLARE documentation](https://nvflare.readthedocs.io/en/dev-2.1/quickstart.html)

## Third party license

See 3rdParty folder for their license files.


