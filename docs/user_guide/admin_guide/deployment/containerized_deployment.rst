.. _containerized_deployment:

########################
Containerized Deployment
########################

Containerized Deployment with Docker
====================================

Docker support has two common uses:

- Run FLARE inside a generic development container for simulation, notebooks,
  POC mode, or manual experiments. In this pattern, Docker provides the Python
  environment and filesystem, while FLARE processes still run normally inside
  that one container.
- Prepare provisioned startup kits for Docker runtime execution. This is the
  recommended path when FLARE parent server/client processes should run in
  Docker and launch server/client jobs as separate Docker containers.

The current Docker runtime workflow is:

1. Build a parent image with NVFlare installed.
2. Build a job image with NVFlare and workload dependencies installed.
3. Provision server, client, and admin startup kits.
4. Run :ref:`deploy_prepare_command` on each server or client startup kit that
   should run in Docker.
5. Start prepared server/client kits with ``startup/start_docker.sh``.
6. Submit jobs whose ``meta.json`` contains Docker settings in
   :ref:`launcher_spec`.

For a runnable end-to-end workflow, see the
:github_nvflare_link:`Docker job launcher example <examples/docker>`.

Prerequisites
-------------
Before starting with containerized deployment, ensure you have:

1. Docker installed on your system
2. NVIDIA Container Toolkit installed for GPU support
3. System requirements met as per the `NVIDIA Container Toolkit Install Guide <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_
4. Provisioned server/client startup kits when preparing a Docker runtime
   deployment

Running NVIDIA FLARE in a Docker container provides several benefits:

- Consistent environment across different systems
- Easy dependency management
- Repeatable runtime preparation
- Isolated execution environment
- GPU support through NVIDIA Container Toolkit

Parent and Job Images
---------------------

The Docker runtime separates parent containers from job containers:

- The parent image runs the long-lived FLARE server process or client process
  from a prepared startup kit.
- The job image runs server job and client job processes launched by
  ``DockerJobLauncher``.

The parent image is configured in the runtime ``docker.yaml`` used by
``nvflare deploy prepare``. The job image is configured in the submitted job's
``meta.json`` under ``launcher_spec``. You can use the same image for both
roles, but keeping them separate is often cleaner: the parent image needs the
FLARE runtime and Docker SDK access, while the job image needs the workload
frameworks and training dependencies.

Docker Runtime Workflow
-----------------------

Build or publish the images that your sites will use. The runnable
:github_nvflare_link:`Docker job launcher example <examples/docker>` builds an
example parent image, ``nvflare-site:latest``, and an example job image,
``nvflare-job:latest``:

.. code-block:: shell

  cd examples/docker
  bash build_docker.sh

After provisioning a project:

.. code-block:: shell

  nvflare provision -p project.yml

Create a Docker runtime config for ``nvflare deploy prepare``:

.. code-block:: yaml

  runtime: docker

  parent:
    docker_image: nvflare-site:latest
    network: nvflare-network

  job_launcher:
    default_python_path: /usr/local/bin/python
    default_job_env:
      NCCL_P2P_DISABLE: "1"
    default_job_container_kwargs:
      shm_size: 8g
      ipc_mode: host

Prepare each server or client startup kit that should run in Docker:

.. code-block:: shell

  nvflare deploy prepare workspace/<project>/prod_00/server \
    --config docker.yaml \
    --output workspace/<project>/prepared/server

  nvflare deploy prepare workspace/<project>/prod_00/site-1 \
    --config docker.yaml \
    --output workspace/<project>/prepared/site-1

The prepared kits contain ``startup/start_docker.sh``, patched launcher
configuration, and a ``local/study_runtime.yaml`` template. Admin startup kits
are not prepared because they do not run parent server or client processes.

Start prepared parent processes with:

.. code-block:: shell

  cd workspace/<project>/prepared/server
  bash startup/start_docker.sh

Run the same command from each prepared client kit. The generated script creates
the configured Docker network if needed and mounts the prepared kit into the
parent container.

Jobs submitted to Docker-mode sites must specify their job image in
``launcher_spec``:

.. code-block:: json

  {
    "launcher_spec": {
      "default": {
        "docker": {"image": "nvflare-job:latest"}
      },
      "site-1": {
        "docker": {"shm_size": "8g", "ipc_mode": "host"}
      }
    },
    "resource_spec": {
      "site-1": {"num_of_gpus": 1}
    }
  }

Use ``launcher_spec`` for launcher-specific image and container settings. Keep
scheduler-facing resource requests, such as ``num_of_gpus``, in
``resource_spec``. Sites that are not configured with the Docker job launcher
continue to use their configured launcher, usually process mode.


Mounting Local CSV Data into Job Containers
-------------------------------------------

The parent container and job container have different responsibilities. The
parent container runs the long-lived FLARE client or server process, while
``DockerJobLauncher`` launches a separate container for each job.

Local datasets should not normally be copied into either Docker image. Instead,
each site can define site-owned dataset mounts in the prepared startup kit's
``local/study_runtime.yaml`` file. The Docker launcher mounts the configured
host directory directly into the job container.

For example, suppose ``site-1`` stores its local CSV file at:

.. code-block:: text

  /srv/nvflare/site-1-data/train.csv

After running ``nvflare deploy prepare``, edit:

.. code-block:: text

  workspace/<project>/prepared/site-1/local/study_runtime.yaml

Keep a copy of this site-specific configuration outside the prepared output.
Running ``nvflare deploy prepare`` again with the same output path replaces the
entire prepared workspace, including edits to ``study_runtime.yaml``.

Configure the dataset as follows:

.. code-block:: yaml

  format_version: 2
  studies:
    default:
      datasets:
        training:
          source: /srv/nvflare/site-1-data
          mode: ro

For Docker deployments, ``source`` is an absolute path on the Docker host. The
dataset is mounted into the job container at:

.. code-block:: text

  /data/<study-name>/<dataset-name>

Therefore, the CSV file in this example is available to the training code at:

.. code-block:: text

  /data/default/training/train.csv

The training program can read it normally. For example:

.. code-block:: python

  import pandas as pd

  data = pd.read_csv("/data/default/training/train.csv")

Each client should configure its own local source directory in its prepared
startup kit. For example, ``site-2`` may map ``training`` to a different host
directory while exposing the same path inside its job container.

Use ``mode: ro`` for training datasets unless the job must modify the mounted
data. The source path must exist on the Docker host before the job container
launches.

The dataset mount is controlled by the site operator through
``study_runtime.yaml``. Do not place arbitrary Docker ``volumes`` or ``mounts``
settings in the submitted job's ``meta.json`` because these fields are managed
by ``DockerJobLauncher``.

Development Container
---------------------

A single-container image can also be useful for local development or simulator
work. In this case, Docker captures a Python environment with NVFlare and your
development dependencies installed. This is an alternative to a bare-metal
Python virtual environment, not a replacement for ``nvflare deploy prepare``
when using the Docker job launcher.

You can use this single-container pattern to run all FLARE processes on one
host for development: run the simulator, run POC mode, or start provisioned
server/client scripts manually from different shells in the same container.
This is different from Docker runtime deployment. The job launcher remains
process mode unless the server or client startup kit is prepared with
``nvflare deploy prepare``.

After you build an image for your environment and tag it ``nvflare-dev:latest``,
run it with GPU support and a persistent workspace:

.. code-block:: shell

  mkdir my-workspace
  docker run --rm -it --gpus all \
      --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      -v $(pwd -P)/my-workspace:/workspace/my-workspace \
      nvflare-dev:latest

Once the container is running, you can also exec into the container, for example if you need another
terminal to start additional FLARE clients.  First find the ``CONTAINER ID`` using ``docker ps``, and then
use that ID to exec into the container:

.. code-block:: shell

  docker ps  # use the CONTAINER ID in the output
  docker exec -it <CONTAINER ID> /bin/bash

Best Practices
--------------

1. Always use the latest compatible NVIDIA Container Toolkit version
2. Use ``nvflare deploy prepare`` for Docker runtime startup kits
3. Keep parent image and job image responsibilities explicit
4. Put Docker image and container settings in ``launcher_spec``
5. Put scheduler-facing resource requests in ``resource_spec``
6. Mount volumes for persistent data storage
7. Keep your base images updated for security patches

.. note::

   Docker Compose deployment is deprecated. Use ``nvflare deploy prepare`` for
   current Docker runtime preparation.

Common Issues and Solutions
---------------------------

1. Docker daemon access: ensure the user running ``start_docker.sh`` can access
   the Docker daemon.
2. Missing job image: ensure every Docker-mode job provides
   ``launcher_spec[site]["docker"]["image"]`` or
   ``launcher_spec["default"]["docker"]["image"]``.
3. GPU access issues: ensure NVIDIA Container Toolkit is properly installed and
   the job's ``resource_spec`` requests the needed GPUs.
4. Memory or shared-memory issues: set container kwargs such as ``shm_size`` or
   ``ipc_mode`` in ``launcher_spec`` or in deploy prepare
   ``default_job_container_kwargs``.
5. Network connectivity: keep the configured Docker network and server hostname
   resolvable from admin, parent, and job containers.
