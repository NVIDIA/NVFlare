.. _containerized_deployment:

########################
Containerized Deployment
########################

Containerized Deployment with Docker
====================================

Prerequisites
-------------
Before starting with containerized deployment, ensure you have:

1. Docker installed on your system
2. NVIDIA Container Toolkit installed for GPU support
3. System requirements met as per the `NVIDIA Container Toolkit Install Guide <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_

Running NVIDIA FLARE in a Docker container provides several benefits:
    - Consistent environment across different systems
    - Easy dependency management
    - Simplified deployment process
    - Isolated execution environment
    - GPU support through NVIDIA Container Toolkit

This can be used as an alternative to the bare-metal Python virtual environment and will
use a similar installation to simplify transitioning between a bare metal and containerized
environment.

Building the Docker Image
-------------------------

A simple Dockerfile is used to capture the base requirements and dependencies.  In
this case, we're building an environment that will support PyTorch-based workflows,
in particular the :github_nvflare_link:`Hello PyTorch <examples/hello-world/hello-pt>`
example. The base for this build is the NGC PyTorch container.  On this base image,
we will install the necessary dependencies and clone the NVIDIA FLARE GitHub
source code into the root workspace directory.

Let's first create a folder called ``build`` and then create a file inside named ``Dockerfile``:

.. code-block:: shell

  mkdir build
  cd build
  touch Dockerfile

Using any text editor to edit the Dockerfile and paste the following:

.. literalinclude:: ../../../resources/Dockerfile
    :language: dockerfile

.. note::
    Feel free to substitute the base image with the latest version of the NGC PyTorch container.
    See the `NGC PyTorch Container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`_
    for the latest versions.

We can then build the new container by running docker build in the directory containing
this Dockerfile, for example tagging it nvflare-pt:

.. code-block:: shell

  docker build -t nvflare-pt . -f Dockerfile

This will result in a docker image, ``nvflare-pt:latest``.

Running the Container
---------------------

To run the container with GPU support and a persistent workspace:

.. code-block:: shell

  mkdir my-workspace
  docker run --rm -it --gpus all \
      --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      -v $(pwd -P)/my-workspace:/workspace/my-workspace \
      nvflare-pt:latest

Once the container is running, you can also exec into the container, for example if you need another
terminal to start additional FLARE clients.  First find the ``CONTAINER ID`` using ``docker ps``, and then
use that ID to exec into the container:

.. code-block:: shell

  docker ps  # use the CONTAINER ID in the output
  docker exec -it <CONTAINER ID> /bin/bash

Deployment Scenarios
--------------------

The containerized environment can be used in several ways:

1. FL Simulator
   - Run the FL simulator inside the container
   - Mount your application code directory
   - All dependencies are pre-installed
   - Useful for development and testing

2. POC Mode
   - Run multiple containers for server and clients
   - Use Docker networking for communication
   - Good for testing multi-party scenarios

3. Production Mode
   - Deploy containers across different machines
   - Use proper networking and security configurations
   - Suitable for real-world deployments

When using the FL Simulator, you can simply mount in any directories needed for
your FLARE application code, and run the Simulator within the Docker container with
all dependencies installed.

For a notebook showcasing this example, see the :github_nvflare_link:`NVIDIA FLARE with Docker example <examples/advanced/docker>`.

Best Practices
--------------

1. Always use the latest compatible NVIDIA Container Toolkit version
2. Mount volumes for persistent data storage
3. Use appropriate resource limits (CPU, memory, GPU) based on your workload
4. Consider using Docker Compose for multi-container deployments
5. Keep your base images updated for security patches

Common Issues and Solutions
---------------------------

1. GPU access issues: Ensure NVIDIA Container Toolkit is properly installed
2. Memory issues: Adjust ulimit settings if needed
3. Network connectivity: Configure proper network settings for multi-container deployments
