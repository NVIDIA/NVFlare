.. _quickstart:

##########
Quickstart
##########

This section provides a starting point for new users to start NVIDIA FLARE.
Users can go through the :ref:`example_apps` and get familiar with how NVIDIA FLARE is designed,
operates and works.

Each example introduces concepts about NVIDIA FLARE, while showcasing how some popular libraries and frameworks can
easily be integrated into the FL process.

.. _installation:

Installation
=============

Python Version
--------------

NVIDIA FLARE requires Python 3.8.  It may work with Python 3.7 but currently is not compatible with Python 3.9 and above.

Install NVIDIA FLARE in virtual environments
--------------------------------------------

It is highly recommended to install NVIDIA FLARE in a virtual environment.
This guide briefly describes how to create a virtual environment with venv.

Virtual Environments and Packages
.................................

Python's official document explains the main idea about virtual environments.
The module used to create and manage virtual environments is called `venv <https://docs.python.org/3.8/library/venv.html#module-venv>`_.
You can find more information there.  We only describe a few necessary steps for a virtual environment for NVIDIA FLARE.


Depending on your OS and the Python distribution, you may need to install the Python's venv package separately.  For example, in ubuntu
20.04, you need to run the following commands to continue creating a virtual environment with venv.

.. code-block:: shell

   $ sudo apt update
   $ sudo apt-get install python3-venv


Once venv is installed, you can use it to create a virtual environment with:

.. code-block:: shell

    $ python3 -m venv nvflare-env

This will create the ``nvflare-env`` directory in current working directory if it doesn’t exist,
and also create directories inside it containing a copy of the Python interpreter,
the standard library, and various supporting files.


Activate the virtualenv by running the following command:

.. code-block:: shell

    $ source nvflare-env/bin/activate


You may find that the pip and setuptools versions in the venv need updating:

.. code-block:: shell

  (nvflare-env) $ python3 -m pip install -U pip
  (nvflare-env) $ python3 -m pip install -U setuptools


Install Stable Release
----------------------

Stable releases are available on `NVIDIA FLARE PyPI <https://pypi.org/project/nvflare>`_::

  $ python3 -m pip install nvflare


Clone Repository and Examples
-----------------------------

The next sections in the :ref:`quickstart` will guide you through the examples included in the repository. To clone the
repo and get the source code:

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git

.. _setting_up_poc:

Setting Up the Application Environment in POC Mode
==================================================

.. warning::

    POC mode is not intended to be secure and should not be run in any type of production environment or any environment
    where the server's ports are exposed. For actual deployment and even development, it is recommended to use a
    :ref:`secure provisioned setup <provisioned_setup>`.

To get started with a proof of concept (POC) setup after :ref:`installation`, run this command to generates a poc folder
with a server, two clients, and one admin:

.. code-block:: shell

    $ poc -n 2

Copy necessary files (the exercise code in the examples directory of the NVFlare repository) to a working folder (upload
folder for the admin):

.. code-block:: shell

  $ mkdir -p poc/admin/transfer
  $ cp -rf NVFlare/examples/* poc/admin/transfer

.. _starting_poc:

Starting the Application Environment in POC Mode
================================================

Once you are ready to start the FL system, you can run the following commands to start all the different parties (it is
recommended that you read into the specific :ref:`example apps <example_apps>` first, then start the FL
system to follow along at the parts with admin commands for you to run the example app).

FL systems usually have an overseer, server, and multiple clients. We therefore have to start the overseer first:

.. code-block:: shell

    $ ./poc/overseer/startup/start.sh

Once the overseer is running, you can start the server and clients in different terminals (make sure your terminals are
using the environment with NVIDIA FLARE :ref:`installed <installation>`).

Open a new terminal and start the server:

.. code-block:: shell

    $ ./poc/server/startup/start.sh

Once the server is running, open a new terminal and start the first client:

.. code-block:: shell

    $ ./poc/site-1/startup/start.sh

Open another terminal and start the second client:

.. code-block:: shell

    $ ./poc/site-2/startup/start.sh

In one last terminal, start the admin:

.. code-block:: shell

  $ ./poc/admin/startup/fl_admin.sh localhost

This will launch a command prompt where you can input admin commands to control and monitor many aspects of
the FL process.

.. tip::

   For anything more than the most basic proof of concept examples, it is recommended that you use a
   :ref:`secure provisioned setup <provisioned_setup>`.
