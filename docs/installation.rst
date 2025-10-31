.. _installation:

#############
Installation
#############

This guide covers the installation of NVIDIA FLARE and its dependencies.
Before proceeding, make sure you understand the basics of federated learning
from the :ref:`fl_introduction` and have reviewed the :ref:`flare_overview` to understand what you'll be installing.

Prerequisites
=============
- Python 3.9+
- pip
- Git

.. note::
   The server and client versions of nvflare must match, we do not support cross-version compatibility.

Supported Operating Systems
---------------------------
- Linux
- OSX (Note: some optional dependencies are not compatible, such as tenseal and openmined.psi)

Installation Methods
====================

Virtual Environment Setup
-------------------------

It is highly recommended to install NVIDIA FLARE in a virtual environment if you are not using :ref:`containerized_deployment`.
This guide briefly describes how to create a virtual environment with venv.

Virtual Environments and Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Python's official document explains the main idea about virtual environments.
The module used to create and manage virtual environments is called `venv <https://docs.python.org/3.10/library/venv.html>`_.
You can find more information there. We only describe a few necessary steps for a virtual environment for NVIDIA FLARE.

Depending on your OS and the Python distribution, you may need to install the Python's venv package separately. For example, in Ubuntu
20.04, you need to run the following commands to continue creating a virtual environment with venv.

.. code-block:: shell

   $ sudo apt update
   $ sudo apt-get install python3-venv

Once venv is installed, you can use it to create a virtual environment with:

.. code-block:: shell

    $ python3 -m venv nvflare-env

This will create the ``nvflare-env`` directory in current working directory if it doesn't exist,
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

Stable releases are available on `NVIDIA FLARE PyPI <https://pypi.org/project/nvflare>`_:

.. code-block:: shell

  $ python3 -m pip install nvflare

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

NVFlare provides several optional dependency groups that you can install based on your needs:

* **HE** - Homomorphic Encryption support:

  .. code-block:: shell

     $ pip install nvflare[HE]

* **PSI** - Private Set Intersection support:

  .. code-block:: shell

     $ pip install nvflare[PSI]

* **PT** - PyTorch support:

  .. code-block:: shell

     $ pip install nvflare[PT]

* **SKLEARN** - Scikit-learn support:

  .. code-block:: shell

     $ pip install nvflare[SKLEARN]

* **TRACKING** - MLflow, Weights & Biases, and TensorBoard support:

  .. code-block:: shell

     $ pip install nvflare[TRACKING]

* **MONITORING** - Datadog monitoring support:

  .. code-block:: shell

     $ pip install nvflare[MONITORING]

* **CONFIG** - OmegaConf configuration support:

  .. code-block:: shell

     $ pip install nvflare[CONFIG]

You can also install multiple optional dependencies at once:

.. code-block:: shell

  $ pip install nvflare[PT,SKLEARN,TRACKING]  # Install PyTorch, Scikit-learn, and tracking support

For development, you can install all dependencies (except HE and PSI on macOS):

.. code-block:: shell

  # On Linux
  $ pip install nvflare[dev]

  # On macOS
  $ pip install nvflare[dev_mac]

Install from Source
-------------------

Clone NVFlare repo and install from source (useful for accessing latest nightly features or testing custom builds):

.. code-block:: shell

  $ git clone https://github.com/NVIDIA/NVFlare.git
  $ cd NVFlare
  $ pip install -e .  # Install in editable mode

You can also install with optional dependencies from source:

.. code-block:: shell

  $ pip install -e ".[dev]"  # Install all development dependencies
  $ pip install -e ".[PT,SKLEARN]"  # Install specific optional dependencies

Note on branches:

* The `main <https://github.com/NVIDIA/NVFlare/tree/main>`_ branch is the default (unstable) development branch
* The 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7 etc. branches are the branches for each major release and there are tags based on these with a third digit for minor patches

To switch to a specific branch:

.. code-block:: shell

  $ git switch 2.7  # Replace with desired version

Building Wheels
---------------

You can build wheel packages for NVFlare using the following steps:

1. Install build dependencies:

.. code-block:: shell

  $ pip install build wheel

2. Build the wheel:

.. code-block:: shell

  $ python -m build

This will create wheel files in the `dist/` directory. The wheel files can be installed using pip:

.. code-block:: shell

  $ pip install dist/nvflare-*.whl

.. note::
   Building wheels requires all build dependencies to be installed. If you encounter any issues,
   make sure you have the latest version of pip, setuptools, and wheel installed.

Building for Specific Platforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build wheels for specific platforms or Python versions, you can use the following environment variables:

.. code-block:: shell

  # For a specific Python version
  $ PYTHON=python3.9 python -m build

  # For a specific platform
  $ PLATFORM=linux_x86_64 python -m build

.. note::
   The platform-specific builds are useful when you need to distribute wheels to systems
   with different architectures or Python versions.

Next Steps
==========
After completing the installation:

1. Follow the :ref:`quickstart` guide to run your first federated learning example
2. Learn more about different ways to use NVFlare in the :ref:`getting_started` guide
3. Explore more examples in the :ref:`example_applications` section
4. When ready for production, see :ref:`deployment_overview` for deployment guidance
