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
