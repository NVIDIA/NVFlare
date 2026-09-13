##########################
Bundled Examples Command
##########################

The ``nvflare examples`` command copies runnable example source bundled with
the installed NVFlare distribution. Editable installations copy from the same
source checkout. Retrieval works offline and the copied source matches the
NVFlare version in use.

Bundled example catalog
=======================

The catalog covers the primary framework examples from ``examples/hello-world``:

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Short name
     - Framework
     - Preparation and run
   * - ``hello-pt``
     - PyTorch
     - Install ``requirements.txt``; run ``python job.py``
   * - ``hello-numpy``
     - NumPy
     - Install ``requirements.txt``; run ``python job.py``
   * - ``hello-tf``
     - TensorFlow
     - Install ``requirements.txt``; run with ``TF_FORCE_GPU_ALLOW_GROWTH=true``
   * - ``hello-jax``
     - JAX, Flax, and Optax
     - Install requirements, then run ``prepare_model.py``, ``prepare_data.py``, and ``job.py``
   * - ``hello-lightning``
     - PyTorch Lightning
     - Install requirements; run ``python job.py --synthetic_data``
   * - ``hello-huggingface``
     - Hugging Face and TRL
     - Install requirements, then run ``prepare_data.py`` and ``job.py``
   * - ``hello-flower``
     - Flower with PyTorch
     - Install requirements; run the ``flwr-pt`` configuration shown by the command

Get an example
==============

Install NVFlare with its PyTorch dependencies, copy the example, and run it:

.. code-block:: bash

   python -m pip install "nvflare[PT]"
   nvflare examples get hello-pt
   cd hello-pt
   pip install -r requirements.txt
   python job.py

Use another catalog name in the same command. The completion output prints its
dependency, preparation, and run commands:

.. code-block:: bash

   nvflare examples get hello-jax
   nvflare examples get hello-numpy --dest ./numpy-demo

The default destination is the selected short name in the current directory.
Use ``--dest`` to choose another new directory:

.. code-block:: bash

   nvflare examples get hello-pt --dest ./my-hello-pt

The destination's parent must already exist. The command never merges with or
overwrites an existing file, directory, or symbolic link. It creates the
destination exclusively. If copying does not complete, remove the incomplete
destination before retrying.

The copied directory contains the same maintained files as the corresponding
``examples/hello-world/<name>`` directory in the NVFlare release. A small
``.nvflare-example.json`` file records the example name, installed NVFlare
version, and canonical source path.

Automation
==========

Use ``--format json`` for structured output:

.. code-block:: bash

   nvflare examples get hello-pt --dest ./automation-copy --format json

Schema discovery does not copy the example:

.. code-block:: bash

   nvflare examples --schema
   nvflare examples get --schema

Failures return a nonzero exit status, an error code, and a recovery hint.
``EXAMPLE_DESTINATION_EXISTS`` identifies an existing destination,
``EXAMPLE_DESTINATION_INVALID`` identifies a missing parent, and
``EXAMPLE_IO_ERROR`` reports filesystem or installation errors.
