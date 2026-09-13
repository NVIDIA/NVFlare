################
Examples Command
################

The ``nvflare examples`` command downloads example source from the public
NVIDIA/NVFlare GitHub repository. It selects the source revision recorded in
the installed NVFlare distribution, so the example matches the version in use.

Example catalog
===============

The installed catalog maps a short CLI name to an example directory anywhere
under ``examples/`` in the NVFlare source tree. Run
``nvflare examples list`` to see the current names and source paths:

.. code-block:: bash

   nvflare examples list
   nvflare examples list --format json

The list, command help, and JSON schema are generated from the catalog, so
adding a short name and source path to ``catalog.json`` makes an example
available without a Python code change.

Get an example
==============

Download an example, then follow its README for dependency installation, data
download, data preparation, model preparation, and other example-specific
steps:

.. code-block:: bash

   python -m pip install "nvflare[PT]"
   nvflare examples get <example-name>
   cd <example-name>

The README may direct you to a root or nested requirements file, then commands
such as ``download_data.py``, ``prepare_data.py``, or ``prepare_model.py``
before the example's job command. Those steps vary by example and are
intentionally not duplicated in the catalog.

For Hello PyTorch, the complete sequence is:

.. code-block:: bash

   python -m pip install "nvflare[PT]"
   nvflare examples get hello-pt
   cd hello-pt
   pip install -r requirements.txt
   python job.py

Use another catalog name in the same command. The completion output identifies
the README:

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

The downloaded directory contains the maintained files from the source
directory selected by the catalog. The command does not create a root
``requirements.txt`` when the source has none. If a root requirements file pins
``nvflare`` to an older version, the downloaded copy removes the version
constraint while preserving extras such as ``nvflare[HE]``. Follow the README
to find the requirements file appropriate for that example. A small
``.nvflare-example.json`` file records the example name, installed NVFlare
version, Git revision, canonical source path, and whether a root NVFlare
requirement was unpinned.

Automation
==========

Use ``--format json`` for structured output:

.. code-block:: bash

   nvflare examples get hello-pt --dest ./automation-copy --format json

Schema discovery does not copy the example:

.. code-block:: bash

   nvflare examples --schema
   nvflare examples list --schema
   nvflare examples get --schema

Failures return a nonzero exit status, an error code, and a recovery hint.
``EXAMPLE_DESTINATION_EXISTS`` identifies an existing destination,
``EXAMPLE_DESTINATION_INVALID`` identifies a missing parent, and
``EXAMPLE_IO_ERROR`` reports filesystem or installation errors.
