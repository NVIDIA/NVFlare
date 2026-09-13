##########################
Bundled Examples Command
##########################

The ``nvflare examples`` command copies runnable example source bundled with
the installed NVFlare distribution. Editable installations copy from the same
source checkout. Retrieval works offline and the copied source matches the
NVFlare version in use.

Bundled example catalog
=======================

The catalog maps a short CLI name to an example directory anywhere in the
NVFlare source tree. Run ``nvflare examples get --help`` to see the current
names. The help and JSON schema are generated from the bundled catalog, so a
new catalog entry becomes available without a Python code change.

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

The copied directory contains the same maintained files as the source directory
selected by the catalog. A small ``.nvflare-example.json`` file records the
example name, installed NVFlare version, and canonical source path.

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
