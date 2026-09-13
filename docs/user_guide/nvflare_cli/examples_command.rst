##########################
Bundled Examples Command
##########################

The ``nvflare examples`` command copies runnable example source bundled with
the installed NVFlare distribution. It works offline and the copied source
always matches the installed NVFlare version.

Get Hello PyTorch
=================

Install NVFlare with its PyTorch dependencies, copy the example, and run it:

.. code-block:: bash

   python -m pip install "nvflare[PT]"
   nvflare examples get hello-pt
   cd hello-pt
   python job.py

The default destination is ``hello-pt`` in the current directory. Use
``--dest`` to choose another new directory:

.. code-block:: bash

   nvflare examples get hello-pt --dest ./my-hello-pt

The destination's parent must already exist. The command never merges with or
overwrites an existing file, directory, or symbolic link. It creates the
destination exclusively and removes it if copying fails or is interrupted.
If the operating system or process stops abruptly, remove any incomplete
destination before retrying.

The copied directory contains the same maintained files as
``examples/hello-world/hello-pt`` in the corresponding NVFlare release. A small
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
