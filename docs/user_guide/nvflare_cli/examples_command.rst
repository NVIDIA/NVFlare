################
Examples Command
################

The ``nvflare examples`` command downloads example source from the public
NVIDIA/NVFlare GitHub repository. It selects the source revision recorded in
the installed NVFlare distribution, so the example matches the version in use.
A nightly package built from ``main`` retrieves examples from that exact build
commit. A release package such as NVFlare 2.9.0 retrieves examples from the
commit recorded in the 2.9.0 package.

Example catalog
===============

The installed catalog maps a short CLI name to an example directory anywhere
under ``examples/`` in the NVFlare source tree. Run
``nvflare examples list`` to see the current names and source paths:

.. code-block:: bash

   nvflare examples list
   nvflare examples list --format json

The list and ``get`` lookup use the catalog. Command help and JSON schema use a
generic ``NAME`` argument instead of enumerating the catalog. Adding a short
name and source path to ``catalog.json`` makes an example available without a
Python code change.

Get an example
==============

Install the optional dependency group required by the example on the same
NVFlare distribution already in use. For example, use one of these forms for
PyTorch support:

.. code-block:: bash

   # Stable installation
   python -m pip install "nvflare[PT]"

   # Nightly installation
   python -m pip install "nvflare-nightly[PT]"

   # Editable source installation, run from the NVFlare checkout
   python -m pip install -e ".[PT]"

Replace ``PT`` with another available group such as ``HE``, ``SKLEARN``, or
``TRACKING`` when the example requires it. Download the example, then follow
its README for remaining dependencies, data download, preparation, and run
steps:

.. code-block:: bash

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
``requirements.txt`` when the source has none, and it does not rewrite root or
nested requirements files. Follow the README to find the requirements file
appropriate for that revision. A small ``.nvflare-example.json`` file records
the example name, installed NVFlare version, Git revision, and canonical source
path.

If any root or nested ``requirements.txt`` names ``nvflare`` or
``nvflare-nightly``, or a ``pyproject.toml`` declares either distribution, the
command reports each affected path in human and JSON output. Keep the NVFlare
distribution already installed, add required extras to that same distribution
as shown above, and install the remaining example dependencies without
reinstalling NVFlare.

If the selected directory has no root README, retrieval still succeeds and
reports a warning to inspect the downloaded files for instructions.

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
