################
Examples Command
################

The ``nvflare examples`` command downloads example source from the public
NVIDIA/NVFlare GitHub repository. It selects the source revision recorded in
the installed NVFlare distribution, so the example matches the version in use.
A nightly package built from ``main`` retrieves examples from that exact build
commit. A release package such as NVFlare 2.10.0 retrieves examples from the
commit recorded in the 2.10.0 package.

Example catalog
===============

The installed catalog assigns a category and maps a short CLI name to an
example directory under ``examples/`` in the NVFlare source tree. Examples
whose code requires files elsewhere in a full source checkout are not listed
until their catalog path includes those shared sources.
Run ``nvflare examples list`` to browse names and source paths grouped by
category:

.. code-block:: bash

   nvflare examples list
   nvflare examples list --format json

The list and ``get`` lookup use the catalog. Command help and JSON schema use a
generic ``NAME`` argument instead of enumerating the catalog. Adding a
category, short name, and source path to ``catalog.json`` makes an example
available without a Python code change. An entry can also name other catalog
examples that it requires. The command retrieves those dependencies from the
same source revision.

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

Replace or combine optional groups according to the example. For example,
experiment tracking with PyTorch requires both groups:

.. code-block:: bash

   # Stable installation
   python -m pip install "nvflare[PT,TRACKING]"

   # Nightly installation
   python -m pip install "nvflare-nightly[PT,TRACKING]"

   # Editable source installation, run from the NVFlare checkout
   python -m pip install -e ".[PT,TRACKING]"

Other examples may use groups such as ``HE`` or ``SKLEARN``. Download the
example, then follow its README for remaining dependencies, data download,
preparation, and run steps:

.. code-block:: bash

   nvflare examples get <example-name>
   cd <example-name>

The README may direct you to a root or nested requirements file, then commands
such as ``download_data.py``, ``prepare_data.py``, or ``prepare_model.py``
before the example's job command. Those steps vary by example and are
intentionally not duplicated in the catalog.

Agent Skills examples download the example inputs and prompt without copying
the repository's shared top-level ``skills/`` directory. Install the skills
directly from the same Git revision recorded in ``.nvflare-example.json``
before opening the example in Codex or Claude Code:

.. code-block:: bash

   NVFLARE_REVISION=$(nvflare examples revision)
   npx skills add "https://github.com/NVIDIA/NVFlare/tree/${NVFLARE_REVISION}/skills" \
     --skill '*' -a codex -a claude-code -y

``nvflare examples revision`` reads the durable revision recorded in the
downloaded directory's ``.nvflare-example.json`` file, so the selected skills
remain aligned if the directory is moved or the active NVFlare installation is
later upgraded. Use ``--dir <downloaded-example>`` when running it outside that
directory.

If you already have NVFlare cloned at the same revision, install the skills
from its top-level ``skills/`` directory instead of downloading them from
GitHub:

.. code-block:: bash

   npx skills add "<nvflare-repo>/skills" --skill '*' -a codex -a claude-code -y

For Hello PyTorch, first install the ``PT`` extra on the NVFlare distribution
already in use with the matching command above. Then run:

.. code-block:: bash

   nvflare examples get hello-pt
   cd hello-pt
   pip install -r requirements.txt
   python job.py

The advanced Hello PyTorch environments example reuses the beginner example.
Download both at their maintained relative locations with one command:

.. code-block:: bash

   nvflare examples get hello-pt-environments
   cd hello-pt-environments/advanced/hello-pt-environments
   python job.py --env poc

The download root contains ``advanced/hello-pt-environments`` and its
``hello-world/hello-pt`` dependency. Each directory has its own provenance
file. A retry reuses a dependency when its repository, revision, example name,
and source path match; a conflicting directory is never overwritten.

Use another catalog name in the same command. The completion output identifies
the README:

.. code-block:: bash

   nvflare examples get hello-jax
   nvflare examples get hello-numpy --dest ./numpy-demo

The default destination is the selected short name in the current directory.
Use ``--dest`` to choose another new directory:

.. code-block:: bash

   nvflare examples get hello-pt --dest ./my-hello-pt

The destination's parent must already exist. For an example without catalog
dependencies, the command never merges with or overwrites an existing file,
directory, or symbolic link and creates the destination exclusively. For an
example with dependencies, the destination is a common download root. Existing
component directories are reused only when their provenance matches exactly;
other existing component paths are rejected. If downloading does not complete,
the error identifies the incomplete destination. Remove that incomplete
component before retrying.

Most examples place their files directly under the destination. When an example
depends on its maintained Python package, script hierarchy, or another catalog
example, the command keeps the required relative layout inside the destination
and reports the requested example's nested README to follow.

The downloaded directory contains the maintained files from the source
directory selected by the catalog. The command does not create a root
``requirements.txt`` when the source has none, and it does not rewrite root or
nested requirements files. Follow the README to find the requirements file
appropriate for that revision. A small ``.nvflare-example.json`` file records
the example name, installed NVFlare version, Git revision, and canonical source
path.

Every successful download includes the same dependency notice in human and JSON
output: keep the NVFlare distribution already installed, add required extras to
that same stable, nightly, or editable distribution as shown above, and skip
any README or dependency-file instruction that installs ``nvflare`` or
``nvflare-nightly``. Install only the remaining example dependencies. The
command does not try to interpret each example's dependency files.

If the selected directory has no root README, retrieval still succeeds and
reports a warning to inspect the downloaded files for instructions.

Automation
==========

Use ``--format json`` for structured output:

.. code-block:: bash

   nvflare examples get hello-pt --dest ./automation-copy --format json

Schema discovery does not download the example:

.. code-block:: bash

   nvflare examples --schema
   nvflare examples list --schema
   nvflare examples get --schema

``nvflare examples list`` and the ``--schema`` commands use the installed
catalog and do not contact GitHub. Each ``nvflare examples get`` invocation
makes one GitHub REST API request to locate the selected subtree, then
downloads its files from ``raw.githubusercontent.com``. The request is
anonymous by default. When ``GITHUB_TOKEN`` or ``GH_TOKEN`` is set, the command
uses that token for the API request; ``GITHUB_TOKEN`` takes precedence when
both are set.

GitHub currently limits unauthenticated REST API traffic to 60 requests per
hour per originating IP address. This allowance can be shared by machines
behind the same proxy or NAT gateway. For repeated CI or agent workflows,
download an example once and reuse that workspace instead of calling ``get``
in a loop. For bulk retrieval, use a revision-pinned Git checkout. If GitHub
returns a ``403`` or ``429`` rate-limit response, wait for the limit window to
reset before retrying. You can check the current allowance and reset time with
``curl https://api.github.com/rate_limit``. See `GitHub REST API rate limits
<https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api>`_.

Failures return a nonzero exit status, an error code, and a recovery hint.
``EXAMPLE_DESTINATION_EXISTS`` identifies an existing destination,
``EXAMPLE_DESTINATION_INVALID`` identifies a missing parent, and
``EXAMPLE_IO_ERROR`` reports filesystem or installation errors.
