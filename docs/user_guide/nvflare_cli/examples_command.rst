.. _examples_command:

Download an example
===================

.. versionadded:: 2.10

``nvflare examples get`` downloads a maintained example into a new local
directory. It selects the example matching the installed NVFlare version and
prints the source revision and commands to run next. The initial catalog
contains ``hello-pt``, a CPU-compatible PyTorch federation with two clients,
three rounds, and synthetic data.

.. code-block:: bash

   python -m pip install "nvflare[PT]"
   nvflare examples get hello-pt
   cd hello-pt
   python job.py

The download includes the example's Python source, README, requirements, and
supporting files. GitHub HTTPS access is required; Git is not. Retrieval does
not run the example or install dependencies. The example README owns data,
training, customization, and continuation instructions.

Choose a destination
--------------------

The default destination is ``hello-pt`` in the current directory. Choose a
different directory with ``--dest``:

.. code-block:: bash

   nvflare examples get hello-pt --dest ./my-training

The destination's parent must already exist. An existing destination is never
merged or overwritten, including an empty directory or a symbolic link.
Explicit ``--dest`` paths are validated before contacting GitHub; the destination
and download cache must not contain one another. Containment checks recognize
symbolic links and alternate-case spellings on case-insensitive filesystems.
Interrupted downloads are cleaned up before they can become the requested
destination. Atomic delivery uses Linux ``renameat2(RENAME_NOREPLACE)`` or macOS
``renamex_np(RENAME_EXCL)`` and requires filesystem support for that operation.
On 64-bit x86 and ARM Linux, a direct system call is used when the C library
lacks the ``renameat2`` wrapper. Unsupported systems fail without replacing
the destination.

Version and source selection
----------------------------

A release selects its exact tag: ``2.10.0`` selects ``2.10.0``, and
``2.10.0rc3`` selects ``2.10.0rc3``. A development build selects the full source
commit recorded by Versioneer. References are resolved to an immutable commit
before reading the catalog or downloading files. Missing tags and catalogs
produce errors rather than falling forward to another release or ``main``.

For development, explicitly select a tag, branch, or commit:

.. code-block:: bash

   nvflare examples get hello-pt --ref main

The installed package must satisfy the selected catalog entry's version
constraint, including with an override. A development build without source
provenance, or an editable checkout with uncommitted changes, requires an
explicit reference. Local modifications are not uploaded or copied.

Examples come from the public ``NVIDIA/NVFlare`` catalog at the selected
revision. Branch names containing slashes can be passed directly to ``--ref``.
Other repositories, private authentication, and Git LFS payloads are not
supported by this command.

The generated ``.nvflare-example.json`` file records the repository, example,
requested reference, resolved commit, canonical source URL, and installed
NVFlare version. Include it when reporting an example problem.

Cache behavior
--------------

Validated downloads are cached by repository, resolved commit, and requested
example name. A repeat request for the same key reports
``cache hit`` and copies locally. Reference resolution still contacts GitHub;
cache hits do not download the catalog or example content again. Cached file
contents are checked against their Git blob hashes before each delivery;
matching file sizes and timestamps alone do not establish integrity. Edits to
a delivered directory do not affect the cached source.

.. code-block:: bash

   nvflare examples get hello-pt --dest ./another-run
   nvflare examples get hello-pt --dest ./fresh-download --refresh
   nvflare examples cache clear

The cache lives under ``$XDG_CACHE_HOME/nvflare/examples`` (default
``~/.cache/nvflare/examples``) on Linux, ``~/Library/Caches/nvflare/examples``
on macOS. At most eight entries are retained, keeping the most recently used.
A process lock serializes cache writes, delivery, and clearing; a busy cache
fails after 60 seconds.
Corrupt entries are removed and downloaded again. Cache permission and other
I/O errors are reported without treating the entry as corrupt. Version
compatibility errors retain their version-specific recovery hints during repair.
Cache clearing affects only
cached downloads, preserving all delivered workspaces.

Set ``NVFLARE_EXAMPLES_CACHE_DIR`` to choose a different cache directory,
for example in an isolated build or validation environment.

Downloads are bounded to 256 tree entries, 8 MiB per file, and 64 MiB total
file content; metadata responses are capped at 2 MiB. Each file response is also
limited by its declared Git object size. Paths, object types, and
Git blob hashes are checked. Symbolic links, submodules, path traversal,
Unicode-equivalent or case-colliding paths, and incomplete GitHub tree responses
are rejected.

Errors and automation
---------------------

Failures return a nonzero exit status, a named error code, and a recovery hint.
For example, ``EXAMPLE_DESTINATION_EXISTS`` suggests another destination,
``EXAMPLE_VERSION_INCOMPATIBLE`` identifies the required version, and
``EXAMPLE_ACCESS_LIMITED`` suggests retrying after GitHub's public API limit
resets. Interrupted retrieval exits 130. The command uses unauthenticated
public requests and does not read credentials from ``.netrc``.

``examples get`` has a five-minute deadline for downloading, validating, and
staging the example, in addition to the HTTP connect and idle-read timeouts.
Even a response that keeps sending small amounts
of data is interrupted at the deadline with ``EXAMPLE_TIMEOUT``. Staging files
are cleaned up and the cache lock is released. The deadline is checked once more
before final publication. Once publication starts, it is allowed to finish so a
successfully created destination is reported as success.

.. code-block:: bash

   nvflare examples get hello-pt --format json
   nvflare examples get --schema
   nvflare examples cache clear --schema

JSON output uses the standard CLI envelope with source provenance, destination,
cache status, required dependency extra, and the next command. Schema discovery
does not contact GitHub or create a cache.

For a full repository checkout, follow the
`repository instructions <https://github.com/NVIDIA/NVFlare#readme>`_.
This remains useful for contributors and examples outside the download catalog.

Maintaining the catalog
-----------------------

``examples/catalog.json`` belongs to the same source revision as the examples.
Schema version 1 maps stable short names to a repository directory, default
destination, NVFlare version constraint, required extra, and
``["python", "job.py"]`` next command. Only independently runnable directories
with ``job.py`` and ``README.md`` should be added. The CLI and catalog validator
are shipped in the distribution; example files and the release-owned catalog
remain in the repository.

Normal unit tests use HTTP fixtures. The ``Example download validation``
workflow builds and installs a distribution in a fresh environment, downloads
the matching official example, and runs its zero-argument job. It checks
provenance, cache reuse, three rounds, two-site final evaluation, learning
thresholds, and global-model loadability. Run it for release candidates as well
as the scheduled development check.
