.. _notebook_testing:

#################
Notebook Testing
#################

NVIDIA FLARE uses `nbmake <https://github.com/treebeardtech/nbmake>`__ to test Jupyter notebooks.
This ensures that example notebooks remain functional as the codebase evolves.

.. note::

   **Not all notebooks are ready for automated testing yet.** Some notebooks require external 
   infrastructure (running FLARE servers, provisioned environments, specific datasets) or contain 
   interactive elements that cannot run in CI. Notebook test coverage is being improved over time.

For general ``runtest.sh`` usage (dependency caching, verbose mode, etc.), see :ref:`developer_testing`.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Start
===========

Use the ``runtest.sh`` script to run notebook tests:

.. code:: bash

   # Test default notebook (flare_simulator.ipynb)
   ./runtest.sh -n

   # Test a specific notebook
   ./runtest.sh -n examples/tutorials/flare_simulator.ipynb

   # Test with verbose output
   ./runtest.sh -n -v examples/tutorials/flare_simulator.ipynb

Notebook-Specific Options
=========================

These options are specific to notebook testing (``-n``):

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--timeout=SECONDS``
     - 1200
     - Timeout in seconds for each notebook execution
   * - ``--nb-clean=MODE``
     - on-success
     - When to clean outputs: ``always``, ``on-success``, ``never``
   * - ``--kernel=NAME``
     - python3
     - Jupyter kernel name (defaults to ``python3`` if available)

Examples
--------

.. code:: bash

   # Set a shorter timeout (5 minutes)
   ./runtest.sh -n --timeout=300 examples/tutorials/flare_simulator.ipynb

   # Use a specific kernel
   ./runtest.sh -n --kernel=python3 examples/tutorials/flare_simulator.ipynb

   # Always clean outputs regardless of pass/fail
   ./runtest.sh -n --nb-clean=always examples/tutorials/

   # Combine multiple options with verbose output
   ./runtest.sh -n -v --timeout=1800 --kernel=python3 examples/tutorials/

Direct pytest Usage
===================

You can also run nbmake directly with pytest:

.. code:: bash

   pytest --nbmake --nbmake-timeout=1200 --nbmake-clean=on-success examples/tutorials/

   # With specific kernel
   pytest --nbmake --nbmake-timeout=1200 --kernel=python3 examples/tutorials/

Skipping Cells in Notebooks
===========================

To skip specific cells during automated testing (e.g., Colab setup cells, interactive 
visualizations, or cells that require user input), add one of these tags to the cell metadata:

- ``skip-execution``
- ``skip``
- ``colab``

Adding Tags in Jupyter
----------------------

**In Jupyter Lab:**

1. Select the cell you want to skip
2. Click the gear icon in the right sidebar (or View → Right Sidebar → Show Property Inspector)
3. Under "Common Tools" → "Cell Tags", add: ``skip-execution``

**In Jupyter Notebook (classic):**

1. Select the cell
2. View → Cell Toolbar → Tags
3. Add tag: ``skip-execution``

**In VS Code:**

1. Click on the cell
2. Click "..." menu on the cell
3. Select "Add Cell Tag"
4. Enter: ``skip-execution``

How It Works
============

The testing framework (implemented in ``conftest.py``) automatically:

1. **Before test**: Creates a ``.backup`` of the original notebook
2. **Filters cells**: Removes cells tagged with ``skip-execution``, ``skip``, or ``colab``
3. **Updates kernel**: Adjusts kernel spec to match the specified or detected kernel
4. **Executes**: nbmake runs the notebook through the Jupyter kernel
5. **Restores**: Original notebook is restored from backup
6. **Cleans outputs**: Cell outputs are cleared based on ``--nbmake-clean`` setting

This ensures:

- Original notebooks in git remain unchanged
- Notebooks can contain Colab-specific or interactive cells that won't break CI
- Consistent kernel usage across different development environments

Troubleshooting
===============

Kernel Not Found
----------------

If you see "Kernel not found" errors:

1. Ensure your virtual environment is activated
2. Install ipykernel: ``pip install ipykernel``
3. Register your kernel: ``python -m ipykernel install --user --name=my_env``
4. Or specify an existing kernel: ``./runtest.sh -n --kernel=python3``

Timeout Errors
--------------

For long-running notebooks, increase the timeout:

.. code:: bash

   ./runtest.sh -n --timeout=3600 examples/advanced/

Notebook Requires External Infrastructure
-----------------------------------------

Some notebooks (e.g., ``flare_api.ipynb``) require a running FLARE server or provisioned 
environment. These notebooks will fail in automated testing unless the infrastructure is set up.

For such notebooks, consider:

1. Running them manually in an interactive environment
2. Adding ``skip-execution`` tags to cells that require external services
3. Creating a simplified version for automated testing

Best Practices
==============

1. **Tag cells appropriately**: Mark Colab setup, interactive widgets, and user-input cells with ``skip-execution``

2. **Keep notebooks focused**: Smaller notebooks are faster to test and easier to debug

3. **Use reasonable timeouts**: Increase ``--timeout`` based on expected execution time plus buffer

4. **Test locally before pushing**: Run ``./runtest.sh -n`` on your notebooks before committing

5. **Clean outputs before committing**: Use ``--nb-clean=always`` or manually clear outputs to keep git diffs clean

6. **Use self-contained examples**: Notebooks that use the simulator (like ``flare_simulator.ipynb``) are easier to test than those requiring external servers

