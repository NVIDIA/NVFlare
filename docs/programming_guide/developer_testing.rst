.. _developer_testing:

##################
Developer Testing
##################

This guide covers the ``runtest.sh`` script used for running tests during NVIDIA FLARE development.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Start
===========

Run the full test suite:

.. code:: bash

   ./runtest.sh

**Default behavior** (no arguments): Runs the complete CI-equivalent test suite:

1. License header check
2. Code style check (black, isort, flake8)
3. Auto-fix formatting
4. Unit tests with coverage reporting

Individual Commands
-------------------

Run specific tests individually:

.. code:: bash

   ./runtest.sh -u    # Unit tests only (faster, no style checks)
   ./runtest.sh -s    # Check code formatting only
   ./runtest.sh -f    # Fix code formatting only
   ./runtest.sh -n    # Notebook tests
   ./runtest.sh -l    # License header check only

.. note::

   ``./runtest.sh`` runs the **full suite** (license + style + unit tests).
   Use ``./runtest.sh -u`` for **faster iteration** when you only need unit tests.

Available Commands
==================

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``-u`` / ``--unit-tests``
     - Run unit tests with pytest
   * - ``-s`` / ``--check-format``
     - Check code formatting (black, isort, flake8)
   * - ``-f`` / ``--fix-format``
     - Auto-fix code formatting issues
   * - ``-n`` / ``--notebook``
     - Run notebook tests using nbmake (see :ref:`notebook_testing`)
   * - ``-l`` / ``--check-license``
     - Check license headers in source files
   * - ``-c`` / ``--coverage``
     - Enable coverage reporting (use with ``-u``)
   * - ``-r`` / ``--test-report``
     - Generate JUnit XML test report (use with ``-u``)
   * - ``--clean``
     - Clean build artifacts and caches

Common Options
==============

These options can be used with any test command:

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``-v`` / ``--verbose``
     - off
     - Enable verbose output
   * - ``-d`` / ``--dry-run``
     - off
     - Print commands without executing
   * - ``--fresh-deps``
     - off
     - Force fresh dependency installation (bypass cache)

Dependency Caching
==================

The ``runtest.sh`` script automatically caches dependency installation to speed up repeated runs:

- **First run**: Dependencies are installed and a cache marker is created
- **Subsequent runs**: Dependencies are skipped (uses cache)
- **Auto-refresh**: Cache expires after **7 days** or **50 runs**, whichever comes first

The cache status is displayed at the start of each run:

.. code:: text

   Dependencies cached (5/50 runs, 2/7 days) - skipping install

To force a fresh dependency install:

.. code:: bash

   # Force reinstall dependencies
   ./runtest.sh -u --fresh-deps

   # Or clear all caches (including dependency cache)
   ./runtest.sh --clean

Examples
========

Unit Tests
----------

.. code:: bash

   # Run all unit tests
   ./runtest.sh -u

   # Run specific test file or directory
   ./runtest.sh -u tests/unit_test/fuel/

   # Run with coverage report
   ./runtest.sh -u -c

   # Run with verbose output
   ./runtest.sh -u -v

Code Quality
------------

.. code:: bash

   # Check formatting (doesn't modify files)
   ./runtest.sh -s

   # Auto-fix formatting issues
   ./runtest.sh -f

   # Check specific directory
   ./runtest.sh -s nvflare/apis/

Notebook Tests
--------------

See :ref:`notebook_testing` for detailed notebook testing options.

.. code:: bash

   # Test default notebook
   ./runtest.sh -n

   # Test specific notebook with verbose output
   ./runtest.sh -n -v examples/tutorials/flare_simulator.ipynb

Troubleshooting
===============

Slow Repeated Runs
------------------

If runs are slower than expected, check if dependency caching is working:

.. code:: text

   Dependencies cached (X/50 runs, Y/7 days) - skipping install

If you see "Dependencies not yet installed", the cache may have expired. This is normal behavior.

Force Clean State
-----------------

To start fresh:

.. code:: bash

   ./runtest.sh --clean
   ./runtest.sh -u  # Will reinstall dependencies


