.. _federated_data_catalog:

######################
Federated Data Catalog
######################

The federated data catalog lets the users of a Study discover what datasets each site
holds, without moving any data. Each site decides what metadata it publishes (table and
column names, types, keys and relationships). You read the catalog with one Python call
and get every site's answer back as one result. Catalog visibility never grants access to
the records themselves.

A catalog read is an ordinary FLARE job. It reads metadata only, never rows. Its result is
saved with the job in the job store and downloaded to you; the server keeps no other copy.

****************
The workflow
****************

.. code-block:: none

   site operator                          you (a Study member)
   -------------                          --------------------
   1. describe the datasets and their     2. get_catalog(env)
      catalog rules in datasets.yaml      3. explore catalog_results.json

1. A site operator describes, in the site's dataset configuration, where each dataset is,
   how to read it, and what may leave the site.
2. You run a catalog read. Every site of the Study reads its own sources and returns only
   what its rules allow.
3. You explore the result, offline if you like.

Each read returns the complete answer; the latest read is the current catalog. To keep it
fresh, run your read from cron or another scheduler.

*******************************************
Python API: read with ``get_catalog``
*******************************************

.. code-block:: python

   from nvflare.recipe import ProdEnv
   from nvflare.recipe.catalog import get_catalog

   env = ProdEnv(startup_kit_location="/path/to/admin", study="oncology")
   results = get_catalog(env, datasets={"site-1": ["omop"]}, output_path="catalog_results.json")

``get_catalog`` submits the read, waits for it, and returns the parsed
``catalog_results.json``. Of the job's downloaded workspace it keeps only that file; when the read fails, the whole
workspace is kept for diagnosis. ``SimEnv`` returns its live simulation workspace instead of a download, and that
workspace is left unchanged. It works in any recipe environment: ``SimEnv``, ``PocEnv`` or ``ProdEnv``.

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - Argument
     - Default
     - Meaning
   * - ``env``
     - required
     - Where the read runs, and which Study it reads.
   * - ``sites``
     - every site of the Study
     - Sites to read.
   * - ``datasets``
     - everything each site shares
     - ``{site: [names]}``. A site reads the listed names it shares and skips the rest.
       A site not in the mapping reads everything it shares.
   * - ``timeout_s``
     - ``300``
     - Seconds to wait for each site before recording it as ``NO_RESULT``.
   * - ``output_path``
     - none
     - Also save ``catalog_results.json`` here.
   * - ``site_config_root``
     - none
     - Required with ``SimEnv``: a folder with one sub-folder per site, each holding that
       site's ``local/datasets.yaml``. Refused in other environments.
   * - ``task_request_interval_s``
     - ``0.1``
     - How often sites ask the server for work during this job.

A completed read returns normally even when some sites failed; each site's status is in
``results["sites"]``. ``get_catalog`` raises ``CatalogReadError`` when the job itself does
not finish successfully or the wait expires.

``get_catalog`` is the ``CatalogRecipe`` job in one call. Use ``CatalogRecipe`` directly to
control the job, for example to submit without waiting:

.. code-block:: python

   from nvflare.recipe.catalog import CatalogRecipe

   recipe = CatalogRecipe(datasets={"site-1": ["omop"]})
   run = recipe.execute(env)
   results = recipe.get_catalog_results(run, output_path="catalog_results.json")

Neither carries a connection string, credential, path or source type. The same call works
when one site stores a dataset as Parquet, another in PostgreSQL and a third in DuckDB.

*******************
The result
*******************

``catalog_results.json`` has one entry per site in ``sites``, with its status (``OK``,
``FAILED`` or ``NO_RESULT``), and every published dataset in ``datasets``, with its site,
tables, columns, types, nullability, keys and relationships. Datasets a site could not read
are listed as ``unavailable``; datasets its read budget did not reach are listed as
``skipped``. A dataset a site does not share does not appear at all, not even by name.

*************************************
Visualize saved catalog results
*************************************

Like federated statistics visualization, ``CatalogVisualization`` accepts a saved result
and provides reusable views. It opens no federation connection. Install ``pandas`` and
``jinja2`` for these optional visualization functions; the example notebook requirements
include them through pandas and JupyterLab.

.. code-block:: python

   from nvflare.app_opt.fdn.catalog.visualization import CatalogVisualization

   vis = CatalogVisualization("catalog_results.json")   # or the dict get_catalog returned
   vis.show_sites()
   vis.show_datasets()
   vis.show_tables(dataset="omop")
   vis.show_schema("person", site="site-1", dataset="omop")
   vis.show_table_comparison(dataset="omop")
   vis.show_column_comparison("person", dataset="omop")
   vis.show_column_comparison("person", dataset="omop", field="nullable")
   vis.show_relationships(dataset="omop")

Each rendered view includes a title, which names the result's Study, and an explanation;
``vis.study`` returns the Study. Each comparison column identifies both
the site and dataset, so several databases at one site remain distinct. Red marks metadata
not published by a source; gray marks a dataset that was not read; yellow marks different
reported column definitions. Unknown nullability is shown as ``unknown``. Type names remain
as reported by their source.

``show_*`` methods return pandas Stylers that Jupyter renders. Their ``get_*`` counterparts
return DataFrames for further analysis or export. Both catalog examples include an offline
``catalog_explorer.ipynb``; the services example includes a checked-in synthetic result.
For a complete workflow, open ``get_catalog.ipynb`` in either example. It reads the catalog
and invokes ``catalog_explorer.ipynb`` with the fresh result.

*****************************************
What sites configure (site operators)
*****************************************

Each site describes its datasets per Study in one file, its dataset configuration,
``<workspace>/local/datasets.yaml``. A job cannot supply or override it.

.. code-block:: yaml

   format_version: 1
   studies:
     oncology:
       datasets:
         omop:
           type: parquet                       # selects the connector
           uri: file:///data/oncology/omop     # where the dataset is
           usage:
             catalog:                          # the catalog's rules for this dataset
               share: true
               exclude_columns: {person: [person_source_value]}

The dataset's ``type`` selects the connector. ``parquet``, ``csv``, ``sql`` and
``unity-catalog`` are built in; a site declares others, such as DuckDB, by class:

.. code-block:: yaml

   usage:
     catalog:
       connectors:
         duckdb: {class: nvflare.app_opt.fdn.catalog.duckdb_connector.DuckDBConnector}

A dataset without ``usage.catalog`` rules, or with ``share: false``, is not published.
Excluded tables and columns, and any keys and relationships that depend on them, are
removed before anything leaves the site. An exclusion that matches nothing is only logged as
a warning at the site, so check its spelling. Network sources name credentials as secrets:
environment variable names, never values.

**********
Try it
**********

Two runnable examples:

* ``examples/advanced/federated-catalog`` runs on one host in a simulation, with no Docker
  and no server. Three sites with different backends (Parquet and CSV, SQLite, DuckDB) hold
  OMOP-shaped data. ``python prepare_data.py`` and ``python job.py`` print what each site
  published and save ``catalog_results.json``.
* ``examples/advanced/federated-catalog-services`` uses Docker to simulate a production
  deployment: a provisioned federation with a Unity Catalog server and a PostgreSQL
  database, with credentials kept at the site. ``python job.py`` reads the catalog with
  ``get_catalog``. Both examples include an offline visualization notebook using the same
  ``CatalogVisualization`` package.

The example data is synthetic and OMOP-shaped for demonstration. It is not validated
against the OHDSI vocabulary.

**********
Limits
**********

* The catalog is shared by all users of the Study. It does not provide per-user views.
* A site that does not answer in time, or a site named in ``sites`` that is offline, is
  reported as ``NO_RESULT`` in that read; nothing from an earlier read is carried over. Without
  ``sites``, a read covers the sites that are online when it starts.
* Each read adds one job to the server's job store.
* Connectors publish only what a source reports. Parquet files carry no keys, and
  descriptions or declared standards are not published.
* The catalog is for discovery. Using a dataset still goes through the Study's normal
  authorization.
