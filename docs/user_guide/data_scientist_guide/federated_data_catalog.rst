.. _federated_data_catalog:

######################
Federated Data Catalog
######################

The federated data catalog lets the users of a Study discover what datasets each site
holds, without moving any data. Each site decides what metadata it publishes (table and
column names, types, keys and relationships). You read the catalog with one Python call
and get every site's answer back as one result. Catalog visibility never grants access to
the records themselves.

A catalog read is an ordinary FLARE job. Each site reads schema metadata directly from its configured sources. Each successful dataset contributes one ODCS v3.2.0 contract. CSV discovery publishes header names only. Its result is
saved with the job in the job store and downloaded to you; the server keeps no other copy.

****************
The workflow
****************

.. code-block:: none

   site operator                          you (a Study member)
   -------------                          --------------------
   1. describe the datasets and their     2. get_catalog(env)
      source identity in datasets.yaml      3. explore catalog_results.json

1. A site operator describes, in the site's dataset configuration, where each dataset is,
   how to read it, and what may leave the site.
2. You run a catalog read. Every site of the Study reads its own sources and returns only
   the metadata its source identity can discover.
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
   results = get_catalog(env, datasets={"site-1": ["clinical_records"]}, output_path="catalog_results.json")

``get_catalog`` submits the read, waits for it, and returns a version 2 snapshot. Each dataset entry contains ``site``, ``dataset`` and ``contract``.
The download retains ``catalog/index.json`` and standalone ODCS files under ``catalog/contracts/``; when the read fails, the whole
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
     - the sites participating when the job starts
     - Explicitly requested sites that do not participate appear as ``NO_RESULT``.
   * - ``datasets``
     - all configured datasets at each participating site
     - ``{site: [names]}``. A site reads the listed names present in its Study configuration.
       A site absent from the mapping reads all its configured Study datasets.
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

   recipe = CatalogRecipe(datasets={"site-1": ["clinical_records"]})
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
``skipped``. Only selected, configured datasets can have their own result entries. Source-reported lineage may also name datasets outside the selection or Study configuration.

*************************************
Visualize saved catalog results
*************************************

Pass the job's default ``catalog/`` output directory (the folder containing ``index.json``
and ``contracts/``) directly to ``CatalogVisualization``. The viewer loads the referenced
contracts offline. It also accepts a saved snapshot JSON file or the dict returned by
``get_catalog()``. Install ``pandas`` and
``jinja2`` for these optional visualization functions; the example notebook requirements
include them through pandas and JupyterLab.

.. code-block:: python

   from nvflare.app_opt.fdn.catalog.visualization import CatalogVisualization

   vis = CatalogVisualization("path/to/job-output/catalog")
   vis.show_sites()
   vis.show_datasets()
   vis.show_tables(dataset="clinical_records")
   vis.show_schema("person", site="site-1", dataset="clinical_records")
   vis.show_table_comparison(dataset="clinical_records")
   vis.show_column_comparison("person", dataset="clinical_records")
   vis.show_column_comparison("person", dataset="clinical_records", field="nullable")
   vis.show_relationships(dataset="clinical_records")
   vis.show_contract(site="site-2", dataset="clinical_records")

Each table includes a title naming the Study and an explanation of its contents.
``vis.study`` returns the Study. Each comparison column identifies both
the site and dataset, so several databases at one site remain distinct. Red marks metadata
not published by a source; gray marks a dataset that was not read; yellow marks different
reported column definitions. Unknown nullability is shown as ``unknown``. Type names remain
as reported by their source.

``show_contract`` displays a published ODCS document in Jupyter using the Data Contract CLI
renderer included with ``nvflare[catalog]``. Tabular views return pandas Stylers, and their
``get_*`` counterparts return DataFrames for further analysis or export. Both catalog examples
include an offline ``catalog_explorer.ipynb``. The services example includes a checked-in synthetic result.
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
         clinical_records:
           type: parquet                       # selects the source reader
           uri: file:///data/oncology/clinical_records     # where the dataset is

The site installs ``nvflare[catalog]``. The examples use ``csv``, ``parquet``, ``sql``, ``duckdb`` and
``unity-catalog``. For another network source, use the name of an installed Data Contract CLI metadata
importer as the dataset type. For example, a site with ``datacontract-cli[snowflake]==1.2.11`` can add:

.. code-block:: yaml

   clinical_records:
     type: snowflake
     uri: snowflake://acme-west
     options: {metadata_source: host, database: CLINICAL_WAREHOUSE, schema: EHR}
     secrets:
       DATACONTRACT_SNOWFLAKE_USERNAME: SITE_DB_USER
       DATACONTRACT_SNOWFLAKE_PASSWORD: SITE_DB_PASSWORD

Use ``metadata_source: host`` for an importer that expects a bare host or account and ``metadata_source: uri``
for one that expects a network URI. Other options use the upstream importer's argument names.
Use ``options.config`` for nonsecret upstream settings and ``secrets`` for credential bindings.
The site verifies that the importer and service identity only read metadata. The researcher calls the same
``get_catalog()`` for the first read and each refresh.

Every dataset configured for the job's Study is eligible for cataloging, optionally narrowed by the request's
dataset selection. The catalog does not apply sharing flags or table, column, resource or metadata exclusions.
The database administrator controls metadata visibility through scoped credentials and source namespaces or
selected projections. Verify metadata visibility separately from row access: database system catalogs may expose
schemas even when the identity cannot read data. For files, control which paths the job can read.

Review every dataset in the Study’s ``datasets.yaml``, including entries without ``usage.catalog``. Remove datasets the Study should not discover,
or expose approved metadata through a database view. For the live PostgreSQL importer, explicitly select that view with
``options: {tables: [approved_view]}`` in the dataset entry. Without a table list it discovers tables and views visible in the configured PostgreSQL schema.
Remove obsolete per-dataset sharing and exclusion rules. If any remain, the site reports
``CONFIG_UNAVAILABLE`` for the Study without naming datasets. Site-wide ``usage.catalog.limits``
sets the read budgets. A PostgreSQL dataset may specify ``?sslmode=verify-full`` in its URI and use the site’s ``PGSSLROOTCERT`` CA file. Network credentials are secret
environment variable names, never values in the configuration or result. Source changes affect the next read;
previously downloaded snapshots are not retracted.

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

* Each read produces a job result. Access follows the Study’s job-download permissions.
* A site that does not answer in time, or a site named in ``sites`` that is offline, is
  reported as ``NO_RESULT`` in that read; nothing from an earlier read is carried over. Without
  ``sites``, a read covers the sites that are online when it starts.
* Each read adds one job to the server's job store.
* Contracts contain structural fields from the source metadata.
* The catalog is for discovery. Using a dataset still goes through the Study's normal
  authorization.
