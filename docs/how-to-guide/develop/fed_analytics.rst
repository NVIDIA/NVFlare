.. _fed_analytics_guide:

####################################
How to Calculate Federated Analytics
####################################

NVIDIA FLARE enables collaborative data analysis across multiple sites without sharing raw data. Federated Analytics
focuses on computing global statistics (such as counts, distributions, and means) by aggregating local analytics results
computed at each participant.

When to Use Federated Analytics
================================

Use Federated Analytics when you want to:

- Understand data distribution and quality across institutions
- Perform cohort discovery or feasibility analysis
- Validate dataset compatibility before federated training

Common outputs include:

- Counts and histograms
- Summary statistics (mean, sum, standard deviation)
- Label or class distributions


Overview
========

NVIDIA FLARE provides built-in federated statistics operators that generate global statistics based on local
client-side statistics. At each client site, you can have one or more datasets (such as "train" and "test"),
and each dataset may have many features. For each feature, the system calculates local statistics and then
combines them to produce global statistics for all numeric features. The output includes complete statistics
for all datasets across all clients, as well as global aggregates.

The supported statistics include commonly used measures:

- **count** - Number of samples
- **sum** - Sum of values
- **mean** - Average value (calculated from count and sum if both are selected)
- **stddev** - Standard deviation
- **histogram** - Distribution of values across bins
- **quantile** - Percentile values (requires additional dependency)

.. note::

   We do not include min and max values to avoid data privacy concerns.
   Only numerical features are supported; non-numerical features will be removed automatically.


Steps to Implement
==================

1. Provide the target data source names (such as "train", "test") and the feature names
2. Configure the target statistics metrics and output location
3. Implement the local statistics generator using the ``Statistics`` spec and configure the client-side data input location

For detailed instructions, see the :ref:`hello_tabular_stats` example.

Example Configuration
---------------------

Here is an example statistics configuration:

.. code-block:: text

    statistic_configs = {
        "count": {},
        "mean": {},
        "sum": {},
        "stddev": {},
        "histogram": {"*": {"bins": 20}, "Age": {"bins": 20, "range": [0, 100]}},
        "quantile": {"*": [0.1, 0.5, 0.9]},
    }

This configuration specifies:

- Calculate count, mean, sum, and stddev for all features
- For histograms, use 20 bins with auto-calculated range, except "Age" which uses a fixed range of 0–100
- For quantiles, calculate the 10%, 50% (median), and 90% percentiles for all features

For tabular data, many required functions are implemented in ``DFStatisticsCore``, so data scientists
only need to provide the data loader and configure the Job Recipe.


Client Code
-----------

The local statistics generator ``AdultStatistics`` implements the ``Statistics`` spec:

.. literalinclude:: ../../../examples/hello-world/hello-tabular-stats/client.py
    :language: python
    :linenos:
    :caption: client.py
    :lines: 14-

The ``AdultStatistics`` class extends ``DFStatisticsCore`` and provides:

- ``data_features``: Array of feature names
- ``load_data()``: Returns a dictionary of Pandas DataFrames (one per data source)
- ``data_path``: Path in the format ``<data_root_dir>/<site-name>/<filename>``


Job Recipe
----------

The job is defined via a recipe and runs in the simulation environment:

.. literalinclude:: ../../../examples/hello-world/hello-tabular-stats/job.py
    :language: python
    :linenos:
    :caption: job.py
    :lines: 14-


Run the Job
-----------

From the terminal, run the job script:

.. code-block:: bash

    python job.py


References
----------

- :ref:`hello_tabular_stats` - Complete tabular statistics example
- :ref:`federated_statistics` - Detailed federated statistics documentation
