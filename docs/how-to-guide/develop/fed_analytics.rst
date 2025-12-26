.. _fed_analytics_guide:

####################################
How to Calculate Federated Analytics
####################################

NVIDIA FLARE enables collaborative data analysis across multiple sites without sharing raw data. Federated Analytics
focuses on computing global statistics (such as counts, distributions, and means) by aggregating local analytics results
computed at each participant.

When to Use Federated Analytics
-------------------------------

Use Federated Analytics when you want to:

- Understand data distribution and quality across institutions
- Perform cohort discovery or feasibility analysis
- Validate dataset compatibility before federated training

Common outputs include:

- Counts and histograms
- Summary statistics (mean, min, max, standard deviation)
- Label or class distributions


Overview
--------

NVIDIA FLARE provides built-in federated statistics operators that generate global statistics based on local client-side statistics.
At each client site, you can have one or more datasets (such as "train" and "test" datasets), and each dataset may have many features.
For each feature in the dataset, the system calculates local statistics and then combines them to produce global statistics
for all numeric features. The output includes complete statistics for all datasets across all clients, as well as global aggregates.

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
------------------

**Step 0**: Provide the target data source names (such as "train", "test") and the feature names.

**Step 1**: Configure the target statistics metrics and output location.

**Step 2**: Implement the local statistics generator using the ``Statistics`` spec and configure the client-side data input location.

For detailed instructions, see the :ref:`hello_tabular_stats` example.

Here is an example configuration:

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
- For histograms, all features will have 20 bins with data range calculated automatically, except for the "Age" feature where the range is fixed between 0 and 100
- For all features, calculate the 10%, 50% (median), and 90% quantiles

For tabular data statistics, many of the required functions have already been implemented in ``DFStatisticsCore``.
As a result, data scientists only need to specify the Job Recipe.


Client Code
-----------

The local statistics generator ``AdultStatistics`` implements the ``Statistics`` spec:

.. literalinclude:: ../../examples/hello-world/hello-tabular-stats/client.py
    :language: python
    :linenos:
    :caption: Client Code (client.py)
    :lines: 14-

Many of the functions needed for tabular statistics have already been implemented in ``DFStatisticsCore``.

In the ``AdultStatistics`` class, you need to provide:

- **data_features** - Array of feature names (hardcoded in this example)
- **load_data()** - Method that returns a dictionary of Pandas DataFrames, one for each data source ("train", "test")
- **data_path** - Path in the format ``<data_root_dir>/<site-name>/<filename>``


Job Recipe
----------

The job is defined via a recipe and runs in the Simulation Execution Environment:

.. literalinclude:: ../../examples/hello-world/hello-tabular-stats/job.py
    :language: python
    :linenos:
    :caption: Job Recipe (job.py)
    :lines: 14-


Run the Job
-----------

From the terminal, run the job script:

.. code-block:: bash

    python job.py


Additional Resources
--------------------

- Complete example: :ref:`hello_tabular_stats`
- More examples and detailed documentation: :ref:`federated_statistics`
