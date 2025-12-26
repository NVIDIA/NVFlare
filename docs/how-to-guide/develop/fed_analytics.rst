.. _fed_analytics_guide:

####################################
How to Calculate Federated Analytics
####################################

NVIDIA FLARE enables collaborative data analysis across multiple sites without sharing raw data. Federated Analytics
focuses on computing global statistics (e.g., counts, distributions, means) by aggregating local analytics results
computed at each participant.

**When Should You Use Federated Analytics?**

- Use Federated Analytics when you want to:
- Understand data distribution and quality across institutions
- Perform cohort discovery or feasibility analysis
- Validate dataset compatibility before federated training

**Common outputs include:**

- Counts and histograms
- Summary statistics (mean, min, max, std)
- Label or class distributions

**Overview**

NVIDIA FLARE provides built-in federated statistics operators that can generate global statistics based on local client-side statistics.
At each client site, we can have one or more datasets (such as "train" and "test" datasets); each dataset may have many features.
For each feature in the dataset, we will calculate the statistics and then combine them to produce global statistics
for all the numeric features. The output will be complete statistics for all datasets in clients and global.

The statistics here are commonly used statistics: count, sum, mean, std_dev, quantiles and histogram for the numerical features.
The max, min are not included as it might violate the client's data privacy. Quantiles require an additional dependency.
If sum and count statistics are selected, the mean will be calculated with count and sum.

A client will only need to implement the selected methods of "Statistics" class from statistics_spec.
The result will be statistics for all features of all datasets at all sites as well as global aggregates.
The result should be visualized via the visualization utility in the notebook.

**Assumptions** -- We only support numerical features, not categorical features, the non-numerical features will be removed.

**Statistics**
Federated statistics includes numeric statistical measures for

count
mean
sum
std_dev
histogram
quantile
We do not include min and max values to avoid data privacy concerns.

**Steps**
**Step 0** -- user needs to provide the target data source names (such as train, test) as well as the features names
**Step 1** -- user provides configuration to specify target statistics metrics and output location
**Step 2** -- user provide the local implementation statistics generator (statistics_spec)
also, provide client side configuration to specify data input location

The detailed example instructions can be found Data frame statistics

here is one example

.. code-block:: text

    statistic_configs = {
        "count": {},
        "mean": {},
        "sum": {},
        "stddev": {},
        "histogram": {"*": {"bins": 20}, "Age": {"bins": 20, "range": [0, 100]}},
        "quantile": {"*": [0.1, 0.5, 0.9]},
    }

This configuration states, that we like to calculate count, mean, sum and stddev for all features. For histogram, all features will
have 20 bins and data range will be calculated based on data, except for age feature where the bin is still 20, but range
is fixed between 0 to 100. For qll features, we like to calculate the 10%, 50% (Median) and 90% quantile.

Take example for Fed Statistics for tabular data, many of the functions needed for tabular statistics have already been
implemented ```DFStatisticsCore```, as result, data scientists only need to specify the JobRecipe.


Client Code
-----------

Local statistics generator. The statistics generator `AdultStatistics` implements `Statistics` spec.

.. literalinclude:: ../../examples/hello-world/hello-tabular-stats/client.py
    :language: python
    :linenos:
    :caption: Client Code (client.py)
    :lines: 14-


Many of the functions needed for tabular statistics have already been implemented DFStatisticsCore

In the `AdultStatistics` class, we really need to have the followings

- data_features -- here we hard-coded the feature name array.
- implement `load_data() -> Dict[str, pd.DataFrame]` function, where
  the method will return a dictionary of panda DataFrames with one for each data source ("train", "test")
- `data_path = <data_root_dir>/<site-name>/<filename>`



Job Recipe
----------

Job is defined via recipe, we will run it in Simulation Execution Env.

.. literalinclude:: ../../examples/hello-world/hello-tabular-stats/job.py
    :language: python
    :linenos:
    :caption: job Recipe (job.py)
    :lines: 14-

Run Job
-------
from terminal try to run the code

.. code-block:: text

    python job.py


For complete example of `../../examples/hello-world/hello-tabular-stats` or find more about other examples in
`../../examples/advanced/federated_statistics`



