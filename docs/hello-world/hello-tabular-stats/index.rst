Tabular Data Federated Statistics
=================================

In this example, we will show how to generate federated statistics for tabular data that can be represented as Pandas Data Frame.


NVIDIA FLARE Installation
-------------------------
for the complete installation instructions, see `installation <../../installation.html>`_

.. code-block:: text

    pip install nvflare


get the example code from github:

.. code-block:: text

    git clone https://github.com/NVIDIA/NVFlare.git

then navigate to the hello-tabular-stats directory:

.. code-block:: text

    git switch <release branch>
    cd examples/hello-world/hello-tabular-stats


Install the dependency
----------------------

    pip install -r requirements.txt


Install Optional Quantile Dependency -- fastdigest
------------------------------------------------------------

If you intend to calculate quantiles, you need to install fastdigest.

Skip this step if you don't need quantile statistics.

.. code-block:: text

    pip install fastdigest==0.4.0


on Ubuntu, you might get the following error:

.. code-block:: text

  Cargo, the Rust package manager, is not installed or is not on PATH.
  This package requires Rust and Cargo to compile extensions. Install it through
  the system's package manager or via https://rustup.rs/

  Checking for Rust toolchain....

This is because fastdigest (or its dependencies) requires Rust and Cargo to build.

You need to install Rust and Cargo on your Ubuntu system. Follow these steps:
Install Rust and Cargo
Run the following command to install Rust using rustup:

.. code-block:: text

    ./install_cargo.sh

Then you can install fastdigest again

.. code-block:: text

    pip install fastdigest==0.4.0


Code Structure
--------------

.. code-block:: text

    hello-tabular-stats
    |
    ├── client.py         # client local training script
    ├── job.py            # job recipe that defines client and server configurations
    ├── prepare_data.py   # utilies to download data
    ├── install_cargo.sh  # scripts to install rust and cargo needed for quantil dependency, only needed if you plan to inistall quantile dependency
    └── requirements.txt  # dependencies
    ├── demo
    │   └── visualization.ipynb # Visualization Notebook


Data
----

In this example, we are using UCI (University of California, Irvine) [adult dataset](https://archive.ics.uci.edu/dataset/2/adult)

The original dataset has already contains "training" and "test" datasets. Here we simply assume that "training" and test data sets are belong to different clients.
so we assigned the training data and test data into two clients.

Now we use data utility to download UCI datasets to separate client package directory to /tmp/nvflare/data/ directory

Please note that the UCI's website may experience occasional downtime.

.. code-block:: text

    python prepare_data.py

it should show something like

prepare data for data directory /tmp/nvflare/df_stats/data

.. code-block:: text

    download to /tmp/nvflare/df_stats/data/site-1/data.csv
    skip empty line


    download to /tmp/nvflare/df_stats/data/site-2/data.csv
    skip empty line

    done with prepare data


Client Code
-----------

Local statistics generator. The statistics generator `AdultStatistics` implements `Statistics` spec.

.. literalinclude:: ../../../examples/hello-world/hello-tabular-stats/client.py
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

Server Code
-----------
The server aggregation have already implemented in Statistics Controller

Job Recipe
----------

Job is defined via recipe, we will run it in Simulation Execution Env.

.. literalinclude:: ../../../examples/hello-world/hello-tabular-stats/job.py
    :language: python
    :linenos:
    :caption: job Recipe (job.py)
    :lines: 14-



The statistics configuration determines which statistics we need generate
Here is an example

.. code-block:: text

    statistic_configs = {
        "count": {},
        "mean": {},
        "sum": {},
        "stddev": {},
        "histogram": {"*": {"bins": 20}, "Age": {"bins": 20, "range": [0, 100]}},
        "quantile": {"*": [0.1, 0.5, 0.9]},
    }


Run Job
-------
from terminal try to run the code

.. code-block:: text

    python job.py


You should see something like

.. code-block:: text

    2025-09-03 20:42:03,392 - INFO - save statistics result to persistence store
    2025-09-03 20:42:03,392 - INFO - job dir = /tmp/nvflare/simulation/stats_df/server/simulate_job
    2025-09-03 20:42:03,395 - INFO - trying to save data to /tmp/nvflare/simulation/stats_df/server/simulate_job/statistics/adults_stats.json
    2025-09-03 20:42:03,395 - INFO - file /tmp/nvflare/simulation/stats_df/server/simulate_job/statistics/adults_stats.json saved


The results are stored in workspace "/tmp/nvflare"

.. code-block:: text

    /tmp/nvflare/simulation/stats_df/server/simulate_job/statistics/adults_stats.json


## Visualization
   with json format, the data can be easily visualized via pandas dataframe and plots.
   download and copy the output adults_stats.json file to demo directory, then you can run jupyter notebook visualization.ipynb





