# Tabular Federated Statistics: Deep dive into the implementations


This example is the same as [Hello-tabular-stats](../../../hello-world/hello-tabular-stats/README.md). For a basic example,
please read that example first. 

Here we would like to describe a few other advanced topics not covered in [Hello-tabular-stats](../../../hello-world/hello-tabular-stats/README.md)

## Data Preparation

In this example, we are using UCI (University of California, Irvine) [adult dataset](https://archive.ics.uci.edu/dataset/2/adult)

The original dataset already contains "training" and "test" datasets. Here we simply assume that the "training" and "test" datasets belong to different clients.
So we assigned the training data and test data to two clients.
 
Now we use the data utility to download the UCI dataset into separate client directories under /tmp/nvflare/data/

Please note that the UCI's website may experience occasional downtime.

```shell
python prepare_data.py
```
It should show something like
```
prepare data for data directory /tmp/nvflare/df_stats/data

download to /tmp/nvflare/df_stats/data/site-1/data.csv
skip empty line


download to /tmp/nvflare/df_stats/data/site-2/data.csv
skip empty line

done with prepare data
```

## Assumptions
* We assume each site has the same features (schema).
* Each site can calculate the local statistics. 

## Quantile Calculation

The design choice for quantile calculation: 

To calculate federated quantiles, we need to select a package that satisfies the following constraints:

* Works in distributed systems
* Does not copy the original data (avoiding privacy leaks)
* Avoids transmitting large amounts of data
* Ideally, has no system-level dependencies 

We chose the fastdigest Python package, a Rust-based package. The digest only carries the cluster coordinates; 
initially, each data point is in its own cluster. By default, we will compress with max_bin = sqrt(datasize) 
to compress the coordinates, so the data won't leak. You can always override max_bins if you prefer more or less compression.

Install and pin:

```bash
pip install fastdigest==0.4.0
```

 
## Configuration and Code

Since FLARE has already developed the operators for the federated
statistics computing, we will only need to provide the following:
* config_fed_server.json (server-side controller configuration)
* config_fed_client.json (client-side executor configuration)
* local statistics calculator

The same configuration can be achieved via the FLARE Job Recipe API, but let's look at the configuration first

### Server-Side Configuration

The server-side configuration specifies the overall statistics metrics one would like to calculate, 
such as stddev, histogram, the number of bins used for histograms for each feature, and whether quantiles should be calculated. 
This can be specified via Job Recipe. 

```
    statistic_configs = {
        "count": {},
        "mean": {},
        "sum": {},
        "stddev": {},
        "histogram": {"*": {"bins": 20}, "Age": {"bins": 20, "range": [0, 100]}},
        "quantile": {"*": [0.1, 0.5, 0.9]},
    }
 
```
 
In the above configuration, we ask FLARE to calculate the following statistics: "count", "mean", "sum", "stddev", "histogram", and "quantile". Each statistic may have its own configuration.
For example, for the Histogram statistic, we specify feature "Age" needs 20 bins and the histogram range is within [0, 100), while for
all other features ("*" indicates default feature), the bins are 20, and the range is not specified, i.e., the ranges will be dynamically estimated.



### Client-Side Implementation: Local Statistics Generator
 
The statistics generator `DFStatistics` implements the `Statistics` spec. In the current example, the input data is in the format of Pandas DataFrame. Although we used a csv file, this can be any
tabular data format that can be expressed as a pandas dataframe.

```
class DFStatistics(Statistics):
    # rest of code 
```
to calculate the local statistics, we will need to implement a few methods
```
    def features(self) -> Dict[str, List[Feature]]:

    def count(self, dataset_name: str, feature_name: str) -> int:
 
    def sum(self, dataset_name: str, feature_name: str) -> float:
 
    def mean(self, dataset_name: str, feature_name: str) -> float:
 
    def stddev(self, dataset_name: str, feature_name: str) -> float:
 
    def variance_with_mean(self, dataset_name: str, feature_name: str, global_mean: float, global_count: float) -> float:
 
    def histogram(self, dataset_name: str, feature_name: str, num_of_bins: int, global_min_value: float, global_max_value: float) -> Histogram:

    def quantiles(self, dataset_name: str, feature_name: str, percentiles: List) -> Dict:

```
Since some features do not provide histogram bin range, we will need to calculate based on local min/max to estimate
the global min/max, and then use the global min/max as the range for all clients' histogram bin range.

So we need to provide local min/max calculation methods
```
   def max_value(self, dataset_name: str, feature_name: str) -> float:
   def min_value(self, dataset_name: str, feature_name: str) -> float:
```

For tabular data, FLARE has already implemented the specification with `DFStatisticsCore`. We just need to subclass `DFStatisticsCore` and implement
a few methods
```
    def load_data(self, fl_ctx: FLContext) -> Dict[str, pd.DataFrame]:
```

This method specifies how to load the data into different DataFrames, for example, "train" dataset and "test" dataset, one for each key in the dictionary.

```
   def features(self) -> Dict[str, List[Feature]]:
   
```
The features for each dataset are also important. We assume each site has the same features. 

By default, the `DFStatisticsCore.features(self) -> Dict[str, List[Feature]]` implementation assumes the DataFrame has feature names.
The method simply retrieves the feature names from the DataFrame. 

```
    def features(self) -> Dict[str, List[Feature]]:
        results: Dict[str, List[Feature]] = {}
        for ds_name in self.data:
            df = self.data[ds_name]
            results[ds_name] = []
            for feature_name in df:
                data_type = dtype_to_data_type(df[feature_name].dtype)
                results[ds_name].append(Feature(feature_name, data_type))

        return results

```
Therefore, the DataFrame must have the column names defined. In this example, the dataset has no headers in the CSV file, 
so we hard-coded the feature names in the `__init__()` function.

If you can't derive features, you can overwrite this method and return a list of features for each dataset. 


```

class AdultStatistics(DFStatisticsCore):
    def __init__(self, filename, data_root_dir="/tmp/nvflare/df_stats/data"):
        super().__init__()
        self.data_root_dir = data_root_dir
        self.filename = filename
        self.data: Optional[Dict[str, pd.DataFrame]] = None
        self.data_features = [
            "Age",
            "Workclass",
            "fnlwgt",
            "Education",
            "Education-Num",
            "Marital Status",
            "Occupation",
            "Relationship",
            "Race",
            "Sex",
            "Capital Gain",
            "Capital Loss",
            "Hours per week",
            "Country",
            "Target",
        ]

        # The original dataset has no header.
        # We will use the adult.train dataset for site-1 and the adult.test dataset for site-2.
        # The adult.test dataset has an incorrectly formatted row at the 1st line, so we will skip it.
        self.skip_rows = {
            "site-1": [],
            "site-2": [0],
        }

    def load_data(self, fl_ctx: FLContext) -> Dict[str, pd.DataFrame]:
        client_name = fl_ctx.get_identity_name()
        self.log_info(fl_ctx, f"load data for client {client_name}")
        try:
            skip_rows = self.skip_rows[client_name]
            data_path = f"{self.data_root_dir}/{fl_ctx.get_identity_name()}/{self.filename}"
            # example of load data from CSV
            df: pd.DataFrame = pd.read_csv(
                data_path, names=self.data_features, sep=r"\s*,\s*", skiprows=skip_rows, engine="python", na_values="?"
            )
            train = df.sample(frac=0.8, random_state=200)  # random state is a seed value
            test = df.drop(train.index).sample(frac=1.0)

            self.log_info(fl_ctx, f"load data done for client {client_name}")
            return {"train": train, "test": test}

        except Exception as e:
            raise Exception(f"Load data for client {client_name} failed! {e}")

    def initialize(self, fl_ctx: FLContext):
        self.data = self.load_data(fl_ctx)
```


### Client-Side Data Privacy Configuration


To ensure privacy, each site wants to specify the min_count. For the min value, we want to add 10% noise,
and for the max value, we want to add 30% noise. 

max_bins_percent: The maximum number of bins allowed in terms of percentage of local data size.

Set this number to avoid the number of bins being equal to or close to the data size, which can lead to data leakage.
For example: max_bins_percent = 15 means the number of bins should not be more than 15% of the data count

```
   min_count = 15
   min_noise_level = 0.1
   max_noise_level = 0.3
   max_bins_percent = 15
```
This privacy control can be expressed in the job recipe at the job level. But it can also be applied at the organization privacy 
policy level, for example, using a configuration file in the [local privacy policy](../local/privacy.json)


## Job Recipe

* First, follow the data preparation steps in the "Data Preparation" section above to get the data
* Now run the job with 

```bash
   python job.py
```

You should see something like

```
2026-02-02 17:44:38,332 - INFO - Save statistics result to persistence store
2026-02-02 17:44:38,333 - INFO - job dir = /tmp/nvflare/simulation/stats_df/server/simulate_job
2026-02-02 17:44:38,337 - INFO - trying to save data to /tmp/nvflare/simulation/stats_df/server/simulate_job/statistics/adults_stats.json
2026-02-02 17:44:38,338 - INFO - file /tmp/nvflare/simulation/stats_df/server/simulate_job/statistics/adults_stats.json saved
```
The results are stored in workspace "/tmp/nvflare"
```
/tmp/nvflare/simulation/stats_df/server/simulate_job/statistics/adults_stats.json
```

## Visualization

With the JSON format, the data can be easily visualized via pandas dataframe and plots. 
You can use the visualization notebook from the [Hello-tabular-stats](../../../hello-world/hello-tabular-stats/README.md) example.

Assuming the NVFLARE_HOME environment variable points to the GitHub project location (NVFlare) which contains the current example.

Copy the output adults_stats.json file to the hello-tabular-stats demo directory, then you can run the jupyter notebook:

```bash
cp /tmp/nvflare/simulation/stats_df/server/simulate_job/statistics/adults_stats.json $NVFLARE_HOME/examples/hello-world/hello-tabular-stats/demo/.

cd $NVFLARE_HOME/examples/hello-world/hello-tabular-stats/demo

jupyter notebook visualization.ipynb
```

You should be able to get the visualization similar to the statistics tables and histogram plots shown in the hello-tabular-stats example.