# Tabular Federated Statistics: Deep dive into the implementations


This example is the same as [Hello-tabular-stats](../../../hello-world/hello-tabular-stats/README.md), for basic example
please read that example first. 

Here we like to describe a few other advanced topics not described in [Hello-tabular-stats](../../../hello-world/hello-tabular-stats/README.md)

## Assumption
* We are assuming each site has the same features (schema)
* Each site can calculate the local statistics 

## Quantile Calculation

The design choice of Quantile calculation: 

To calculate federated quantiles, we needed to select a package that satisfies the following constraints:

* Works in distributed systems
* Does not copy the original data (avoiding privacy leaks)
* Avoids transmitting large amounts of data
* Ideally, no system-level dependency 

We chose the fastdigest python package, a rust-based package. tdigest only carries the cluster coordinates, 
initially each data point is in its own cluster. By default, we will compress with max_bin = sqrt(datasize) 
to compress the coordinates, so the data won't leak. You can always override max_bins if you prefer more or less compression.

 
## Configuration and Code

Since Flare has already developed the operators for the federated
statistics computing, we will only need to provide the followings
* config_fed_server.json (server side controller configuration)
* config_client_server.json (client side executor configuration)
* local statistics calculator

The same configuration can be achieved via FLARE Job recipe API, but lets looks first

### server side configuration

The server side configuration specifies what the overall statistics metrics one like to calculate: 
such as stddev, histogram, how many bins in used for historgram for each feature, quantiles to be calculated or not. 
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
 
In above configuration, We ask the FLARE to calculate the following statistic
statistics: "count", "mean", "sum", "stddev", "histogram" and "Age". Each statistic may have its own configuration.
For example, Histogram statistic, we specify feature "Age" needs 5 bins and histogram range is within [0, 120), while for
all other features ("*" indicate default feature), the bin is 10, range is not specified, i.e. the ranges will be dynamically estimated.



### Client side Implementation: Local statistics generator
 
The statistics generator `DFStatistics` implements `Statistics` spec. In current example, the input data in the format of Pandas DataFrame. Although we used csv file, but this can be any
tabular data format that be expressed in pandas dataframe.

```
class DFStatistics(Statistics):
    # rest of code 
```
to calculate the local statistics, we will need to implements few methods
```
    def features(self) -> Dict[str, List[Feature]] -> Dict[str, List[Feature]]:

    def count(self, dataset_name: str, feature_name: str) -> int:
 
    def sum(self, dataset_name: str, feature_name: str) -> float:
 
    def mean(self, dataset_name: str, feature_name: str) -> float:
 
    def stddev(self, dataset_name: str, feature_name: str) -> float:
 
    def variance_with_mean(self, dataset_name: str, feature_name: str, global_mean: float, global_count: float) -> float:
 
    def histogram(self, dataset_name: str, feature_name: str, num_of_bins: int, global_min_value: float, global_max_value: float) -> Histogram:

    def quantiles(self, dataset_name: str, feature_name: str, percentiles: List) -> Dict:

```
since some of features do not provide histogram bin range, we will need to calculate based on local min/max to estimate
the global min/max, and then use the global bin/max as the range for all clients' histogram bin range.

so we need to provide local min/max calculation methods
```
   def max_value(self, dataset_name: str, feature_name: str) -> float:
   def min_value(self, dataset_name: str, feature_name: str) -> float:
```

For Tabular data, FLARE has implemented specification already with ```DFStatisticsCore```. We just need to subClass the ```DFStatisticsCore``` and implement
a few methods
```
    def load_data(self, fl_ctx: FLContext) -> Dict[str, pd.DataFrame]:
```

This method specifies how to load the data into different DataFrame, for example, "train" dataset and "test" dataset, one for each key in the dictionary.

```
   def features(self) -> Dict[str, List[Feature]]:
   
```
The features for each dataset is also important. We are assuming each site has the same features. 

by default ```DFStatisticsCore.features(self) -> Dict[str, List[Feature]]``` implementation assumes the DataFrame has feature names
the method simply get the feature name from DataFrame 

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
Therefore, the DataFrame must have the column names defined. In his example, the dataset has no headers from CVS file, 
we hard-coded the feature names in the init() function.

If you can't derived features, you can overwrite the this method and return list of features for each dataset. 


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

        # the original dataset has no header,
        # we will use the adult.train dataset for site-1, the adult.test dataset for site-2
        # the adult.test dataset has incorrect formatted row at 1st line, we will skip it.
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


### Client side data privacy configuration


to make sure the privacy, for each site want to specify the min_count. For min value, we want to add 15%,
the max value, we like to add 30%.  max_bins_percent:   max number of bins allowed in terms of percent of local data size.

Set this number to avoid number of bins equal or close equal to the data size, which can lead to data leak.
for example: max_bins_percent = 15, number of bins should not more than 15% * data count

```
   min_count = 15
   min_noise_level = 0.1
   max_noise_level = 0.3
   max_bins_percent = 15
```
This privacy control can be expressed in the job recipe at job level. But it can also be applied at organization privacy 
policy level. for example, using configuration file. in [local private policy](../local/privacy.json)


## Job Recipe

* First follow the installation guide and data preparation step to get the data:  [Hello-tabular-stats](../../../hello-world/hello-tabular-stats/README.md)
* Now run the Job with 

```bash
   python job.py
```

## Visualization