# Data Frame Federated Statistics 

In this example, we will show how to generate federated statistics for data that be represented as Pandas Data Frame

## setup NVFLARE
follow the [Quick Start Guide](https://nvflare.readthedocs.io/en/main/quickstart.html) to setup virtual environment and install NVFLARE
```
install required packages.
```
pip install --upgrade pip
pip install -r ./requirements.txt

## 1. Prepare data

In this example, we are using UCI (University of California, Irwin) [adult dataset](https://archive.ics.uci.edu/ml/datasets/adult)
The original dataset has already contains "training" and "test" datasets. Here we simply assume that "training" and test data sets are belong to different clients.
so we assigned the training data and test data into two clients.
 
Now we use data utility to download UCI datasets to separate client package directory to /tmp/nvflare/data/ directory

```
python3 data_utils.py  --prepare-data
```
it should showing something like
```
wget download to /tmp/nvflare/data/site-1/data.csv
100% [..........................................................................] 3974305 / 3974305
remove existing data at /tmp/nvflare/data/site-2/data.csv
wget download to /tmp/nvflare/data/site-2/data.csv
100% [..........................................................................] 2003153 / 2003153
done with prepare data

```

## 2. Run job in FL Simulator

With FL simulator, we can just run the example with CLI command 

```
nvflare simulator $NVFLARE_HOME/examples/federated_statistics/df_stats/df_stats_job -w /tmp/nvflare -n 2 -t 2
```

The results are stored in workspace "/tmp/nvflare"
```
/tmp/nvflare/simulate_job/statistics/adults_stats.json
```

## 3. Visualization
   with json format, the data can be easily visualized via pandas dataframe and plots. 
   A visualization utility tools are showed in show_stats.py in visualization directory
   You can run jupyter notebook visualization.ipynb

   assuming NVFLARE_HOME env variable point tp the github project location (NVFlare) which contains current example. 

```bash
    cp /tmp/nvflare/simulate_job/statistics/adults_stats.json $NVFLARE_HOME/examples/federated_statistics/df_stats/demo/.
    
    cd $NVFLARE_HOME/examples/federated_statistics/df_stats/demo
    
    jupyter notebook  visualization.ipynb
```
you should be able to get the visualization similar to the followings

![stats](demo/df_stats.png) and ![histogram plot](demo/hist_plot.png)


## 4. Run Example using POC command

Alternative way to run job is using POC mode

### 4.1 Prepare POC Workspace

```
   nvflare poc --prepare 
```
This will create a poc at /tmp/nvflare/poc with n = 2 clients.

If your poc_workspace is in a different location, use the following command

```
export NVFLARE_POC_WORKSPACE=<new poc workspace location>
```
then repeat above

### 4.2 Start nvflare in POC mode

```
nvflare poc --start
```
once you have done with above command, you are already login to the NVFLARE console (aka Admin Console)
if you prefer to have NVFLARE Console in separate terminal, you can do

```
nvflare poc --start ex admin
```
Then open a separate terminal to start the NVFLARE console
```
nvflare poc --start -p admin
```

### 4.3 Submit job

Inside the console, submit the job:
```
submit_job federated_statistics/df_stats/df_stats_job
```

### 4.4 List the submitted job

You should see the server and clients in your first terminal executing the job now.
You can list the running job by using `list_jobs` in the admin console.
Your output should be similar to the following.

```
> list_jobs 
-------------------------------------------------------------------------------------------------==--------------------------------
| JOB ID                               | NAME     | STATUS                       | SUBMIT TIME                                    |
-----------------------------------------------------------------------------------------------------------------------------------
| 10a92352-5459-47d2-8886-b85abf70ddd1 | df_stats_job | FINISHED:COMPLETED           | 2022-08-05T22:50:40.968771-07:00 | 0:00:29.4493|
-----------------------------------------------------------------------------------------------------------------------------------
```

### 4.5 Get the result

If successful, the computed statis can be downloaded using this admin command:
```
download_job [JOB_ID]
```
After download, it will be available in the stated download directory under `[JOB_ID]/workspace/statistics` as  `adult_stats.json`
then go to section [6. Visualization]

## 5. Configuration and Code

Since Flare has already developed the operators (server side controller and client side executor) for the federated
statistics computing, we will only need to provide the followings
* config_fed_server.json ( server side controller configuration)
* config_client_server.json ( client side executor configuration)
* local statistics calculator

### 5.1 server side configuration

```
"workflows": [
    {
      "id": "fed_stats_controller",
      "path": "nvflare.app_common.workflows.statistics_controller.StatisticsController",
      "args": {
        "metric_configs": {
          "count": {},
          "mean": {},
          "sum": {},
          "stddev": {},
          "histogram": { "*": {"bins": 10 },
                         "Age": {"bins": 5, "range":[0,120]}
                       }
        },
        "writer_id": "stats_writer"
      }
    }
  ],
```
In above configuration, `StatisticsController` is controller. We ask the controller to calculate the following statistic
metrics: "count", "mean", "sum", "stddev", "histogram" and "Age". Each metric may have its own configuration.
For example, Histogram metric, we specify feature "Age" needs 5 bins and histogram range is within [0, 120), while for
all other features ("*" indicate default feature), the bin is 10, range is not specified, i.e. the ranges will be dynamically estimated.

The StatisticController also takes writer_id = "stats_writer", the writer_id identify the output writer component, defined as

```
 "components": [
    {
      "id": "stats_writer",
      "path": "nvflare.app_common.statistics.json_stats_file_persistor.JsonStatsFileWriter",
      "args": {
        "output_path": "statistics/adults_stats.json",
        "json_encoder_path": "nvflare.app_common.utils.json_utils.ObjectEncoder"
      }
    }
```
This configuration shows a JSON file output writer, the result will be saved to the <job workspace>/"statistics/adults_stats.json",
in FLARE job store.

### 5.2 client side configuration
 
First, we specify the built-in client side executor: `StatisticsExecutor`, which takes a local stats generator Id

```
 "executor": {
        "id": "Executor",
        "path": "nvflare.app_common.executors.statistics_executor.StatisticsExecutor",
        "args": {
          "generator_id": "df_stats_generator",
  },

```

The local statistics generator is defined as FLComponent: `DFStatistics` which implement the `Statistics` spec.

```
  "components": [
    {
      "id": "df_stats_generator",
      "path": "df_statistics.DFStatistics",
      "args": {
        "data_path": "data.csv"
      }
    },
   ...
  ]
```

Next, we specify the `task_result_filters`. The task_result_filters are the post-process filter that takes the results
of executor and then apply the filter before sending to server.

In this example, task_result_filters is defined as task privacy filter : `StatisticsPrivacyFilter`
```
  "task_result_filters": [
    {
      "tasks": ["fed_stats"],
      "filters":[
        {
          "name": "StatisticsPrivacyFilter",
          "args": {
            "result_cleanser_ids": [
              "min_count_cleanser",
              "min_max_noise_cleanser",
              "hist_bins_cleanser"
            ]
          }
        }
      ]
    }
  ],
``` 
`StatisticsPrivacyFilter` is using three separate the `MetricPrivacyCleanser`, you can find more details in
[local privacy policy](../local/README.md) and in later discussion on privacy.

The privacy cleansers specify policy can be find in
```
  "components": [
    {
      "id": "df_stats_generator",
      "path": "df_statistics.DFStatistics",
      "args": {
        "data_path": "data.csv"
      }
    },
    {
      "id": "min_max_cleanser",
      "path": "nvflare.app_common.statistics.min_max_cleanser.AddNoiseToMinMax",
      "args": {
        "min_noise_level": 0.1,
        "max_noise_level": 0.3
      }
    },
    {
      "id": "hist_bins_cleanser",
      "path": "nvflare.app_common.statistics.histogram_bins_cleanser.HistogramBinsCleanser",
      "args": {
        "max_bins_percent": 10
      }
    },
    {
      "id": "min_count_cleanser",
      "path": "nvflare.app_common.statistics.min_count_cleanser.MinCountCleanser",
      "args": {
        "min_count": 10
      }
    }
  ]

```
Or in [local private policy](../local/privacy.json)

### 5.3 Local statistics generator

The statistics generator `DFStatistics` implements `Statistics` spec.
In current example, the input data in the format of Pandas DataFrame. Although we used csv file, but this can be any
tabular data format that be expressed in pandas dataframe.

```
class DFStatistics(Statistics):
    # rest of code 
```
to calculate the local metrics, we will need to implements few methods
```
    def features(self) -> Dict[str, List[Feature]] -> Dict[str, List[Feature]]:

    def count(self, dataset_name: str, feature_name: str) -> int:
 
    def sum(self, dataset_name: str, feature_name: str) -> float:
 
    def mean(self, dataset_name: str, feature_name: str) -> float:
 
    def stddev(self, dataset_name: str, feature_name: str) -> float:
 
    def variance_with_mean(self, dataset_name: str, feature_name: str, global_mean: float, global_count: float) -> float:
 
    def histogram(self, dataset_name: str, feature_name: str, num_of_bins: int, global_min_value: float, global_max_value: float) -> Histogram:

```
since some of features do not provide histogram bin range, we will need to calculate based on local min/max to estimate
the global min/max, and then use the global bin/max as the range for all clients' histogram bin range.

so we need to provide local min/max calculation methods
```
   def max_value(self, dataset_name: str, feature_name: str) -> float:
   def min_value(self, dataset_name: str, feature_name: str) -> float:
```


## 6. Privacy Policy

There are different ways to set privacy filter depending the use cases

### 6.1 Set Privacy Policy as researcher

one can specify the "task_result_filters" config_fed_client.json to specify
the privacy control.  This is useful when you develop these filters

### 6.2 setup site privacy policy as org admin

Once the company decides to instrument certain privacy policy independent of individual
job, one can copy the local directory privacy.json content to clients' local privacy.json ( merge not overwrite).
in this example, since we only has one app, we can simply copy the private.json from local directory to

<poc-workspace>/site-1/local/privacy.json
<poc-workspace>/site-2/local/privacy.json

we need to remove the same filters from the job definition in config_fed_client.json
by simply set the "task_result_filters" to empty list to avoid **double filtering**
```
"task_result_filters": []
```
### 6.3 job filter vis filters in private.json filters

privacy filters are defined within a privacy scope.
If a job's privacy scope is defined or has default scope, then the scope’s filters (if any) are applied
before the job-specified filters (if any). This rule is enforced during task execution time.

With such rules, if we have both task result filters and privacy scoped filters, we need to understand
that the privacy filters will be applied first, then job filters. 
