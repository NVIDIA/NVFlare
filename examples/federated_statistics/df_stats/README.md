# Federated Statistics of Panda DataFame

## setup NVFLARE
follow the [Quick Start Guide](https://nvflare.readthedocs.io/en/main/quickstart.html) to setup virtual environment and install NVFLARE
```
install required packages.
```
pip install --upgrade pip
pip install -r ./requirements.txt

## 1. prepare POC workspace

```
   nvflare poc --prepare
```
This will create a poc at /tmp/nvflare/poc with n = 2 clients.


## 2. Download the example data
In this example, we are using UCI (University of California, Irwin) [adult dataset](https://archive.ics.uci.edu/ml/datasets/adult)
We split the training data and test data into two clients (one client with training, and another client with test dataset) 

```applicaiton.conf

fed_stats {
data {
features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
             "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"]

        clients { # inherit common properties
            site-1 {
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
                filename = "data.csv"
                skiprows = [] # Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file.
            }
            site-2 {
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
                filename = "data.csv"
                skiprows = [0] # Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file.
            }
        }
    }

}

```
The application.conf format is [pyhocon](https://github.com/chimpler/pyhocon) format.

Now we use data utility to download UCI datasets to separate client package directory in POC workspace

<poc workspace>/site-1/data.csv
<poc workspace>/site-2/data.csv


```
python3 data_utils.py  -h

usage: data_utils.py [-h] [--prepare-data]

fed_stats parser

optional arguments:
  -h, --help            show this help message and exit
  --prepare-data        prepare data based on configuration

```

```
python3 data_utils.py  --prepare-data

prepare data for poc workspace:/tmp/nvflare/poc
remove existing data at /tmp/nvflare/poc/site-1/data.csv
wget download to /tmp/nvflare/poc/site-1/data.csv
100% [..........................................................................] 3974305 / 3974305remove existing data at /tmp/nvflare/poc/site-2/data.csv
wget download to /tmp/nvflare/poc/site-2/data.csv
100% [..........................................................................] 2003153 / 2003153done with prepare data

```
If your poc_workspace is in a different location, use the following command

```
export NVFLARE_POC_WORKSPACE=<new poc workspace location>
```
then repeat above


## 3. Compute the local and global statistics for each numerical feature

### 3.1 Specify client side configuration

We are using a built-in NVFLARE executor,  

```
 "executor": {
        "id": "Executor",
        "path": "nvflare.app_common.executors.statistics_executor.StatisticsExecutor",
        "args": {
          "generator_id": "df_stats_generator",
          "min_count" : 10,
          "min_random": 0.1,
          "max_random": 0.3
  },

```
here we specify a minimum of 10 rows (min_count), if the number of rows less than min_count, the job will fail.
In case, user will need to calculate histogram, user will need to provide the min/max value of given feature. 
But instead of directly send the local min/max to server, we introduce some randomness to the min/max value. 

so that estimate min = min * ( 1 - random ); max = max * ( 1 + random )
where random value is within min_random and max_random, in this example ( 0.1 and 0.3) ( data privacy police defined noise level)

The estimated global min and max should be 

estimated global min <  all clients' min value and  global max >  all clients' max value
and global min < global max

We then use this estimate global min and max to dynamic calculate the histogram range for all histogram buckets ( client and global)
if the feature range is not specified. 


Then user need to implemented the local statistics generator, identified as "generator_id"
```
"components": [
    {
    "id": "df_stats_generator",
    "path": "df_stats_generator.DFStatistics",
    "args": {
        "data_path" : "{workspace_dir}/{client_name}/data.csv",
    }
}
```
In this example, Data Frame Statistics generator is DFStatistics and it takes input data_path for each client.  


### 3.2 Specify Server side configuration
 
Here we use the built-in Controller, called GlobalStatistics. Here we selected all the available metrics.  

```
"workflows": [
    {
      "id": "fed_stats_controller",
      "path": "nvflare.app_common.workflows.global_statistics.GlobalStatistics",
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
In above configuration "*" indicate default feature.  Here we specify feature "Age" needs 5 bins and histogram range is within 0.120
for all other features, the bin is 10, range is not specified, i.e. the ranges will be dynamically estimated. 

We also defined a writer, so we writer to output path
    ```
    "components": [
        {
        "id": "stats_writer",
        "path": "nvflare.app_common.statistics.stats_file_persistor.StatsFileWriter",
        "args": {
            "output_path": "statistics/stats.json"
        }
    }
    ```
If you using some class that is special handling to to encode Json, you can registered the JsonEncoder like the following

```
 "components": [
    {
      "id": "stats_writer",
      "path": "nvflare.app_common.statistics.json_stats_file_persistor.JsonStatsFileWriter",
      "args": {
        "output_path": "@workspace_dir@/statistics/adults_stats.json",
        "json_encoder_path": "nvflare.app_common.utils.json_utils.ObjectEncoder"
      }
    }
  ]
```
by default "nvflare.app_common.utils.json_utils.ObjectEncoder" is used if the json_encoder_path is not specified

"@workspace_dir@" is variable that will be replace by real poc workspace directory at runtime


### 3.3 write a local statistics generator 

   The statistics generator implements `Statistics` spec. 

```

class DFStatistics(Statistics):
    # rest of code 

```
### 3.4 Start nvflare in poc mode

```
   nvflare -w <workspace> --start

```
or 
```
   nvflare --start

```
if the default poc workspace is used.

once you have done with above command, you are already login to the NVFLARE console (aka Admin Console) 


### 3.5 Submit job using flare console

Inside the console, submit the job: `submit_job [PWD]/df_statistics` (replace `[PWD]` with your current path)

For a complete list of available flare console commands, see [here](https://nvflare.readthedocs.io/en/main/user_guide/operation.html).

### 3.2 List the submitted job

You should see the server and clients in your first terminal executing the job now.
You can list the running job by using `list_jobs` in the admin console.
Your output should be similar to the following.

```
> list_jobs 
-------------------------------------------------------------------------------------------------==--------------------------------
| JOB ID                               | NAME     | STATUS                       | SUBMIT TIME                                    |
-----------------------------------------------------------------------------------------------------------------------------------
| 10a92352-5459-47d2-8886-b85abf70ddd1 | df_stats | FINISHED:COMPLETED           | 2022-08-05T22:50:40.968771-07:00 | 0:00:29.4493|
-----------------------------------------------------------------------------------------------------------------------------------
```
 
## 4. get the result

If successful, the computed statis can be downloaded using this admin command:
```
download_job [JOB_ID]
```
After download, it will be available in the stated download directory under `[JOB_ID]/workspace/statistics` as  `adult_stats.json`

## 5. Output Format
By default save the result in JSON format. You are free to write another StatsWriter to output in other format.

### JSON FORMAT
The output of the json is like the followings
``` 
{ 
     "metric": {
        "site-1" : {
            "dataset-1": {           
                  "feature-1": metric_value
                  "feature-2": metric_value
                  ...
            },
            "dataset-2": {           
                  "feature-1": metric_value
                  "feature-2": metric_value
                  ...
            } 
        },
        "site-2" : {
            "dataset-1": {           
                "feature-1": metric_value
                "feature-2": metric_value
                ...
            },
            "dataset-2": {           
                "feature-1": metric_value
                "feature-2": metric_value
                ...
            }
        }, 
        ...
        "Global" : {
            "dataset-1": {           
                "feature-1": metric_value
                "feature-2": metric_value
                ...
            },
            "dataset-2": {           
                "feature-1": metric_value
                "feature-2": metric_value
                ...
            }
        }
     }
```

```

class Bin(NamedTuple):
    # The low value of the bucket, inclusive.
    low_value: float

    # The high value of the bucket, exclusive (unless the highValue is positive infinity).
    high_value: float

    # quantile sample count could be fractional
    sample_count: float
 

class HistogramType(IntEnum):
    STANDARD = 0
    QUANTILES = 1

class Histogram(NamedTuple):
    # The type of the histogram. A standard histogram has equal-width buckets.
    # The quantiles type is used for when the histogram message is used to store
    # quantile information (by using equal-count buckets with variable widths).

    # The type of the histogram.
    hist_type: HistogramType

    # A list of buckets in the histogram, sorted from lowest bucket to highest bucket.
    bins: List[Bin]

    # An optional descriptive name of the histogram, to be used for labeling.
    hist_name: Optional[str] = None

Json format of histogram (NamedTuple) is a List of HisgramType (0), bin list and hist_name (null)
    bin_list = [ [lower value_1, high_value_1, sample_count_1 ], ..., [lower value_n, high_value_n, sample_count_n ] ]
    [ 0, bin_list, null]  
