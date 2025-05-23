# Calculate Image Histogram with NVIDIA FLARE

Compute the local and global image statistics.
You can also follow the [notebook](../image_stats.ipynb) or the following:

## Setup NVFLARE
Follow the [Getting Started](https://nvflare.readthedocs.io/en/main/getting_started.html) to set up virtual environment and install NVFLARE

Let's first install required packages.

```
pip install --upgrade pip

cd NVFlare/examples/advanced/federated-statistics/image_stats

pip install -r requirements.txt
```

## 1. Download the example data

As an example, we use the dataset from the ["COVID-19 Radiography Database"](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database).
it contains png image files in four different classes: `COVID`, `Lung_Opacity`, `Normal`, and `Viral Pneumonia`.
First, download and extract to `/tmp/nvflare/image_stats/data/`.

Next, create the data lists simulating different clients with varying amounts and types of images. 
The downloaded archive contains sub-folders for four different classes: `COVID`, `Lung_Opacity`, `Normal`, and `Viral Pneumonia`.
Here we assume each class of image corresponds to a different sites.

```shell
prepare_data.sh
```

With this ratio setting, site-3 will have the largest number of images. You should see the following output
```
Created 4 data lists for ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia'].
Saved 3616 entries at /tmp/nvflare/image_stats/data/site-1_COVID.json
Saved 6012 entries at /tmp/nvflare/image_stats/data/site-2_Lung_Opacity.json
Saved 10192 entries at /tmp/nvflare/image_stats/data/site-3_Normal.json
Saved 1345 entries at /tmp/nvflare/image_stats/data/site-4_Viral Pneumonia.json
```
## 2. Run the Federated Statistics Job in Simulator 

## 2.1. Simulator CLI

you can run the job in CLI command, since we have 4 clients, we will set `-n 4` with 4 threads `-t 4`

```
cd NVFlare/examples/advanced/federated-statistics
nvflare simulator image_stats/jobs/image_stats -w /tmp/nvflare/image_stats/workspace -n 4 -t 4
```

The results are stored in workspace "/tmp/nvflare/image_stats/workspace"
```
/tmp/nvflare/image_stats/workspace/server/simulate_job/statistics/image_histogram.json
```

## 2.2 Visualization

```bash
    cp /tmp/nvflare/image_stats/workspace/server/simulate_job/statistics/image_histogram.json NVFlare/examples/advanced/federated-statistics/image_stats/demo/.
    
    cd NVFlare/examples/advanced/federated-statistics/image_stats/demo
    
    jupyter notebook  visualization.ipynb
```
![compare all sites' histograms](figs/image_histogram.png)
 

## 3. Federated Statistics Configuration

In this section, we will try to dive a bit more onto the configuration and code. 

Since FLARE has already built-in operators, all we need to supply the followings
* config_fed_server.json
* config_fed_client.json
* local statistics generator

### 3.1 Server configuration
Here we ask the statistics_controller (server side operator) to compute & display the following statistics: count and histogram
````
     "statistic_configs": {
          "count": {},
          "histogram": {
            "*": {
              "bins": 255, "range": [0,256]
            }
          }
        },
````
for histogram, we specified the histogram range, for all features ("*"), to be [0,256), and bins = 255.
The writer component
```
  "components": [
    {
      "id": "stats_writer",
      "path": "nvflare.app_common.statistics.json_stats_file_persistor.JsonStatsFileWriter",
      "args": {
        "output_path": "statistics/image_statistics.json",
        "json_encoder_path": "nvflare.app_common.utils.json_utils.ObjectEncoder"
      }
    }
  ]
```
provided by FLARE is going to output the result as JSON format to the <workspace>/statistics/image_statistics.json

### 3.2 Client configuration

On client side, we first specify the client's side operator (StatisticExecutor) and specify the pre-defined task to be "fed_stats"
and generator_id = "local_hist_generator". Where local_hist_generator will be local statistics generator defined by custom code.

```
 "executors": [
    {
      "tasks": [
        "fed_stats"
      ],
      "executor": {
        "id": "Executor",
        "path": "nvflare.app_common.executors.statistics.statistics_executor.StatisticsExecutor",
        "args": {
          "generator_id": "local_hist_generator"
        }
      }
    }
  ],
```
the local stats generator is defined as FLComponent (ImageStatistics)
```
    {
      "id": "local_hist_generator",
      "path": "image_statistics.ImageStatistics",
      "args": {
        "data_root": "/tmp/nvflare/data"
      }
    },
```

In addition, we also specified privacy filter as task_result_filter, in production, this is usually set as privacy policy
on site-level by the company, there is no need to set privacy filter at job level (see later discuss on privacy)

```
  "task_result_filters": [
    {
      "tasks": ["fed_stats"],
      "filters":[
        {
          "name": "StatisticsPrivacyFilter",
          "args": {
            "result_cleanser_ids": [
              "min_count_cleanser"
            ]
          }
        }
      ]
    }
  ],
```


### 3.3 Local statistics generator

```python

class ImageStatistics(Statistics):

    def __init__(
        self,
        data_root: str = "/tmp/nvflare/data",
        data_list_key: str = "data"
    ):

```
The local statistics generator implements `Statistics` spec.

Besides loading data methods, the class mainly implements few functions

```

    def features(self) -> Dict[str, List[Feature]]:
        return {"train": [Feature("density", DataType.FLOAT)]}

    def count(self,
              dataset_name: str,
              feature_name: str) -> int:
        image_paths = self.data_list[dataset_name]
        return len(image_paths)

    def histogram(self, dataset_name: str, feature_name: str, num_of_bins: int, global_min_value: float,
                  global_max_value: float) -> Histogram:
       ...
```
additional optional failure_count methods is used if you would like to make sure the data privacy is against effective
count - failure_count

```
    def failure_count(self,
              dataset_name: str,
              feature_name: str) -> int:
        return self.failure_images
```

If you would like to see failure_count as one statistic in reporting, you will need to add "failure_count" to the statistic_config
arguments for the statistics controller.

## 4. Privacy Policy

In current example, we include one cleanser in the `StatisticsPrivacyFilter`: `min_count_cleanser` 

### Experiments with Privacy Filter
 
You can experiment with different privacy policy, for example, what if you set
```min_count``` to  1600, which is larger than the 1345 images site-4. 

```
  {
      "id": "min_count_cleanser",
      "path": "nvflare.app_common.statistics.min_count_cleanser.MinCountCleanser",
      "args": {
        "min_count": 1600
      }
    }
```
