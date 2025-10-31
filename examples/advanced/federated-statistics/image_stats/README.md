# Calculate Image Histogram with NVIDIA FLARE

Compute the local and global image statistics.
You can also follow the [notebook](image_stats.ipynb) or the following:

## Code Structure
First, get the example code from github:

```
git clone https://github.com/NVIDIA/NVFlare.git
```
Then navigate to the federated-statistics directory:

```
git switch <release branch>
cd examples/advanced/federated-statistics/image_stats

```
``` bash
image_stats
|
|-- client.py             # client local training script
|-- job.py                # job recipe that defines client and server configurations
|-- download_data.py      # download dataset from Kaggle
|-- prepare_data.py       # prepare dataset by splitting them into multiple sites
|-- requirements.txt      # dependencies
├── demo
│   └── visualization.ipynb # Visualization Notebook

```


## Setup NVFLARE
Follow the [Getting Started](https://nvflare.readthedocs.io/en/main/getting_started.html) to set up virtual environment and install NVFLARE

Let's first install required packages.

```
pip install --upgrade pip

cd NVFlare/examples/advanced/federated-statistics/image_stats

pip install -r requirements.txt
```

## Data

We use the dataset from the ["COVID-19 Radiography Database"](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database).
It contains png image files in four different classes: `COVID`, `Lung_Opacity`, `Normal`, and `Viral Pneumonia`.
First, download and extract to `/tmp/nvflare/image_stats/data/`.

```
python download_data.py
```

Next, create the data lists simulating different clients with varying amounts and types of images. 
The downloaded archive contains sub-folders for four different classes: `COVID`, `Lung_Opacity`, `Normal`, and `Viral Pneumonia`.
Here we assume each class of image corresponds to a different site.

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


## Client Side Code: Local statistics generator

The local statistics generator implements the `Statistics` spec.

Besides loading data methods, the class mainly implements a few functions

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
An additional optional failure_count method is used if you would like to ensure data privacy against the effective count (count - failure_count)

```
    def failure_count(self,
              dataset_name: str,
              feature_name: str) -> int:
        return self.failure_images
```

If you would like to see failure_count as one statistic in reporting, you will need to add "failure_count" to the statistic_config
arguments for the statistics controller.

```
class ImageStatistics(Statistics):
    def __init__(self, data_root: str = "/tmp/nvflare/image_stats/data", data_list_key: str = "data"):
        <skip code>
        
    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        self.client_name = fl_ctx.get_identity_name()
        self.loader = LoadImage(image_only=True)
        self.loader.register(ITKReader())
        self._load_data_list(self.client_name, fl_ctx)

    def _load_data_list(self, client_name, fl_ctx: FLContext) -> bool:
        # load the data set json file for each training and test dataset
         <skip code>
         
    def features(self) -> Dict[str, List[Feature]]:
        return {"train": [Feature("intensity", DataType.FLOAT)]}

    def count(self, dataset_name: str, feature_name: str) -> int:
        image_paths = self.data_list[dataset_name]
        return len(image_paths)

    def failure_count(self, dataset_name: str, feature_name: str) -> int:
        return self.failure_images

    def histogram(
        self, dataset_name: str, feature_name: str, num_of_bins: int, global_min_value: float, global_max_value: float
    ) -> Histogram:
        <skip code>
        for each filepath in the site's image list
        load image and generate histogram
        
```

## Server Code
The server aggregation has already been implemented in Statistics Controller

## Job Recipe

For histogram, we specify the histogram range for all features ("*") to be [0,256) and bins = 255.

**Privacy Filter**
You can experiment with different privacy policies. For example, what if you set
```min_count``` to 1000 to avoid leaking information

```
    statistic_configs = {"count": {}, "histogram": {"*": {"bins": 20, "range": [0, 256]}}}
    # define local stats generator
    stats_generator = ImageStatistics(data_root_dir)

    sites = [f"site-{i + 1}" for i in range(n_clients)]
    recipe = FedStatsRecipe(
        name="stats_image",
        stats_output_path=output_path,
        sites=sites,
        statistic_configs=statistic_configs,
        stats_generator=stats_generator,
        min_count = 10
    )

    env = SimEnv(clients=sites, num_threads=n_clients)
    recipe.execute(env=env)


```

The results are stored in workspace "/tmp/nvflare/simulation/stats_image/server/simulate_job"

```
/tmp/nvflare/simulation/stats_image/server/simulate_job/statistics/image_statistics.json
```

## Visualization

```bash
    cp /tmp/nvflare/simulation/stats_image/server/simulate_job/statistics/image_statistics.json demo/.
    cd demo
    jupyter notebook  visualization.ipynb
```
You should see a histogram similar to this: 
![compare all sites' histograms](figs/image_histogram.png)
