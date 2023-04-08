# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    W&B vs MLFLOW vs. Tensorboad
    DATA --> PARAMETER --> Scalar
    DATA --> PARAMETERS --> Scalars
    DATA --> METRIC --> Scalar
    DATA --> METRICS --> Scalars
    DATA --> TEXT --> TEXT
    DATA --> IMAGE --> IMAGE
    ARTIFACT:type=model --> MODEL

    NOT SUPPORTED
    PLOT --> FIGURE
    ARTIFACT --> TEXT
    DATA --> ? --> Histogram
    ARTIFACT --> DICT
    ARTIFACT:type=dataset --> ARTIFACT

    wandb.log(data) ==> mlflow.log_params(data) ==> writer.add_scalas(data)
    wandb.log(data) ==> mlflow.log_metrics(data) ==> writer.add_scalas(data)
    wandb.log({"examples": images} ==>  mlflow.log_image(image, output_path) ==> writer.add_image('images', image, 0)

    art = wandb.Artifact("my-object-detector", type="model")
    art.add_file("saved_model_weights.pt")
    wandb.log_artifact(art) ==> mlflow.register_model(model_uri, "my-object-detector")  ==> ?

  """
from enum import Enum

ANALYTIC_EVENT_TYPE = "analytix_log_stats"


class Tracker(Enum):
    TORCH_TB = "TORCH_TENSORBOARD"
    MLFLOW = "MLFLOW"
    WANDB = "WEIGHTS_AND_BIASES"


class TrackConst(object):
    TRACKER_KEY = "tracker_key"

    TRACK_KEY = "track_key"
    TRACK_VALUE = "track_value"

    TAG_KEY = "tag_key"
    TAGS_KEY = "tags_key"

    EXP_TAGS_KEY = "tags_key"

    GLOBAL_STEP_KEY = "global_step"
    PATH_KEY = "path"
    DATA_TYPE_KEY = "analytics_data_type"
    KWARGS_KEY = "analytics_kwargs"

    PROJECT_NAME = "project_name"
    PROJECT_TAGS = "project_name"

    EXPERIMENT_NAME = "experiment_name"
    EXPERIMENT_TAG = "experiment_tag"
    INIT_CONFIG = "init_config"
    RUN_TAGS = "run_tags"

    SITE_KEY = "site"
    JOB_ID_KEY = "job_id"
