# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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


class ExecutorTasks:

    TRAIN = "train"
    VALIDATE = "validate"
    CROSS_VALIDATION = "__cross_validation"
    SUBMIT_BEST = "__submit_best"
    REPORT_STATUS = "report_status"


class AppConstants(object):

    CONFIG_PATH = "config_path"
    MODEL_NETWORK = "model_network"
    MULTI_GPU = "multi_gpu"
    TRAIN_CONTEXT = "train_context"
    DEVICE = "device"
    MODEL_NAME = "model_name"
    MODEL_URL = "model_url"
    START_ROUND = "start_round"
    CURRENT_ROUND = "current_round"
    CONTRIBUTION_ROUND = "contribution_round"
    CONTRIBUTION_CLIENT = "contribution_client"
    NUM_ROUNDS = "num_rounds"
    WAIT_AFTER_MIN_CLIENTS = "wait_after_min_clients"

    NUM_TOTAL_STEPS = "num_total_steps"  # TOTAL_STEPS
    NUM_EPOCHS_CURRENT_ROUND = "num_epochs_current_round"  # CURRENT_EPOCHS
    NUM_TOTAL_EPOCHS = "num_total_epochs"  # LOCAL_EPOCHS
    LOCAL_EPOCHS = "local_epochs"

    IS_FIRST_ROUND = "is_first_round"
    MY_RANK = "my_rank"
    INITIAL_LEARNING_RATE = "initial_learning_rate"
    CURRENT_LEARNING_RATE = "current_learning_rate"
    NUMBER_OF_GPUS = "number_of_gpus"
    META_COOKIE = "cookie"
    META_DATA = "meta_data"
    GLOBAL_MODEL = "global_model"

    IS_BEST = "is_best"
    FAILURE = "failure"

    LOG_DIR = "model_log_dir"
    CKPT_PRELOAD_PATH = "ckpt_preload_path"

    DXO = "DXO"

    PHASE = "_phase_"
    PHASE_INIT = "_init_"
    PHASE_TRAIN = "train"
    PHASE_MODEL_VALIDATION = "model_validation"
    PHASE_FINISHED = "_finished_"

    STATUS_WAIT = "_wait_"
    STATUS_DONE = "_done_"
    STATUS_TRAINING = "_training_"
    STATUS_IDLE = "_idle_"

    MODEL_LOAD_PATH = "_model_load_path"
    MODEL_SAVE_PATH = "_model_save_path"
    DEFAULT_MODEL_DIR = "models"

    ROUND = "_round_"
    MODEL_WEIGHTS = "_model_weights_"
    AGGREGATION_RESULT = "_aggregation_result"
    AGGREGATION_TRIGGERED = "_aggregation_triggered"
    AGGREGATION_ACCEPTED = "_aggregation_accepted"
    TRAIN_SHAREABLE = "_train_shareable_"
    TRAINING_RESULT = "_training_result_"

    SUBMIT_MODEL_FAILURE_REASON = "_submit_model_failure_reason"
    CROSS_VAL_DIR = "cross_site_val"
    CROSS_VAL_MODEL_DIR_NAME = "model_shareables"
    CROSS_VAL_RESULTS_DIR_NAME = "result_shareables"
    CROSS_VAL_MODEL_PATH = "_cross_val_model_path_"
    CROSS_VAL_RESULTS_PATH = "_cross_val_results_path_"
    RECEIVED_MODEL = "_receive_model_"
    RECEIVED_MODEL_OWNER = "_receive_model_owner_"
    MODEL_TO_VALIDATE = "_model_to_validate_"
    DATA_CLIENT = "_data_client_"
    VALIDATION_RESULT = "_validation_result_"

    TASK_SUBMIT_MODEL = "submit_model"
    TASK_VALIDATION = "validate"

    CROSS_VAL_SERVER_MODEL = "_cross_val_server_model_"
    CROSS_VAL_CLIENT_MODEL = "_cross_val_client_model_"
    PARTICIPATING_CLIENTS = "_particpating_clients_"

    MODEL_OWNER = "_model_owner_"

    DEFAULT_FORMATTER_ID = "formatter"
    DEFAULT_MODEL_LOCATOR_ID = "model_locator"

    TASK_END_RUN = "_end_run_"
    TASK_TRAIN = "train"

    DEFAULT_AGGREGATOR_ID = "aggregator"
    DEFAULT_PERSISTOR_ID = "persistor"
    DEFAULT_SHAREABLE_GENERATOR_ID = "shareable_generator"

    SUBMIT_MODEL_NAME = "submit_model_name"
    VALIDATE_TYPE = "_validate_type"


class EnvironmentKey(object):

    CHECKPOINT_DIR = "APP_CKPT_DIR"
    CHECKPOINT_FILE_NAME = "APP_CKPT"


class DefaultCheckpointFileName(object):

    GLOBAL_MODEL = "FL_global_model.pt"
    BEST_GLOBAL_MODEL = "best_FL_global_model.pt"


class ModelName(object):

    BEST_MODEL = "best_model"
    FINAL_MODEL = "final_model"


class ModelFormat(object):

    PT_CHECKPOINT = "pt_checkpoint"
    TORCH_SCRIPT = "torch_script"
    PT_ONNX = "pt_onnx"
    TF_CHECKPOINT = "tf_checkpoint"
    KERAS = "keras_model"


class ValidateType(object):

    BEFORE_TRAIN_VALIDATE = "before_train_validate"
    MODEL_VALIDATE = "model_validate"


class AlgorithmConstants(object):

    SCAFFOLD_CTRL_DIFF = "scaffold_c_diff"
    SCAFFOLD_CTRL_GLOBAL = "scaffold_c_global"
    SCAFFOLD_CTRL_AGGREGATOR_ID = "scaffold_ctrl_aggregator"


class StatisticsConstants(AppConstants):
    STATS_COUNT = "count"
    STATS_MEAN = "mean"
    STATS_SUM = "sum"
    STATS_VAR = "var"
    STATS_STDDEV = "stddev"
    STATS_HISTOGRAM = "histogram"
    STATS_MAX = "max"
    STATS_MIN = "min"
    STATS_FEATURES = "stats_features"

    STATS_GLOBAL_MEAN = "global_mean"
    STATS_GLOBAL_COUNT = "global_count"
    STATS_BINS = "bins"
    STATS_BIN_RANGE = "range"
    STATS_TARGET_METRICS = "metrics"

    FED_STATS_TASK = "fed_stats"
    METRIC_TASK_KEY = "fed_stats_metric"
    STATS_1st_METRICS = "fed_stats_1st_metric"
    STATS_2nd_METRICS = "fed_stats_2nd_metric"

    GLOBAL = "Global"

    ordered_metrics = {
        STATS_1st_METRICS: [STATS_COUNT, STATS_SUM, STATS_MEAN, STATS_MIN, STATS_MAX],
        STATS_2nd_METRICS: [STATS_HISTOGRAM, STATS_VAR, STATS_STDDEV],
    }
