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

    START_ROUND = "start_round"
    CURRENT_ROUND = "current_round"
    CONTRIBUTION_ROUND = "contribution_round"
    NUM_ROUNDS = "num_rounds"

    GLOBAL_MODEL = "global_model"

    LOG_DIR = "model_log_dir"
    CKPT_PRELOAD_PATH = "ckpt_preload_path"

    DXO = "DXO"

    PHASE = "_phase_"
    PHASE_INIT = "_init_"
    PHASE_TRAIN = "train"
    PHASE_FINISHED = "_finished_"

    MODEL_WEIGHTS = "_model_weights_"
    AGGREGATION_RESULT = "_aggregation_result"
    AGGREGATION_ACCEPTED = "_aggregation_accepted"
    TRAIN_SHAREABLE = "_train_shareable_"
    TRAINING_RESULT = "_training_result_"

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

    PARTICIPATING_CLIENTS = "_participating_clients_"

    MODEL_OWNER = "_model_owner_"

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
