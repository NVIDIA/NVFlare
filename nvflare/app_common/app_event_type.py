# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


class AppEventType(object):
    """Defines application events."""

    BEFORE_AGGREGATION = "_before_aggregation"
    END_AGGREGATION = "_end_aggregation"

    SUBMIT_LOCAL_BEST_MODEL = "_submit_local_best_model"
    SERVER_RECEIVE_BEST_MODEL = "_server_receive_best_model"
    RECEIVE_VALIDATION_MODEL = "_receive_validation_model"
    SEND_VALIDATION_RESULTS = "_send_validation_results"
    RECEIVE_VALIDATION_RESULTS = "_receive_validation_results"

    BEFORE_INITIALIZE = "_before_initialize"
    AFTER_INITIALIZE = "_after_initialize"
    BEFORE_TRAIN = "_before_train"
    AFTER_TRAIN = "_after_train"

    BEFORE_SHAREABLE_TO_LEARNABLE = "_before_model_update"
    AFTER_SHAREABLE_TO_LEARNABLE = "_after_model_update"
    BEFORE_LEARNABLE_PERSIST = "_before_save_model"
    AFTER_LEARNABLE_PERSIST = "_after_save_model"
    BEFORE_SEND_BEST_MODEL = "_before_send_best_model"
    AFTER_SEND_BEST_MODEL = "_after_send_best_model"
    LOCAL_BEST_MODEL_AVAILABLE = "_local_best_model_available"
    GLOBAL_BEST_MODEL_AVAILABLE = "_global_best_model_available"
    BEFORE_GET_VALIDATION_MODELS = "_before_get_validation_models"
    AFTER_GET_VALIDATION_MODELS = "_after_get_validation_models"
    SEND_MODEL_FOR_VALIDATION = "_send_model_for_validation"
    BEFORE_VALIDATE_MODEL = "_before_validate_model"
    AFTER_VALIDATE_MODEL = "_after_validate_model"
    BEFORE_SUBMIT_VALIDATION_RESULTS = "_before_submit_validation_results"
    AFTER_SUBMIT_VALIDATION_RESULTS = "_after_submit_validation_results"

    # Events
    ROUND_STARTED = "_round_started"
    ROUND_DONE = "_round_done"
    INITIAL_MODEL_LOADED = "_initial_model_loaded"
    BEFORE_TRAIN_TASK = "_before_train_task"
    RECEIVE_CONTRIBUTION = "_receive_contribution"
    AFTER_CONTRIBUTION_ACCEPT = "_after_contribution_accept"
    AFTER_AGGREGATION = "_after_aggregation"
    BEFORE_CONTRIBUTION_ACCEPT = "_before_contribution_accept"
    GLOBAL_WEIGHTS_UPDATED = "_global_weights_updated"
    TRAINING_STARTED = "_training_started"
    TRAINING_FINISHED = "_training_finished"
    TRAIN_DONE = "_train_done"

    CROSS_VAL_INIT = "_cross_val_init"
    VALIDATION_RESULT_RECEIVED = "_validation_result_received"
    RECEIVE_BEST_MODEL = "_receive_best_model"
