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

import json
import os
import traceback
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
from data import DataManager
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter
from trainer import ConDistTrainer
from utils.get_model import get_model
from utils.model_weights import extract_weights, load_weights
from validator import Validator

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType


class ConDistLearner(Learner):
    def __init__(
        self,
        task_config: str,
        data_config: str,
        aggregation_steps: int,
        method: Literal["ConDist"] = "ConDist",
        seed: Optional[int] = None,
        max_retry: int = 1,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
    ):
        super().__init__()

        self.task_config = task_config
        self.data_config = data_config
        self.aggregation_steps = aggregation_steps

        self._method = method
        self._seed = seed
        self._max_retry = max_retry

        self.train_task_name = train_task_name
        self.submit_model_task_name = submit_model_task_name

    def initialize(self, parts: Dict, fl_ctx: FLContext) -> None:
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)

        # Load configurations
        prefix = Path(self.app_root)
        with open(prefix / self.task_config) as f:
            task_config = json.load(f)

        with open(prefix / self.data_config) as f:
            data_config = json.load(f)

        # Initialize variables
        self.key_metric = "val_meandice"
        self.best_metric = -np.inf
        self.best_model_path = "models/best_model.pt"
        self.last_model_path = "models/last.pt"

        # Create data manager
        self.dm = DataManager(self.app_root, data_config)

        # Create model
        self.model = get_model(task_config["model"])

        # Configure trainer & validator
        if self._method == "ConDist":
            self.trainer = ConDistTrainer(task_config)
        self.validator = Validator(task_config)

        # Create logger
        self.tb_logger = SummaryWriter(log_dir=prefix / "logs")

    def train(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # Log training info
        num_rounds = data.get_header(AppConstants.NUM_ROUNDS)
        current_round = data.get_header(AppConstants.CURRENT_ROUND)
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{num_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # Make a copy of model weight for weight diff calculation
        dxo = from_shareable(data)
        global_weights = dxo.data

        # Create dataset & data loader (if necessary)
        if self.dm.get_data_loader("train") is None:
            self.dm.setup("train")
        if self.dm.get_data_loader("validate") is None:
            self.dm.setup("validate")

        # Run training
        for i in range(self._max_retry + 1):
            try:
                self.trainer.run(
                    self.model,
                    self.dm.get_data_loader("train"),
                    num_steps=self.aggregation_steps,
                    logger=self.tb_logger,
                )
                break
            except Exception as e:
                if i < self._max_retry:
                    self.log_warning(fl_ctx, f"Someting wrong in training, retrying ({i+1}/{self._max_retry}).")
                    # Restore trainer states to the beginning of the round
                    if os.path.exists(self.last_model_path):
                        self.trainer.load_checkpoint(self.last_model_path, self.model)
                        load_weights(self.model, global_weights)
                        self.model = self.model.to("cuda:0")
                    # Reset dataset & dataloader
                    self.dm._data_loader["train"] = None
                    self.dm._dataset["train"] = None
                    self.dm.setup("train")
                else:
                    raise RuntimeError(traceback.format_exc())

        # Run validation
        for i in range(self._max_retry + 1):
            try:
                metrics = self.validator.run(self.model, self.dm.get_data_loader("validate"))
                break
            except Exception as e:
                if i < self._max_retry:
                    self.log_warning(fl_ctx, f"Someting wrong in training, retrying ({i+1}/{self._max_retry}).")
                    # Reset dataset & dataloader
                    self.dm._data_loader["validate"] = None
                    self.dm._dataset["validate"] = None
                    self.dm.setup("validate")
                else:
                    raise RuntimeError(traceback.format_exc())

        # Log validation results
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        for m, v in metrics.items():
            table.add_row([m, v])
            self.tb_logger.add_scalar(m, v, current_round)
        self.log_info(fl_ctx, str(table))

        # Save checkpoint if necessary
        if self.best_metric < metrics[self.key_metric]:
            self.best_metric = metrics[self.key_metric]
            self.trainer.save_checkpoint(self.best_model_path, self.model)
        self.trainer.save_checkpoint(self.last_model_path, self.model)

        # Calculate weight diff
        local_weights = extract_weights(self.model)
        weight_diff = {}
        for var_name in local_weights:
            weight_diff[var_name] = local_weights[var_name] - global_weights[var_name]
            if np.any(np.isnan(weight_diff[var_name])):
                self.system_panic(f"{var_name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # Create DXO and return
        dxo = DXO(
            data_kind=DataKind.WEIGHT_DIFF,
            data=weight_diff,
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self.aggregation_steps},
        )
        return dxo.to_shareable()

    def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) -> Shareable:
        if model_name == ModelName.BEST_MODEL:
            model_data = None
            try:
                model_data = torch.load(self.best_model_path, map_location="cpu")
                self.log_info(fl_ctx, f"Load best model from {self.best_model_path}")
            except Exception as e:
                self.log_error(fl_ctx, f"Unable to load best model: {e}")

            if model_data:
                data = {}
                for var_name in model_data["model"]:
                    data[var_name] = model_data[var_name].numpy()
                dxo = DXO(data_kind=DataKind.WEIGHTS, data=data)
                return dxo.to_shareable()
            else:
                self.log_error(fl_ctx, f"best local model not available at {self.best_model_path}")
                return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
        else:
            self.log_error(fl_ctx, f"Unknown model_type {model_name}")
            return make_reply(ReturnCode.BAD_TASK_DATA)

    def validate(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # 1. Extract data from shareable
        model_owner = data.get_header(AppConstants.MODEL_OWNER, "global_model")
        validate_type = data.get_header(AppConstants.VALIDATE_TYPE)

        # 2. Prepare dataset
        phase = None
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            phase = "validate"
        elif validate_type == ValidateType.MODEL_VALIDATE:
            phase = "test"

        if self.dm.get_data_loader(phase) is None:
            self.dm.setup(phase)
        data_loader = self.dm.get_data_loader(phase)

        # 3. Update model weight
        try:
            dxo = from_shareable(data)
        except:
            self.log_error(fl_ctx, "Error when extracting DXO from shareable")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        if not dxo.data_kind == DataKind.WEIGHTS:
            self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS")
            return make_reply(ReturnCode.BAD_TASK_DATA)
        load_weights(self.model, dxo.data)

        # 4. Run validation
        self.model = self.model.to("cuda:0")
        for i in range(self._max_retry + 1):
            try:
                data_loader = self.dm.get_data_loader(phase)
                raw_metrics = self.validator.run(self.model, data_loader)
                break
            except Exception as e:
                if i < self._max_retry:
                    self.log_warning(fl_ctx, f"Error encountered in validation, retrying ({i+1}/{self._max_retry}).")
                    # Cleanup previous dataset & dataloader
                    data_loader = None
                    self.dm._data_loader[phase] = None
                    self.dm._dataset[phase] = None
                    # Recreate dataset & dataloader
                    self.dm.setup(phase)
                    # Assume both the model & validator are correct
                else:
                    raise RuntimeError(traceback.format_exc())

        self.log_info(
            fl_ctx,
            f"Validation metrics of {model_owner}'s model on" f" {fl_ctx.get_identity_name()}'s data: {raw_metrics}",
        )

        # For validation before training, only key metric is needed
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            metrics = {MetaKey.INITIAL_METRICS: raw_metrics[self.key_metric]}
            # Save as best model
            if self.best_metric < raw_metrics[self.key_metric]:
                self.best_metric = raw_metrics[self.key_metric]
                self.trainer.save_checkpoint(self.best_model_path, self.model)
        else:
            metrics = raw_metrics

        # 5. Return results
        dxo = DXO(data_kind=DataKind.METRICS, data=metrics)
        return dxo.to_shareable()

    def finalize(self, fl_ctx: FLContext):
        self.dm.teardown()
        self.tb_logger.close()
