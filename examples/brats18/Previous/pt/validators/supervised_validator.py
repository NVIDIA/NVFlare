# Copyright (c) 2021, NVIDIA CORPORATION.
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

import os

import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants


class SupervisedValidator(Executor):
    def __init__(
        self,
        train_config_filename,
        validate_task_name=AppConstants.TASK_VALIDATION,
    ):
        """Simple Supervised Validator.

        Args:
            train_config_filename: directory of config file.
            validate_task_name: name of the task to validate the model.

        Returns:
            a Shareable with the validation metrics.
        """
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point

        self.train_config_filename = train_config_filename
        self._validate_task_name = validate_task_name

    def _initialize_validator(self, fl_ctx: FLContext):
        # when the run starts, this is where the actual settings get initialized for trainer

        # Epoch counter
        self.epoch_of_start_time = 0

        # Set the paths according to fl_ctx
        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        app_config_dir = ws.get_app_config_dir(fl_ctx.get_run_number())

        train_config_file_path = os.path.join(app_config_dir, self.train_config_filename)

        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized with args: \n {fl_args}",
        )

        # Set the training-related contexts
        self._validation_config(fl_ctx, train_config_file_path=train_config_file_path)

    def _validation_config(self, fl_ctx: FLContext, train_config_file_path: str):
        """monai traning configuration
        Customized to invididual tasks
        Needed for further training and validation:
        self.model
        self.device
        self.transform_post
        self.train_loader
        self.valid_loader
        self.inferer
        self.valid_metric
        """
        pass

    def _terminate_executor(self):
        # collect threads, close files here
        pass

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        # the start and end of a run - only happen once
        if event_type == EventType.START_RUN:
            try:
                self._initialize_validator(fl_ctx)
            except BaseException as e:
                error_msg = f"Exception in _initialize_validator: {e}"
                self.log_exception(fl_ctx, error_msg)
                self.system_panic(error_msg, fl_ctx)
        elif event_type == EventType.END_RUN:
            self._terminate_executor()

    def local_valid(self, valid_loader, abort_signal: Signal) -> dict:
        """
        Return a Dictionary that may contain multiple validation metrics.
        """
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for i, (inputs, labels) in enumerate(valid_loader):
                if abort_signal.triggered:
                    return None
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, pred_label = torch.max(outputs.data, 1)

                total += inputs.data.size()[0]
                correct += (pred_label == labels.data).sum().item()
            metric = correct / float(total)
        return {"acc": metric}

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        if task_name == self._validate_task_name:
            # get task information
            self.log_info(fl_ctx, f"Task name: {task_name}")
            self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

            # update local model weights with received weights
            dxo = from_shareable(shareable)
            global_weights = dxo.data

            # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
            local_var_dict = self.model.state_dict()
            model_keys = global_weights.keys()
            for var_name in local_var_dict:
                if var_name in model_keys:
                    weights = global_weights[var_name]
                    try:
                        # update the local dict
                        local_var_dict[var_name] = torch.as_tensor(np.reshape(weights, local_var_dict[var_name].shape))
                    except Exception as e:
                        raise ValueError("Convert weight from {} failed with error: {}".format(var_name, str(e)))
            self.model.load_state_dict(local_var_dict)

            # perform valid
            train_metric_dict = self.local_valid(self.train_loader, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            for k in train_metric_dict.keys():
                self.log_info(fl_ctx, f"training {k}: {train_metric_dict[k]:.4f}")

            val_metric_dict = self.local_valid(self.valid_loader, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            for k in val_metric_dict.keys():
                self.log_info(fl_ctx, f"validation {k}: {val_metric_dict[k]:.4f}")

            self.log_info(fl_ctx, "Evaluation finished. Returning shareable")

            val_results = {
                "train_metrics": train_metric_dict,
                "val_metrics": val_metric_dict,
            }

            metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
            return metric_dxo.to_shareable()

        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)
