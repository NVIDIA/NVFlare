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
from pt.networks.cifar10_nets import ModerateCNN
from pt.utils.cifar10_dataset import CIFAR10_Idx
from torchvision import datasets, transforms

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants


class CIFAR10Validator(Executor):
    def __init__(
        self,
        dataset_root: str = "./dataset",
        validate_task_name=AppConstants.TASK_VALIDATION,
    ):
        """Simple CIFAR-10 Validator.

        Args:
            dataset_root: directory with CIFAR-10 data.
            validate_task_name: name of the task to validate the model.

        Returns:
            a Shareable with the validation metrics.
        """
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point

        self.dataset_root = dataset_root
        self._validate_task_name = validate_task_name

    def _initialize_validator(self, fl_ctx: FLContext):
        # when the run starts, this is where the actual settings get initialized for trainer

        # Epoch counter
        self.epoch_of_start_time = 0

        # Set the paths according to fl_ctx
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized at \n {self.app_root} \n with args: {fl_args}",
        )

        # Set datalist, here the path and filename are hard-coded, can also be fed as an argument
        site_idx_file_name = os.path.join(self.dataset_root, self.client_id + ".npy")
        self.log_info(fl_ctx, f"IndexList Path: {site_idx_file_name}")
        if os.path.exists(site_idx_file_name):
            self.log_info(fl_ctx, "Loading subset index")
            site_idx = np.load(site_idx_file_name).tolist()
        else:
            self.system_panic(f"No subset index found! File {site_idx_file_name} does not exist!", fl_ctx)
            return
        self.log_info(fl_ctx, f"Client subset size: {len(site_idx)}")

        # set the training-related parameters
        # can be replaced by a config-style block
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = ModerateCNN().to(self.device)

        self.transform_valid = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                ),
            ]
        )

        # Set dataset
        self.train_dataset = CIFAR10_Idx(
            root=self.dataset_root,
            data_idx=site_idx,
            train=True,
            download=True,
            transform=self.transform_valid,
        )
        self.valid_dataset = datasets.CIFAR10(
            root=self.dataset_root,
            train=False,
            download=True,
            transform=self.transform_valid,
        )

        self.train_loader = data.DataLoader(self.train_dataset, batch_size=64, shuffle=True, num_workers=2)

        self.valid_loader = data.DataLoader(self.valid_dataset, batch_size=64, shuffle=False, num_workers=2)

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

    def local_valid(self, valid_loader, abort_signal: Signal):
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
        return metric

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        if task_name == self._validate_task_name:
            # Check abort signal
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            # get round information
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
            train_acc = self.local_valid(self.train_loader, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"training acc: {train_acc:.4f}")

            val_acc = self.local_valid(self.valid_loader, abort_signal)
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(fl_ctx, f"validation acc: {val_acc:.4f}")

            self.log_info(fl_ctx, "Evaluation finished. Returning shareable")

            val_results = {"train_accuracy": train_acc, "val_accuracy": val_acc}

            metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
            return metric_dxo.to_shareable()

        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)
