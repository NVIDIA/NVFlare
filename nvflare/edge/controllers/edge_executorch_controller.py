# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import base64
from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.edge.aggregators.edge_json_accumulator import EdgeJsonAccumulator
from nvflare.edge.constants import MsgKey
from nvflare.edge.model_protocol import ModelBufferType, ModelEncoding
from nvflare.edge.model_protocol import ModelExchangeFormat as MEF
from nvflare.edge.model_protocol import ModelNativeFormat
from nvflare.edge.models.model import Cifar10Net, TrainingNet, XorNet, export_model
from nvflare.security.logging import secure_format_exception

tb_writer = SummaryWriter()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EdgeExecutorchController(Controller):
    def __init__(
        self,
        num_rounds: int,
        task_name: str,
        input_shape: List,
        output_shape: List,
    ):
        super().__init__()
        self.task_name = task_name
        if task_name == "cifar10":
            self.model = TrainingNet(Cifar10Net())
        elif task_name == "xor":
            self.model = TrainingNet(XorNet())
        else:
            # not supported error
            raise ValueError(f"Task {task_name} is not supported,supported [`cifar10`, `xor`]")
        self.model.to(DEVICE)
        self.num_rounds = num_rounds
        self.current_round = None
        self.aggregator = None
        self.input_shape = input_shape
        self.output_shape = output_shape

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.log_info(fl_ctx, "Initializing ExecuTorch mobile workflow.")
        self.aggregator = EdgeJsonAccumulator(aggr_key="data")

        # initialize global model
        fl_ctx.set_prop(AppConstants.START_ROUND, 1, private=True, sticky=True)
        fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self.num_rounds, private=True, sticky=False)

    def stop_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Stopping ExecuTorch mobile workflow.")

    def _tensor_from_json(self, tensor_data: Dict[str, Any], divide_factor: int) -> Dict[str, Tensor]:
        """Convert JSON tensor data to PyTorch tensors."""
        grad_dict = {}
        for key, value in tensor_data.items():
            tensor = torch.Tensor(value["data"]).reshape(value["sizes"])
            grad_dict[key] = tensor / divide_factor
        return grad_dict

    def _update_model(self, aggregated_grads: Dict[str, Tensor]) -> None:
        """Update model weights using aggregated gradients."""
        for key, param in self.model.state_dict().items():
            if key in aggregated_grads:
                self.model.state_dict()[key] += aggregated_grads[key].to(DEVICE)

    def _eval_model(self) -> float:
        if self.task_name == "xor":
            # XOR task
            test_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
            test_labels = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
            test_data, test_labels = test_data.to(DEVICE), test_labels.to(DEVICE)
            self.model.to(DEVICE)
            with torch.no_grad():
                pred = self.model(test_data)
                # calculate mean square error
                mse = torch.nn.functional.mse_loss(pred, test_labels)
                # calculate accuracy
                pred_binary = torch.round(pred)
                correct = (pred_binary == test_labels).sum().item()
                total = test_labels.size(0)
                acc = 100 * correct / total
            return {"MSE": mse.item(), "ACC": acc}

        elif self.task_name == "cifar10":
            CIFAR10_ROOT = "/tmp/nvflare/dataset/cifar10"
            from torchvision import datasets, transforms

            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
            test_set = datasets.CIFAR10(root=CIFAR10_ROOT, train=False, download=True, transform=transform)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

            self.model.to(DEVICE)
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    # (optional) use GPU to speed things up
                    inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                    # calculate outputs by running images through the network
                    outputs = self.model(inputs)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                acc = 100 * correct // total
            return {"ACC": acc}

    def _export_current_model(self) -> bytes:
        """Export current model in ExecutorTorch format."""
        input_tensor = torch.randn(self.input_shape)
        label_tensor = torch.ones(self.output_shape, dtype=torch.int64)
        model_buffer = export_model(self.model, input_tensor, label_tensor).buffer
        base64_encoded = base64.b64encode(model_buffer).decode("utf-8")
        return base64_encoded

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        try:
            self.log_info(fl_ctx, "Beginning Executorch mobile training phase.")

            fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self.num_rounds, private=True, sticky=False)
            self.fire_event(AppEventType.TRAINING_STARTED, fl_ctx)

            for i in range(self.num_rounds):
                self.current_round = i
                if abort_signal.triggered:
                    return

                self.log_info(fl_ctx, f"Round {self.current_round} started.")
                fl_ctx.set_prop(
                    AppConstants.CURRENT_ROUND,
                    self.current_round,
                    private=True,
                    sticky=True,
                )
                self.fire_event(AppEventType.ROUND_STARTED, fl_ctx)

                # Create task and send global model to clients
                encoded_buffer = self._export_current_model()

                # Compose shareable
                task_data = Shareable()
                task_data[MsgKey.PAYLOAD] = {
                    MEF.MODEL_BUFFER: encoded_buffer,
                    MEF.MODEL_BUFFER_TYPE: ModelBufferType.EXECUTORCH,
                    MEF.MODEL_BUFFER_NATIVE_FORMAT: ModelNativeFormat.BINARY,
                    MEF.MODEL_BUFFER_ENCODING: ModelEncoding.BASE64,
                    MEF.MODEL_VERSION: self.current_round,
                }
                task_data.set_header(AppConstants.CURRENT_ROUND, self.current_round)
                task_data.set_header(AppConstants.NUM_ROUNDS, self.num_rounds)
                task_data.add_cookie(AppConstants.CONTRIBUTION_ROUND, self.current_round)

                train_task = Task(
                    name="train",
                    data=task_data,
                    result_received_cb=self.process_train_result,
                )

                self.broadcast_and_wait(
                    task=train_task,
                    min_responses=1,
                    wait_time_after_min_received=10,
                    fl_ctx=fl_ctx,
                    abort_signal=abort_signal,
                )

                if abort_signal.triggered:
                    return

                self.log_info(fl_ctx, "Start aggregation.")
                self.fire_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)
                aggr_result = self.aggregator.aggregate(fl_ctx)
                fl_ctx.set_prop(
                    AppConstants.AGGREGATION_RESULT,
                    aggr_result,
                    private=True,
                    sticky=False,
                )
                self.fire_event(AppEventType.AFTER_AGGREGATION, fl_ctx)
                self.log_info(fl_ctx, "End aggregation.")

                # reset aggregator
                self.aggregator.reset(fl_ctx)

                # Convert aggregated gradients to PyTorch tensors
                divide_factor = aggr_result["num_devices"]
                aggregated_grads = self._tensor_from_json(aggr_result[MsgKey.RESULT], divide_factor)

                # Update model weights using aggregated gradients
                self._update_model(aggregated_grads)

                # Evaluate the model
                metric = self._eval_model()
                for key, value in metric.items():
                    tb_writer.add_scalar(key, value, i)

                if abort_signal.triggered:
                    return

            self.log_info(fl_ctx, "Finished Mobile Training.")
            # save the final model
            torch.save(self.model.state_dict(), "global_model.pth")

        except Exception as e:
            error_msg = f"Exception in mobile control_flow: {secure_format_exception(e)}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx)

    def process_train_result(self, client_task: ClientTask, fl_ctx: FLContext) -> None:
        result = client_task.result
        client_name = client_task.client.name
        rc = result.get_return_code()

        # Raise errors if bad peer context or execution exception.
        if rc and rc != ReturnCode.OK:
            self.system_panic(
                f"Result from {client_name} is bad, error code: {rc}. "
                f"{self.__class__.__name__} exiting at round {self.current_round}.",
                fl_ctx=fl_ctx,
            )

            return

        accepted = self.aggregator.accept(result, fl_ctx)
        accepted_msg = "ACCEPTED" if accepted else "REJECTED"
        self.log_info(
            fl_ctx,
            f"Contribution from {client_name} {accepted_msg} by the aggregator at round {self.current_round}.",
        )
