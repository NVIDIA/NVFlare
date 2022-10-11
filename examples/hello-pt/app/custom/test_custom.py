# Copyright (c) 2022, NVIDIA CORPORATION.
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

from unittest.mock import patch

from cifar10trainer import Cifar10Trainer
from cifar10validator import Cifar10Validator

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal


class TestCifar10Trainer:
    # mock unneeded method
    @patch.object(Cifar10Trainer, "_save_local_model")
    @patch.object(Cifar10Trainer, "_load_local_model")
    def test_execute(self, mock_save, mock_load):
        train_task_name = "train"
        trainer = Cifar10Trainer(train_task_name=train_task_name, epochs=1)
        # just take first batch
        trainer._train_loader = [iter(trainer._train_loader).next()]

        dxo = DXO(data_kind=DataKind.WEIGHTS, data=trainer.model.state_dict())
        result = trainer.execute(
            train_task_name, shareable=dxo.to_shareable(), fl_ctx=FLContext(), abort_signal=Signal()
        )
        assert result.get_return_code() == ReturnCode.OK


class TestCifar10Validator:
    def test_execute(self):
        validate_task_name = "validate"
        validator = Cifar10Validator(validate_task_name=validate_task_name)
        # just take first batch
        validator._test_loader = [iter(validator._test_loader).next()]

        dxo = DXO(data_kind=DataKind.WEIGHTS, data=validator.model.state_dict())
        result = validator.execute(
            validate_task_name, shareable=dxo.to_shareable(), fl_ctx=FLContext(), abort_signal=Signal()
        )
        assert result.get_return_code() == ReturnCode.OK
