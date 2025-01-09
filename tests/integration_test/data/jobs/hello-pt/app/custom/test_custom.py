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


from unittest.mock import patch

import pytest
from cifar10trainer import Cifar10Trainer
from cifar10validator import Cifar10Validator

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal

TRAIN_TASK_NAME = "train"


@pytest.fixture()
def get_cifar_trainer():
    with patch.object(Cifar10Trainer, "_save_local_model") as mock_save:
        with patch.object(Cifar10Trainer, "_load_local_model") as mock_load:
            yield Cifar10Trainer(train_task_name=TRAIN_TASK_NAME, epochs=1)


class TestCifar10Trainer:
    @pytest.mark.parametrize("num_rounds", [1, 3])
    def test_execute(self, get_cifar_trainer, num_rounds):
        trainer = get_cifar_trainer
        # just take first batch
        iterator = iter(trainer._train_loader)
        trainer._train_loader = [next(iterator)]

        dxo = DXO(data_kind=DataKind.WEIGHTS, data=trainer.model.state_dict())
        result = dxo.to_shareable()
        for i in range(num_rounds):
            result = trainer.execute(TRAIN_TASK_NAME, shareable=result, fl_ctx=FLContext(), abort_signal=Signal())
            assert result.get_return_code() == ReturnCode.OK

    @patch.object(Cifar10Trainer, "_save_local_model")
    @patch.object(Cifar10Trainer, "_load_local_model")
    def test_execute_rounds(self, mock_save, mock_load):
        train_task_name = "train"
        trainer = Cifar10Trainer(train_task_name=train_task_name, epochs=2)
        # just take first batch
        myitt = iter(trainer._train_loader)
        trainer._train_loader = [next(myitt)]

        dxo = DXO(data_kind=DataKind.WEIGHTS, data=trainer.model.state_dict())
        result = dxo.to_shareable()
        for i in range(3):
            result = trainer.execute(train_task_name, shareable=result, fl_ctx=FLContext(), abort_signal=Signal())
            assert result.get_return_code() == ReturnCode.OK


class TestCifar10Validator:
    def test_execute(self):
        validate_task_name = "validate"
        validator = Cifar10Validator(validate_task_name=validate_task_name)
        # just take first batch
        iterator = iter(validator._test_loader)
        validator._test_loader = [next(iterator)]

        dxo = DXO(data_kind=DataKind.WEIGHTS, data=validator.model.state_dict())
        result = validator.execute(
            validate_task_name, shareable=dxo.to_shareable(), fl_ctx=FLContext(), abort_signal=Signal()
        )
        assert result.get_return_code() == ReturnCode.OK
