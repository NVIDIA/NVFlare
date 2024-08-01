# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.app_common.app_constant import AppConstants
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.import_utils import optional_import

torch, torch_ok = optional_import(module="torch")
if torch_ok:
    from nvflare.app_opt.pt.params_converter import NumpyToPTParamsConverter, PTToNumpyParamsConverter

    DEFAULT_PARAMS_EXCHANGE_FORMAT = ExchangeFormat.PYTORCH
else:
    DEFAULT_PARAMS_EXCHANGE_FORMAT = ExchangeFormat.NUMPY

tensorflow, tf_ok = optional_import(module="tensorflow")
if tf_ok:
    from nvflare.app_opt.tf.params_converter import KerasModelToNumpyParamsConverter, NumpyToKerasModelParamsConverter


class ScriptExecutor:
    def __init__(
        self,
        script: str,
        script_args: str = "",
        launch_external_process: bool = False,
        launch_once: bool = True,
        params_transfer_type: TransferType = TransferType.FULL,
        params_exchange_format=DEFAULT_PARAMS_EXCHANGE_FORMAT,
    ):
        """ScriptExecutor is used with FedJob API to run or launch a script.

        in-process `launch_external_process=False` uses InProcessClientAPIExecutor (default).
        ex-process `launch_external_process=True` uses ClientAPILauncherExecutor.

        Args:
            script (str): Script to run. For in-process must be a python script path. For ex-process can be any command supported by python subprocess.
            script_args (str): Optional arguments for script.
            launch_external_process (bool): Whether to launch the script in external process. Defaults to False.
            launch_once (bool): If True launch script once, else launch for every task. Defaults to True.
            params_transfer_type (str): How to transfer the parameters. FULL means the whole model parameters are sent.
                DIFF means that only the difference is sent. Defaults to TransferType.FULL.
            params_exchange_format (str): Format to exchange the parameters. Defaults based on detected imports.
        """
        self._script = script
        self._script_args = script_args
        self._launch_external_process = launch_external_process
        self._launch_once = launch_once
        self._params_transfer_type = params_transfer_type
        self._params_exchange_format = params_exchange_format

        self._from_nvflare_converter = None
        self._to_nvflare_converter = None

        if torch_ok:
            if params_exchange_format == ExchangeFormat.PYTORCH:
                self._from_nvflare_converter = NumpyToPTParamsConverter(
                    [AppConstants.TASK_TRAIN, AppConstants.TASK_VALIDATION]
                )
                self._to_nvflare_converter = PTToNumpyParamsConverter(
                    [AppConstants.TASK_TRAIN, AppConstants.TASK_SUBMIT_MODEL]
                )
        if tf_ok:
            if params_exchange_format == ExchangeFormat.NUMPY:
                self._from_nvflare_converter = NumpyToKerasModelParamsConverter(
                    [AppConstants.TASK_TRAIN, AppConstants.TASK_VALIDATION]
                )
                self._to_nvflare_converter = KerasModelToNumpyParamsConverter(
                    [AppConstants.TASK_TRAIN, AppConstants.TASK_SUBMIT_MODEL]
                )
        # TODO: support other params_exchange_format
