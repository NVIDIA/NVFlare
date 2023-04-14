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

import os
from typing import Union

from nvflare.apis.dxo import DXO, from_bytes
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.app_opt.he.homomorphic_encrypt import load_tenseal_context_from_workspace, serialize_nested_dict
from nvflare.security.logging import secure_format_exception


# TODO: Might be able to use CrossSiteModelEval directly
class HECrossSiteModelEval(CrossSiteModelEval):
    def __init__(
        self,
        tenseal_context_file="server_context.tenseal",
        task_check_period=0.5,
        cross_val_dir=AppConstants.CROSS_VAL_DIR,
        submit_model_timeout=600,
        validation_timeout: int = 6000,
        model_locator_id="",
        formatter_id="",
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        validation_task_name=AppConstants.TASK_VALIDATION,
        cleanup_models=False,
        participating_clients=None,
        wait_for_clients_timeout=300,
    ):
        """Cross Site Model Validation workflow for HE.

        Args:
            task_check_period (float, optional): How often to check for new tasks or tasks being finished.
                Defaults to 0.5.
            cross_val_dir (str, optional): Path to cross site validation directory relative to run directory.
                Defaults to `AppConstants.CROSS_VAL_DIR`.
            submit_model_timeout (int, optional): Timeout of submit_model_task. Defaults to 600 secs.
            validation_timeout (int, optional): Timeout for validate_model task. Defaults to 6000 secs.
            model_locator_id (str, optional): ID for model_locator component. Defaults to "".
            formatter_id (str, optional): ID for formatter component. Defaults to "".
            submit_model_task_name (str, optional): Name of submit_model task. Defaults to `AppConstants.TASK_SUBMIT_MODEL`.
            validation_task_name (str, optional): Name of validate_model task. Defaults to `AppConstants.TASK_VALIDATION`.
            cleanup_models (bool, optional): Whether or not models should be deleted after run. Defaults to False.
            participating_clients (list, optional): List of participating client names. If not provided, defaults
                to all clients connected at start of controller.
            wait_for_clients_timeout (int, optional): Timeout for clients to appear. Defaults to 300 secs
        """
        super().__init__(
            task_check_period=task_check_period,
            cross_val_dir=cross_val_dir,
            validation_timeout=validation_timeout,
            model_locator_id=model_locator_id,
            formatter_id=formatter_id,
            validation_task_name=validation_task_name,
            submit_model_task_name=submit_model_task_name,
            submit_model_timeout=submit_model_timeout,
            cleanup_models=cleanup_models,
            participating_clients=participating_clients,
            wait_for_clients_timeout=wait_for_clients_timeout,
        )
        self.tenseal_context = None
        self.tenseal_context_file = tenseal_context_file

    def start_controller(self, fl_ctx: FLContext):
        super().start_controller(fl_ctx)
        self.tenseal_context = load_tenseal_context_from_workspace(self.tenseal_context_file, fl_ctx)

    def _save_validation_content(self, name: str, save_dir: str, dxo: DXO, fl_ctx: FLContext) -> str:
        """Saves shareable to given directory within the app_dir.

        Args:
            name (str): Name of shareable
            save_dir (str): Relative path to directory in which to save
            dxo (DXO): DXO object
            fl_ctx (FLContext): FLContext object

        Returns:
            str: Path to the file saved.
        """
        # Save the model with name as the filename to shareable directory
        data_filename = os.path.join(save_dir, name)

        try:
            serialize_nested_dict(dxo.data)
            bytes_to_save = dxo.to_bytes()
        except Exception as e:
            raise ValueError(f"Unable to extract shareable contents. Exception: {(secure_format_exception(e))}")

        # Save contents to path
        try:
            with open(data_filename, "wb") as f:
                f.write(bytes_to_save)
        except Exception as e:
            raise ValueError(f"Unable to save shareable contents: {secure_format_exception(e)}")

        self.log_debug(fl_ctx, f"Saved cross validation model with name: {name}.")

        return data_filename

    def _load_validation_content(self, name: str, load_dir: str, fl_ctx: FLContext) -> Union[DXO, None]:
        # Load shareable from disk
        shareable_filename = os.path.join(load_dir, name)

        # load shareable
        try:
            with open(shareable_filename, "rb") as f:
                data = f.read()

            dxo: DXO = from_bytes(data)

            self.log_debug(fl_ctx, f"Loading cross validation shareable content with name: {name}.")
        except Exception as e:
            raise ValueError(f"Exception in loading shareable content for {name}: {secure_format_exception(e)}")

        return dxo
