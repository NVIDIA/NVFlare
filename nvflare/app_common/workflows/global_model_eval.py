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

from nvflare.app_common.app_constant import AppConstants

from .cross_site_model_eval import CrossSiteModelEval


class GlobalModelEval(CrossSiteModelEval):
    def __init__(
        self,
        task_check_period=0.5,
        cross_val_dir=AppConstants.CROSS_VAL_DIR,
        validation_timeout: int = 6000,
        model_locator_id="",
        formatter_id="",
        validation_task_name=AppConstants.TASK_VALIDATION,
        cleanup_models=False,
        participating_clients=None,
        wait_for_clients_timeout=300,
    ):
        """Cross Site Model Validation workflow.

        Args:
            task_check_period (float, optional): How often to check for new tasks or tasks being finished.
                Defaults to 0.5.
            cross_val_dir (str, optional): Path to cross site validation directory relative to run directory.
                Defaults to "cross_site_val".
            validation_timeout (int, optional): Timeout for validate_model task. Defaults to 6000.
            model_locator_id (str, optional): ID for model_locator component. Defaults to None.
            formatter_id (str, optional): ID for formatter component. Defaults to None.
            validation_task_name (str, optional): Name of validate_model task. Defaults to "validate".
            cleanup_models (bool, optional): Whether or not models should be deleted after run. Defaults to False.
            participating_clients (list, optional): List of participating client names. If not provided, defaults
                to all clients connected at start of controller.
            wait_for_clients_timeout (int, optional): Timeout for clients to appear. Defaults to 300 secs
        """
        if not model_locator_id:
            raise ValueError("missing required model_locator_id")

        CrossSiteModelEval.__init__(
            self,
            task_check_period=task_check_period,
            cross_val_dir=cross_val_dir,
            validation_timeout=validation_timeout,
            model_locator_id=model_locator_id,
            formatter_id=formatter_id,
            validation_task_name=validation_task_name,
            submit_model_task_name="",
            cleanup_models=cleanup_models,
            participating_clients=participating_clients,
            wait_for_clients_timeout=wait_for_clients_timeout,
        )
