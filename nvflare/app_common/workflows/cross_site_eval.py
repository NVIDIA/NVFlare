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

import json
import os
import shutil
import time

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants, DefaultCheckpointFileName, ModelName
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.fuel.utils import fobs

from .model_controller import ModelController


class CrossSiteEval(ModelController):
    def __init__(
        self,
        *args,
        cross_val_dir=AppConstants.CROSS_VAL_DIR,
        submit_model_timeout=600,
        validation_timeout: int = 6000,
        server_models=[DefaultCheckpointFileName.GLOBAL_MODEL],
        participating_clients=None,
        **kwargs,
    ):
        """Cross Site Evaluation Workflow.

        # TODO: change validation to evaluation to reflect the real meaning

        Args:
            cross_val_dir (str, optional): Path to cross site validation directory relative to run directory.
                Defaults to "cross_site_val".
            submit_model_timeout (int, optional): Timeout of submit_model_task. Defaults to 600 secs.
            validation_timeout (int, optional): Timeout for validate_model task. Defaults to 6000 secs.
            participating_clients (list, optional): List of participating client names. If not provided, defaults
                to all clients connected at start of controller.

        """
        super().__init__(*args, **kwargs)
        self._cross_val_dir = cross_val_dir
        self._submit_model_timeout = submit_model_timeout
        self._validation_timeout = validation_timeout
        self._server_models = server_models
        self._participating_clients = participating_clients

        self._val_results = {}
        self._client_models = {}

        self._cross_val_models_dir = None
        self._cross_val_results_dir = None

        self._results_dir = AppConstants.CROSS_VAL_DIR
        self._json_val_results = {}
        self._json_file_name = "cross_val_results.json"

    def initialize(self, fl_ctx):
        super().initialize(fl_ctx)

        # Create shareable dirs for models and results
        cross_val_path = os.path.join(self.get_run_dir(), self._cross_val_dir)
        self._cross_val_models_dir = os.path.join(cross_val_path, AppConstants.CROSS_VAL_MODEL_DIR_NAME)
        self._cross_val_results_dir = os.path.join(cross_val_path, AppConstants.CROSS_VAL_RESULTS_DIR_NAME)

        # Cleanup/create the cross val models and results directories
        if os.path.exists(self._cross_val_models_dir):
            shutil.rmtree(self._cross_val_models_dir)
        if os.path.exists(self._cross_val_results_dir):
            shutil.rmtree(self._cross_val_results_dir)

        os.makedirs(self._cross_val_models_dir)
        os.makedirs(self._cross_val_results_dir)

        if self._participating_clients is None:
            self._participating_clients = self.sample_clients()

        for c_name in self._participating_clients:
            self._client_models[c_name] = None
            self._val_results[c_name] = {}

    def run(self) -> None:
        self.info("Start Cross-Site Evaluation.")

        data = FLModel(params={})
        data.meta[AppConstants.SUBMIT_MODEL_NAME] = ModelName.BEST_MODEL
        # Create submit_model task and broadcast to all participating clients
        self.send_model(
            task_name=AppConstants.TASK_SUBMIT_MODEL,
            data=data,
            targets=self._participating_clients,
            timeout=self._submit_model_timeout,
            callback=self._receive_local_model_cb,
        )

        if self.persistor and not isinstance(self.persistor, ModelPersistor):
            self.warning(
                f"Model Persistor {self._persistor_id} must be a ModelPersistor type object, "
                f"but got {type(self.persistor)}"
            )
            self.persistor = None

        # Obtain server models and send to clients for validation
        for server_model_name in self._server_models:
            try:
                if self.persistor:
                    server_model_learnable = self.persistor.get_model(server_model_name, self.fl_ctx)
                    server_model = FLModelUtils.from_model_learnable(server_model_learnable)
                else:
                    server_model = fobs.loadf(server_model_name)
            except Exception as e:
                self.exception(f"Unable to load server model {server_model_name}: {e}")
            self._send_validation_task(server_model_name, server_model)

        # Wait for all standing tasks to complete, since we used non-blocking `send_model()`
        while self.get_num_standing_tasks():
            if self.abort_signal.triggered:
                self.info("Abort signal triggered. Finishing cross site validation.")
                return
            self.debug("Checking standing tasks to see if cross site validation finished.")
            time.sleep(self._task_check_period)

        self.save_results()
        self.info("Stop Cross-Site Evaluation.")

    def _receive_local_model_cb(self, model: FLModel):
        client_name = model.meta["client_name"]

        save_path = os.path.join(self._cross_val_models_dir, client_name)
        fobs.dumpf(model, save_path)

        self.info(f"Saved client model {client_name} to {save_path}")
        self._client_models[client_name] = save_path

        # Send this model to all clients to validate
        self._send_validation_task(client_name, model)

    def _send_validation_task(self, model_name: str, model: FLModel):
        self.info(f"Sending {model_name} model to all participating clients for validation.")
        # Create validation task and broadcast to all participating clients.
        model.meta[AppConstants.MODEL_OWNER] = model_name

        self.send_model(
            task_name=AppConstants.TASK_VALIDATION,
            data=model,
            targets=self._participating_clients,
            timeout=self._validation_timeout,
            callback=self._receive_val_result_cb,
        )

    def _receive_val_result_cb(self, model: FLModel):
        client_name = model.meta["client_name"]
        model_owner = model.meta["props"].get(AppConstants.MODEL_OWNER, None)

        self.track_results(model_owner, client_name, model)

        file_path = os.path.join(self._cross_val_models_dir, client_name + "_" + model_owner)
        fobs.dumpf(model, file_path)

        client_results = self._val_results.get(client_name, None)
        if not client_results:
            client_results = {}
            self._val_results[client_name] = client_results
        client_results[model_owner] = file_path
        self.info(f"Saved validation result from client '{client_name}' on model '{model_owner}' in {file_path}")

    def track_results(self, model_owner, data_client, val_results: FLModel):
        if not model_owner:
            self.error("model_owner unknown. Validation result will not be saved to json")
        if not data_client:
            self.error("data_client unknown. Validation result will not be saved to json")

        if val_results:
            try:
                if data_client not in self._json_val_results:
                    self._json_val_results[data_client] = {}
                self._json_val_results[data_client][model_owner] = val_results.metrics

            except Exception:
                self.exception("Exception in handling validation result.")
        else:
            self.error("Validation result not found.", fire_event=False)

    def save_results(self):
        cross_val_res_dir = os.path.join(self.get_run_dir(), self._results_dir)
        if not os.path.exists(cross_val_res_dir):
            os.makedirs(cross_val_res_dir)

        res_file_path = os.path.join(cross_val_res_dir, self._json_file_name)
        self.info(f"saving validation result {self._json_val_results} to {res_file_path}")
        with open(res_file_path, "w") as f:
            f.write(json.dumps(self._json_val_results, indent=2))
