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
import os.path
import threading
import uuid

from nvflare.collab import fox
from nvflare.collab.api.constants import BackendType
from nvflare.collab.examples.np.mains.utils import load_np_model, parse_array_def, save_np_model
from nvflare.collab.sys.downloader import Downloader, download_file
from nvflare.fuel.utils.log_utils import get_obj_logger


class _AggrResult:

    def __init__(self):
        self.total = 0
        self.count = 0
        self.lock = threading.Lock()


class NPFedAvgStream:

    def __init__(self, initial_model, num_rounds=10, timeout=2.0):
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self.timeout = timeout
        self.name = "NPFedAvgStream"
        self.logger = get_obj_logger(self)
        self._init_model = parse_array_def(initial_model)

    @fox.main
    def execute(self):
        self.logger.info(f"[{fox.call_info}] Start training for {self.num_rounds} rounds")
        current_model = self._init_model
        for i in range(self.num_rounds):
            current_model = self._do_one_round(i, current_model)
            if current_model is None:
                self.logger.error(f"training failed at round {i}")
                break
        self.logger.info(f"FINAL MODEL: {current_model}")
        return current_model

    def _do_one_round(self, r, current_model):
        aggr_result = _AggrResult()
        grp = fox.clients(
            process_resp_cb=self._accept_train_result,
            aggr_result=aggr_result,
        )

        # pretend the model is big
        file_name = None
        if fox.backend_type == BackendType.FLARE:
            file_name = f"/tmp/np_{str(uuid.uuid4())}.npy"
            save_np_model(current_model, file_name)
            downloader = Downloader(
                num_receivers=grp.size,
                timeout=5.0,
            )
            model = downloader.add_file(file_name=file_name, file_downloaded_cb=self._model_downloaded)
            model_type = "ref"
            self.logger.info(f"prepared model as ref: {model}")
        else:
            model = current_model
            model_type = "model"

        grp.train(r, model, model_type)

        if file_name:
            # train is a blocking call that does not return until train results (success or not) are received
            # from all clients.
            # remove the file regardless.
            os.remove(file_name)

        if aggr_result.count == 0:
            return None
        else:
            result = aggr_result.total / aggr_result.count
            self.logger.info(f"[{fox.call_info}] round {r}: aggr result from {aggr_result.count} clients: {result}")
            return result

    def _accept_train_result(self, gcc, result, aggr_result: _AggrResult):
        self.logger.info(f"[{fox.call_info}] got train result from {fox.caller}: {result}")

        model, model_type = result
        if model_type == "ref":
            err, file_path = download_file(ref=model, per_request_timeout=5.0)
            if err:
                raise RuntimeError(f"failed to download model file {model}: {err}")
            self.logger.info(f"downloaded model file to {file_path}")
            model = load_np_model(file_path)
            os.remove(file_path)

        with aggr_result.lock:
            aggr_result.total += model
            aggr_result.count += 1
        return None

    def _model_downloaded(self, to_site: str, status: str, file_name):
        self.logger.info(f"model file {file_name} downloaded by {to_site}: {status=}")


class NPTrainer:

    def __init__(self, delta: float):
        self.delta = delta
        self.logger = get_obj_logger(self)

    @fox.publish
    def train(self, current_round, weights, model_type: str):
        if fox.is_aborted:
            self.logger.debug("training aborted")
            return None, ""

        self.logger.debug(f"[{fox.call_info}] training round {current_round}: {model_type=} {weights=}")
        if model_type == "ref":
            err, file_path = download_file(ref=weights, per_request_timeout=5.0)
            if err:
                raise RuntimeError(f"failed to download model file {weights}: {err}")
            self.logger.info(f"downloaded model file to {file_path}")
            weights = load_np_model(file_path)
            self.logger.info(f"loaded model from file: {weights}")
            os.remove(file_path)

        result = weights + self.delta

        if model_type == "ref":
            # stream it
            file_name = f"/tmp/np_{str(uuid.uuid4())}.npy"
            save_np_model(result, file_name)
            downloader = Downloader(
                num_receivers=1,
                timeout=5.0,
            )
            result = downloader.add_file(file_name=file_name, file_downloaded_cb=self._result_downloaded)
            self.logger.info(f"prepared result as ref: {result}")

        return result, model_type

    def _result_downloaded(self, to_site: str, status: str, file_name):
        self.logger.info(f"model file {file_name} downloaded to {to_site}: {status=}")
        if not to_site:
            # downloaded to all sites
            os.remove(file_name)
            self.logger.info(f"model file {file_name} removed")
