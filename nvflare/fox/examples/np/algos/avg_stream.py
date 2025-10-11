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
import uuid

from nvflare.fox.api.app import ClientApp
from nvflare.fox.api.constants import ContextKey, EnvType
from nvflare.fox.api.ctx import Context
from nvflare.fox.api.dec import collab
from nvflare.fox.api.group import all_clients
from nvflare.fox.api.strategy import Strategy
from nvflare.fox.examples.np.algos.utils import load_np_model, parse_array_def, save_np_model
from nvflare.fox.sys.file_downloader import download_file, prepare_file_for_download
from nvflare.fuel.f3.streaming.obj_downloader import DownloadStatus
from nvflare.fuel.utils.log_utils import get_obj_logger


class _AggrResult:

    def __init__(self):
        self.total = 0
        self.count = 0


class NPFedAvgStream(Strategy):

    def __init__(self, initial_model, num_rounds=10, timeout=2.0):
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self.timeout = timeout
        self.name = "NPFedAvgStream"
        self.logger = get_obj_logger(self)
        self._init_model = parse_array_def(initial_model)

    def execute(self, context: Context):
        self.logger.info(f"[{context.header_str()}] Start training for {self.num_rounds} rounds")
        current_model = context.get_prop(ContextKey.INPUT, self._init_model)
        for i in range(self.num_rounds):
            current_model = self._do_one_round(i, current_model, context)
        self.logger.info(f"FINAL MODEL: {current_model}")
        return current_model

    def _do_one_round(self, r, current_model, ctx: Context):
        aggr_result = _AggrResult()

        # pretend the model is big
        file_name = None
        if ctx.env_type == EnvType.SYSTEM:
            file_name = f"/tmp/np_{str(uuid.uuid4())}.npy"
            save_np_model(current_model, file_name)
            model = prepare_file_for_download(
                file_name=file_name,
                ctx=ctx,
                timeout=5.0,
                file_downloaded_cb=self._model_downloaded,
            )
            model_type = "ref"
            self.logger.info(f"prepared model as ref: {model}")
        else:
            model = current_model
            model_type = "model"

        all_clients(
            ctx,
            process_resp_cb=self._accept_train_result,
            aggr_result=aggr_result,
        ).train(r, model, model_type)

        if file_name:
            os.remove(file_name)

        if aggr_result.count == 0:
            return None
        else:
            result = aggr_result.total / aggr_result.count
            self.logger.info(f"[{ctx.header_str()}] round {r}: aggr result from {aggr_result.count} clients: {result}")
            return result

    def _accept_train_result(self, result, aggr_result: _AggrResult, context: Context):
        self.logger.info(f"[{context.header_str()}] got train result from {context.caller}: {result}")

        model, model_type = result
        if model_type == "ref":
            err, file_path = download_file(ref=model, per_request_timeout=5.0, ctx=context)
            if err:
                raise RuntimeError(f"failed to download model file {model}: {err}")
            self.logger.info(f"downloaded model file to {file_path}")
            model = load_np_model(file_path)
            os.remove(file_path)

        aggr_result.total += model
        aggr_result.count += 1
        return None

    def _model_downloaded(self, ref_id: str, to_site: str, status: str, file_name):
        self.logger.info(f"model file {file_name} downloaded by {to_site}: {ref_id=} {status=}")


class NPTrainer(ClientApp):

    def __init__(self, delta: float):
        ClientApp.__init__(self)
        self.delta = delta

    @collab
    def train(self, current_round, weights, model_type: str, context: Context):
        if context.is_aborted():
            self.logger.debug("training aborted")
            return 0
        self.logger.debug(f"[{context.header_str()}] training round {current_round}: {model_type=} {weights=}")
        if model_type == "ref":
            err, file_path = download_file(ref=weights, per_request_timeout=5.0, ctx=context)
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
            model = prepare_file_for_download(
                file_name=file_name,
                ctx=context,
                timeout=5.0,
                file_downloaded_cb=self._result_downloaded,
            )
            model_type = "ref"
            self.logger.info(f"prepared result as ref: {model}")
        else:
            model = result
            model_type = "model"
        return model, model_type

    def _result_downloaded(self, ref_id: str, to_site: str, status: str, file_name):
        self.logger.info(f"file {file_name} downloaded to {to_site}: {ref_id=} {status=}")
        os.remove(file_name)
