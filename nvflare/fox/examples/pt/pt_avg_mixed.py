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
import logging
import os
import threading
import uuid

import torch

from nvflare.app_opt.pt.tensor_downloader import TensorDownloadable
from nvflare.fox.api.app import ClientApp, ServerApp
from nvflare.fox.api.constants import EnvType
from nvflare.fox.api.ctx import Context
from nvflare.fox.api.dec import collab
from nvflare.fox.api.group import all_clients
from nvflare.fox.api.strategy import Strategy
from nvflare.fox.api.utils import simple_logging
from nvflare.fox.examples.np.algos.utils import load_np_model, parse_array_def, save_np_model
from nvflare.fox.examples.pt.utils import parse_state_dict
from nvflare.fox.sim.simulator import Simulator
from nvflare.fox.sys.file_downloader import download_file
from nvflare.fox.sys.general_downloader import GeneralDownloader
from nvflare.fox.sys.model_downloader import download_model
from nvflare.fuel.f3.streaming.file_downloader import FileDownloadable
from nvflare.fuel.utils.log_utils import get_obj_logger


class _AggrResult:

    def __init__(self):
        self.pt_total = {}
        self.np_total = 0
        self.count = 0
        self.lock = threading.Lock()  # ensure update integrity


class PTFedAvgMixed(Strategy):

    def __init__(self, pt_model, np_model, num_rounds=10, timeout=2.0):
        self.num_rounds = num_rounds
        self.pt_model = pt_model
        self.np_model = np_model
        self.timeout = timeout
        self.name = "PTFedAvgMixed"
        self.logger = get_obj_logger(self)
        self._pt_model = parse_state_dict(pt_model)
        self._np_model = parse_array_def(np_model)

    def execute(self, context: Context):
        self.logger.info(f"[{context.header_str()}] Start training for {self.num_rounds} rounds")
        pt_model, np_model = self._pt_model, self._np_model
        for i in range(self.num_rounds):
            pt_model, np_model = self._do_one_round(i, pt_model, np_model, context)
        self.logger.info(f"FINAL MODEL: {pt_model=} {np_model=}")
        return pt_model, np_model

    def _do_one_round(self, r, pt_model, np_model, ctx: Context):
        aggr_result = _AggrResult()

        grp = all_clients(
            ctx,
            process_resp_cb=self._accept_train_result,
            aggr_result=aggr_result,
        )

        file_name = None
        if ctx.env_type == EnvType.SYSTEM:
            file_name = f"/tmp/np_{str(uuid.uuid4())}.npy"
            save_np_model(np_model, file_name)

            downloader = GeneralDownloader(
                num_receivers=grp.size,
                ctx=ctx,
                timeout=5.0,
            )
            model_type = "ref"
            pt_model = downloader.add_object(TensorDownloadable(pt_model, 2))
            np_model = downloader.add_object(FileDownloadable(file_name))
            self.logger.info(f"prepared model as ref: {pt_model=} {np_model=}")
        else:
            model_type = "model"

        grp.train(r, pt_model, np_model, model_type)

        if file_name:
            os.remove(file_name)

        if aggr_result.count == 0:
            return None
        else:
            pt_result = {}
            for k, v in aggr_result.pt_total.items():
                pt_result[k] = torch.div(v, aggr_result.count)

            self.logger.info(
                f"[{ctx.header_str()}] round {r}: aggr PT result from {aggr_result.count} clients: {pt_result}"
            )

            np_result = aggr_result.np_total / aggr_result.count
            self.logger.info(
                f"[{ctx.header_str()}] round {r}: aggr NP result from {aggr_result.count} clients: {np_result}"
            )
            return pt_result, np_result

    def _accept_train_result(self, result, aggr_result: _AggrResult, context: Context):
        self.logger.info(f"[{context.header_str()}] got train result from {context.caller}: {result}")

        pt_result, np_result, model_type = result
        if model_type == "ref":
            err, pt_result = download_model(
                ref=pt_result,
                per_request_timeout=5.0,
                ctx=context,
                model_received_cb=self._aggregate_tensors,
                aggr_result=aggr_result,
                context=context,
            )
            if err:
                raise RuntimeError(f"failed to download model {pt_result}: {err}")

            err, file_path = download_file(ref=np_result, per_request_timeout=5.0, ctx=context)
            if err:
                raise RuntimeError(f"failed to download NP model file {np_result}: {err}")
            self.logger.info(f"downloaded model file to {file_path}")
            np_result = load_np_model(file_path)
            os.remove(file_path)
        else:
            for k, v in pt_result.items():
                if k not in aggr_result.pt_total:
                    aggr_result.pt_total[k] = v
                else:
                    aggr_result.pt_total[k] += v

        aggr_result.np_total += np_result
        aggr_result.count += 1
        return None

    def _aggregate_tensors(self, td: dict[str, torch.Tensor], aggr_result: _AggrResult, context: Context):
        self.logger.info(f"[{context.header_str()}] aggregating received tensor: {td}")
        with aggr_result.lock:
            for k, v in td.items():
                if k not in aggr_result.pt_total:
                    aggr_result.pt_total[k] = v
                else:
                    aggr_result.pt_total[k] += v


class PTTrainer(ClientApp):

    def __init__(self, delta: float):
        ClientApp.__init__(self)
        self.delta = delta

    @collab
    def train(self, current_round, pt_model, np_model, model_type: str, context: Context):
        if context.is_aborted():
            self.logger.debug("training aborted")
            return 0
        self.logger.debug(
            f"[{context.header_str()}] training round {current_round}: {model_type=} {pt_model=} {np_model=}"
        )
        if model_type == "ref":
            err, pt_model = download_model(ref=pt_model, per_request_timeout=5.0, ctx=context)
            if err:
                raise RuntimeError(f"failed to download PT model {pt_model}: {err}")
            self.logger.info(f"downloaded PT model {pt_model}")

            err, file_path = download_file(ref=np_model, per_request_timeout=5.0, ctx=context)
            if err:
                raise RuntimeError(f"failed to download NP model file {np_model}: {err}")
            self.logger.info(f"downloaded model file to {file_path}")
            np_model = load_np_model(file_path)
            self.logger.info(f"loaded NP model from file: {np_model}")
            os.remove(file_path)

        pt_result = {}
        for k, v in pt_model.items():
            pt_result[k] = v + self.delta

        np_result = np_model + self.delta

        if model_type == "ref":
            # stream it
            downloader = GeneralDownloader(
                num_receivers=1,
                ctx=context,
                timeout=5.0,
            )
            pt_result = downloader.add_object(TensorDownloadable(pt_result, 2))
            self.logger.info(f"prepared PT result as ref: {pt_result}")

            file_name = f"/tmp/np_{str(uuid.uuid4())}.npy"
            save_np_model(np_result, file_name)
            np_result = downloader.add_object(FileDownloadable(file_name, file_downloaded_cb=self._result_downloaded))
            self.logger.info(f"prepared NP result as ref: {np_result}")

        return pt_result, np_result, model_type

    def _result_downloaded(self, to_site: str, status: str, file_name):
        self.logger.info(f"NP model file {file_name} downloaded to {to_site}: {status=}")
        if not to_site:
            # downloaded to all sites
            os.remove(file_name)
            self.logger.info(f"NP model file {file_name} removed")


def main():
    simple_logging(logging.DEBUG)

    server_app = ServerApp(
        strategy_name="fed_avg_mixed",
        strategy=PTFedAvgMixed(
            pt_model={
                "x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "y": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            },
            np_model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            num_rounds=4,
        ),
    )

    client_app = PTTrainer(delta=1.0)

    simulator = Simulator(
        root_dir="/tmp/fox",
        experiment_name="fedavg_mixed",
        server_app=server_app,
        client_app=client_app,
        num_clients=2,
    )

    simulator.run()


if __name__ == "__main__":
    main()
