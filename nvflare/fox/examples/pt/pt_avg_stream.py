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
import threading

import torch

from nvflare.fox.api.app import ClientApp, ServerApp
from nvflare.fox.api.constants import ContextKey, EnvType
from nvflare.fox.api.ctx import Context
from nvflare.fox.api.dec import collab
from nvflare.fox.api.group import all_clients
from nvflare.fox.api.strategy import Strategy
from nvflare.fox.api.utils import simple_logging
from nvflare.fox.examples.pt.utils import parse_state_dict
from nvflare.fox.sim.simulator import Simulator
from nvflare.fox.sys.model_downloader import ModelDownloader, download_model
from nvflare.fuel.utils.log_utils import get_obj_logger


class _AggrResult:

    def __init__(self):
        self.total = {}
        self.count = 0
        self.lock = threading.Lock()  # ensure update integrity


class PTFedAvgStream(Strategy):

    def __init__(self, initial_model, num_rounds=10, timeout=2.0):
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self.timeout = timeout
        self.name = "PTFedAvgStream"
        self.logger = get_obj_logger(self)
        self._init_model = parse_state_dict(initial_model)

    def execute(self, context: Context):
        self.logger.info(f"[{context.header_str()}] Start training for {self.num_rounds} rounds")
        current_model = context.get_prop(ContextKey.INPUT, self._init_model)
        for i in range(self.num_rounds):
            current_model = self._do_one_round(i, current_model, context)
        self.logger.info(f"FINAL MODEL: {current_model}")
        return current_model

    def _do_one_round(self, r, current_model, ctx: Context):
        aggr_result = _AggrResult()

        grp = all_clients(
            ctx,
            process_resp_cb=self._accept_train_result,
            aggr_result=aggr_result,
        )

        if ctx.env_type == EnvType.SYSTEM:
            downloader = ModelDownloader(
                num_receivers=grp.size,
                ctx=ctx,
                timeout=5.0,
            )
            model_type = "ref"
            model = downloader.add_model(current_model, 2)
            self.logger.info(f"prepared model as ref: {model}")
        else:
            model = current_model
            model_type = "model"

        grp.train(r, model, model_type)

        if aggr_result.count == 0:
            return None
        else:
            result = {}
            for k, v in aggr_result.total.items():
                result[k] = torch.div(v, aggr_result.count)
            self.logger.info(f"[{ctx.header_str()}] round {r}: aggr result from {aggr_result.count} clients: {result}")
            return result

    def _accept_train_result(self, result, aggr_result: _AggrResult, context: Context):
        self.logger.info(f"[{context.header_str()}] got train result from {context.caller}: {result}")

        model, model_type = result
        if model_type == "ref":
            err, model = download_model(
                ref=model,
                per_request_timeout=5.0,
                ctx=context,
                model_received_cb=self._aggregate_tensors,
                aggr_result=aggr_result,
                context=context,
            )
            if err:
                raise RuntimeError(f"failed to download model {model}: {err}")
        else:
            for k, v in model.items():
                if k not in aggr_result.total:
                    aggr_result.total[k] = v
                else:
                    aggr_result.total[k] += v

        aggr_result.count += 1
        return None

    def _aggregate_tensors(self, td: dict[str, torch.Tensor], aggr_result: _AggrResult, context: Context):
        self.logger.info(f"[{context.header_str()}] aggregating received tensor: {td}")
        with aggr_result.lock:
            for k, v in td.items():
                if k not in aggr_result.total:
                    aggr_result.total[k] = v
                else:
                    aggr_result.total[k] += v


class PTTrainer(ClientApp):

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
            err, model = download_model(ref=weights, per_request_timeout=5.0, ctx=context)
            if err:
                raise RuntimeError(f"failed to download model {weights}: {err}")
            self.logger.info(f"downloaded model {model}")
            weights = model

        result = {}
        for k, v in weights.items():
            result[k] = v + self.delta

        if model_type == "ref":
            # stream it
            downloader = ModelDownloader(
                num_receivers=1,
                ctx=context,
                timeout=5.0,
            )
            model_type = "ref"
            model = downloader.add_model(result, 2)
            self.logger.info(f"prepared result as ref: {model}")
        else:
            model = result
            model_type = "model"
        return model, model_type


def main():
    simple_logging(logging.DEBUG)

    server_app = ServerApp(
        strategy_name="fed_avg_in_time",
        strategy=PTFedAvgStream(
            initial_model={
                "x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "y": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                "z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            },
            num_rounds=4,
        ),
    )

    client_app = PTTrainer(delta=1.0)

    simulator = Simulator(
        root_dir="/tmp/fox",
        experiment_name="pt_fedavg_intime",
        server_app=server_app,
        client_app=client_app,
        num_clients=2,
    )

    simulator.run()


if __name__ == "__main__":
    main()
