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

import torch

from nvflare.collab import fox
from nvflare.collab.api.constants import BackendType
from nvflare.collab.api.utils import simple_logging
from nvflare.collab.examples import get_experiment_root
from nvflare.collab.examples.pt.utils import parse_state_dict
from nvflare.collab.sim.simulator import Simulator
from nvflare.collab.sys.downloader import Downloader, download_tensors
from nvflare.collab.utils.tensor_receiver import TensorReceiver
from nvflare.fuel.utils.log_utils import get_obj_logger


class PTFedAvgStream:

    def __init__(self, initial_model, num_rounds=10, timeout=2.0):
        self.num_rounds = num_rounds
        self.initial_model = initial_model
        self.timeout = timeout
        self.name = "PTFedAvgStream"
        self.logger = get_obj_logger(self)
        self._init_model = parse_state_dict(initial_model)

    @fox.algo
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
        grp = fox.clients(
            blocking=False,
            process_resp_cb=TensorReceiver(),
        )

        if fox.backend_type == BackendType.FLARE:
            downloader = Downloader(
                num_receivers=grp.size,
                timeout=5.0,
            )
            model_type = "ref"
            model = downloader.add_tensors(current_model, 0)
            self.logger.info(f"prepared model as ref: {model}")
        else:
            model = current_model
            model_type = "model"

        aggr_result = {}
        aggr_count = {}
        results = grp.train(r, model, model_type)

        # results is a queue that contains chunks of tensors that are downloaded from clients.
        # we aggregate them while they are being downloaded in parallel.
        # Note that the chunks from different sites may arrive in any order.
        for n, tensors in results:
            self.logger.info(f"got tensors from {n}: {tensors}")
            if not tensors:
                # we use None to indicate the end of all chunks from a site.
                continue

            for k, v in tensors.items():
                if k not in aggr_result:
                    aggr_result[k] = v
                    aggr_count[k] = 1
                else:
                    aggr_result[k] += v
                    aggr_count[k] += 1

        final_result = {}
        for k, v in aggr_result.items():
            final_result[k] = torch.div(v, aggr_count[k])
        self.logger.info(f"[{fox.call_info}] round {r}: aggr result: {final_result}")
        return final_result


class PTTrainer:

    def __init__(self, delta: float):
        self.delta = delta
        self.logger = get_obj_logger(self)

    @fox.collab
    def train(self, current_round, model, model_type: str):
        if fox.is_aborted:
            self.logger.debug("training aborted")
            return None, "model"

        self.logger.debug(f"[{fox.call_info}] training round {current_round}: {model_type=} {model=}")
        if model_type == "ref":
            err, model = download_tensors(ref=model, per_request_timeout=5.0)
            if err:
                raise RuntimeError(f"failed to download model {model}: {err}")
            self.logger.info(f"downloaded model1 {model}")

        result = {}
        for k, v in model.items():
            result[k] = v + self.delta

        if model_type == "ref":
            # stream it
            downloader = Downloader(
                num_receivers=1,
                timeout=5.0,
            )
            model_type = "ref"
            model = downloader.add_tensors(result, 0)
            self.logger.info(f"prepared result as ref: {model}")
        else:
            model = result
            model_type = "model"
        return model, model_type


def main():
    simple_logging(logging.DEBUG)

    server = PTFedAvgStream(
        initial_model={
            "x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "y": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        },
        num_rounds=2,
    )

    client = PTTrainer(delta=1.0)

    simulator = Simulator(
        root_dir=get_experiment_root(),
        experiment_name="pt_fedavg_stream2",
        server=server,
        client=client,
        num_clients=2,
    )

    result = simulator.run()
    print(f"final result: {result}")


if __name__ == "__main__":
    main()
