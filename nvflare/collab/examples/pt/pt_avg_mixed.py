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

import numpy as np
import torch

from nvflare.collab import collab
from nvflare.collab.api.constants import BackendType
from nvflare.collab.api.utils import simple_logging
from nvflare.collab.examples import get_experiment_root
from nvflare.collab.examples.np.mains.utils import add as add_np
from nvflare.collab.examples.np.mains.utils import div as div_np
from nvflare.collab.examples.np.mains.utils import parse_state_dict as parse_np
from nvflare.collab.examples.pt.utils import add as add_pt
from nvflare.collab.examples.pt.utils import div as div_pt
from nvflare.collab.examples.pt.utils import parse_state_dict as parse_pt
from nvflare.collab.sim.simulator import Simulator
from nvflare.collab.sys.downloader import Downloader, download_arrays, download_tensors
from nvflare.fuel.utils.log_utils import get_obj_logger


class _AggrResult:

    def __init__(self):
        self.pt_total = {}
        self.np_total = {}
        self.count = 0
        self.lock = threading.Lock()  # ensure update integrity


class PTFedAvgMixed:

    def __init__(self, pt_model, np_model, num_rounds=10, timeout=2.0):
        self.num_rounds = num_rounds
        self.pt_model = pt_model
        self.np_model = np_model
        self.timeout = timeout
        self.name = "PTFedAvgMixed"
        self.logger = get_obj_logger(self)
        self._pt_model = parse_pt(pt_model)
        self._np_model = parse_np(np_model)

    @collab.main
    def execute(self):
        self.logger.info(f"[{collab.call_info}] Start training for {self.num_rounds} rounds")
        pt_model, np_model = self._pt_model, self._np_model
        for i in range(self.num_rounds):
            pt_model, np_model = self._do_one_round(i, pt_model, np_model)
            if pt_model is None or np_model is None:
                self.logger.error(f"training failed at round {i}")
                break
        self.logger.info(f"FINAL MODEL: {pt_model=} {np_model=}")
        return pt_model, np_model

    def _do_one_round(self, r, pt_model, np_model):
        aggr_result = _AggrResult()

        grp = collab.clients(
            process_resp_cb=self._accept_train_result,
            aggr_result=aggr_result,
        )

        if collab.backend_type == BackendType.FLARE:
            downloader = Downloader(
                num_receivers=grp.size,
                timeout=5.0,
            )
            model_type = "ref"
            pt_model = downloader.add_tensors(pt_model, 0)
            np_model = downloader.add_arrays(np_model, 0)
            self.logger.info(f"prepared model as ref: {pt_model=} {np_model=}")
        else:
            model_type = "model"

        grp.train(r, pt_model, np_model, model_type)

        if aggr_result.count == 0:
            return None, None
        else:
            pt_result = aggr_result.pt_total
            div_pt(pt_result, aggr_result.count)
            self.logger.info(
                f"[{collab.call_info}] round {r}: aggr PT result from {aggr_result.count} clients: {pt_result}"
            )

            np_result = aggr_result.np_total
            div_np(np_result, aggr_result.count)
            self.logger.info(
                f"[{collab.call_info}] round {r}: aggr NP result from {aggr_result.count} clients: {np_result}"
            )
            return pt_result, np_result

    def _accept_train_result(self, gcc, result, aggr_result: _AggrResult):
        self.logger.info(f"[{collab.call_info}] got train result from {collab.caller}: {result}")

        pt_result, np_result, model_type = result
        if model_type == "ref":
            err, pt_result = download_tensors(
                ref=pt_result,
                per_request_timeout=5.0,
                tensors_received_cb=self._aggregate_tensors,
                aggr_result=aggr_result,
            )
            if err:
                raise RuntimeError(f"failed to download model {pt_result}: {err}")

            err, np_result = download_arrays(
                ref=np_result,
                per_request_timeout=5.0,
                arrays_received_cb=self._aggregate_arrays,
                aggr_result=aggr_result,
            )
            if err:
                raise RuntimeError(f"failed to download NP model file {np_result}: {err}")
        else:
            with aggr_result.lock:
                add_pt(pt_result, aggr_result.pt_total)
                add_np(np_result, aggr_result.np_total)

        with aggr_result.lock:
            aggr_result.count += 1
        return None

    def _aggregate_tensors(self, td: dict[str, torch.Tensor], aggr_result: _AggrResult):
        self.logger.info(f"[{collab.call_info}] aggregating received tensor: {td}")
        with aggr_result.lock:
            add_pt(td, aggr_result.pt_total)

    def _aggregate_arrays(self, td: dict[str, np.ndarray], aggr_result: _AggrResult):
        self.logger.info(f"[{collab.call_info}] aggregating received array: {td}")
        with aggr_result.lock:
            add_np(td, aggr_result.np_total)


class PTTrainer:

    def __init__(self, delta: float):
        self.delta = delta
        self.logger = get_obj_logger(self)

    @collab.publish
    def train(self, current_round, pt_model, np_model, model_type: str):
        if collab.is_aborted:
            self.logger.debug("training aborted")
            return None, None, ""

        self.logger.debug(f"[{collab.call_info}] training round {current_round}: {model_type=} {pt_model=} {np_model=}")

        if model_type == "ref":
            err, pt_model = download_tensors(ref=pt_model, per_request_timeout=5.0)
            if err:
                raise RuntimeError(f"failed to download PT model {pt_model}: {err}")
            self.logger.info(f"downloaded PT model {pt_model}")

            err, np_model = download_arrays(ref=np_model, per_request_timeout=5.0)
            if err:
                raise RuntimeError(f"failed to download NP model {np_model}: {err}")
            self.logger.info(f"downloaded NP model {np_model}")

        pt_result = {}
        for k, v in pt_model.items():
            pt_result[k] = v + self.delta

        np_result = {}
        for k, v in np_model.items():
            np_result[k] = v + self.delta

        if model_type == "ref":
            # stream it
            downloader = Downloader(
                num_receivers=1,
                timeout=5.0,
            )
            pt_result = downloader.add_tensors(pt_result, 0)
            self.logger.info(f"prepared PT result as ref: {pt_result}")

            np_result = downloader.add_arrays(np_result, 0)
            self.logger.info(f"prepared NP result as ref: {np_result}")
        return pt_result, np_result, model_type


def main():
    simple_logging(logging.DEBUG)

    init_model = {
        "x": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "y": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "z": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    }

    server = PTFedAvgMixed(
        pt_model=init_model,
        np_model=init_model,
        num_rounds=4,
    )

    client = PTTrainer(delta=1.0)

    simulator = Simulator(
        root_dir=get_experiment_root(),
        experiment_name="fedavg_mixed",
        server=server,
        client=client,
        num_clients=2,
    )

    result = simulator.run()
    print(f"Final result: {result}")


if __name__ == "__main__":
    main()
