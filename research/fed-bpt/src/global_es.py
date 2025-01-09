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

import copy

import cma
import numpy as np

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.workflows.fedavg import FedAvg


class GlobalES(FedAvg):
    """Controller for [FedBPT](https://arxiv.org/abs/2310.01467) Workflow.
    Inherits arguments from the FedAvg base class.
    *Note*: This class is based on the experimental `ModelController`.

    Provides the implementations for the `run` routine, controlling the main workflow:
        - def run(self)

    The parent classes provide the default implementations for other routines.

    Args:
        num_clients (int, optional): The number of clients. Defaults to 3.
        num_rounds (int, optional): The total number of training rounds. Defaults to 5.
        persistor_id (str, optional): ID of the persistor component. Defaults to "persistor".
        ignore_result_error (bool, optional): whether this controller can proceed if client result has errors.
            Defaults to False.
        allow_empty_global_weights (bool, optional): whether to allow empty global weights. Some pipelines can have
            empty global weights at first round, such that clients start training from scratch without any global info.
            Defaults to False.
        task_check_period (float, optional): interval for checking status of tasks. Defaults to 0.5.
        persist_every_n_rounds (int, optional): persist the global model every n rounds. Defaults to 1.
            If n is 0 then no persist.
        frac: Fraction of the number of clients used to determine the parents selection parameter. Sets popsize.
            Defaults to 1.
        sigma: initial standard deviation. Defaults to 1.
        intrinsic_dim: intrinsic dimimension of the initial solution. Defaults to 500.
        seed: Seed for CMAEvolutionStrategy. Defaults to 42.
        bound: bounds set for CMAEvolutionStrategy solutions if non-zero. Defaults to 0, i.e. no bounds used.
    """

    def __init__(self, *args, frac=1, sigma=1, intrinsic_dim=500, seed=42, bound=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.frac = frac
        self.seed = seed
        self.sigma = sigma
        self.intrinsic_dim = intrinsic_dim
        self.bound = bound

    def run(self) -> None:
        local_cma_mu = 0.0

        m = max(int(self.frac * self.num_clients), 1)

        self.info("Start FedBPT.")
        cma_opts = {
            "seed": self.seed,
            "popsize": m,
            "maxiter": self.num_rounds,  # args.epochs,
            "verbose": -1,
            "CMA_mu": m,
        }
        self.info(f"Start GlobalES with {cma_opts}")
        if self.bound > 0:
            cma_opts["bounds"] = [-1 * self.bound, 1 * self.bound]
        global_es = cma.CMAEvolutionStrategy(self.intrinsic_dim * [0], self.sigma, inopts=cma_opts)

        local_sigma_current = global_es.sigma

        client_prompt_dict = {}
        for c in range(self.num_clients):
            client_prompt_dict[c] = [copy.deepcopy(global_es.mean)]
        server_prompts = [copy.deepcopy(global_es.mean)]

        # best_test_acc = 0
        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            global_solutions = []
            global_fitnesses = []
            client_sigma_list = []

            self.info(f"Round {self.current_round} started.")

            clients = self.sample_clients(self.num_clients)

            global_model = FLModel(params={"global_es": global_es}, current_round=self.current_round)
            results = self.send_model_and_wait(targets=clients, data=global_model)

            # get solutions from clients
            for result in results:
                global_solutions.append(result.params["solutions"])
                global_fitnesses.append(result.params["fitnesses"])
                client_sigma_list.append(np.sum(np.array(result.params["local_sigmas"]) ** 2))
                local_cma_mu = result.params["local_cma_mu"]

            # Global update
            global_solutions = np.concatenate(global_solutions, axis=0)
            global_fitnesses = np.concatenate(global_fitnesses)
            self.info(f"Received {len(global_solutions)} solutions and {len(global_fitnesses)} fitnesses from clients")
            if len(global_solutions) != len(global_fitnesses):
                raise ValueError(
                    f"Mismatch between {len(global_solutions)} solutions and {len(global_fitnesses)} fitnesses!"
                )

            # calculate global sigma
            global_sigma = np.sqrt(np.sum(np.array(client_sigma_list)) / m / local_cma_mu)

            global_es.sigma = global_sigma
            self.info(f"Check sigma before: {global_es.sigma}")
            global_sigma_old = global_es.sigma

            global_es.ask()
            global_es.tell(global_solutions, global_fitnesses)

            server_prompts.append(copy.deepcopy(global_es.mean))

            self.info(f"Check sigma after: {global_es.sigma}")
            global_sigma_new = global_es.sigma

            # set local sigma
            global_es.sigma = global_sigma_new / global_sigma_old * local_sigma_current

            local_sigma_current = global_es.sigma

            if global_es.sigma < 0.5:
                global_es.sigma = 0.5
                self.info("Set sigma local: 0.5")
            if global_es.sigma > local_sigma_current:
                global_es.sigma = local_sigma_current
                self.info("Set sigma local: not change")

            self.info(f"Check sigma local: {global_es.sigma}")

        self.info("Finished FedBPT.")
