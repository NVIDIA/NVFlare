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
"""
   Federated Averaging for Logistic Regression with Newton-Raphson method
   using Numpy
"""
from typing import List, Optional

import numpy as np

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.np.constants import NPConstants
from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.app_common.workflows.base_fedavg import BaseFedAvg
from nvflare.app_common.workflows.lr.np_persistor import LRModelPersistor


class FedAvgLR(BaseFedAvg):
    def __init__(
        self,
        damping_factor: float,
        epsilon: float = 1.0,
        model_dir: str = "models",
        model_name: str = "weights.npy",
        n_features: int = 13,
        aggregator: WeightedAggregationHelper = WeightedAggregationHelper(),
        persistor: Optional[NPModelPersistor] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the FedAvgLR class for Federated Averaging with Newton-Raphson optimization.

        Args:
            damping_factor (float): Damping factor for Newton-Raphson updates, used to control the step size.
            epsilon (float, optional): Regularization factor to avoid empty Hessian matrix inversion. Defaults to 1.0.
            model_dir (str, optional): Directory to save and load the model. Defaults to "models".
            model_name (str, optional): Name of the model file. Defaults to "weights.npy".
            n_features (int, optional): Number of features in the dataset. Defaults to 13.
            aggregator (WeightedAggregationHelper, optional): Helper for weighted aggregation of model updates.
            persistor (Optional[NPModelPersistor], optional): Custom persistor for model saving and loading. If not provided, a default NewtonRaphsonModelPersistor is used.
            *args: Additional positional arguments passed to the base class.
            **kwargs: Additional keyword arguments passed to the base class.

        This class implements the Federated Averaging algorithm using the Newton-Raphson method for optimization.
        It supports flexible model persistence through customizable persistors, allowing integration with different
        storage backends or model formats.
        """
        super().__init__(*args, **kwargs)
        self.damping_factor = damping_factor
        self.epsilon = epsilon
        self.model_dir = model_dir
        self.model_name = model_name
        self.n_features = n_features
        self.aggregator = aggregator
        self._default_persistor = LRModelPersistor(
            model_dir=self.model_dir, model_name=self.model_name, n_features=self.n_features
        )
        self.persistor = persistor

    def run(self) -> None:
        """
        The run function executes the logic of federated
        second order Newton-Raphson optimization.

        """
        self.info("starting Federated Averaging Newton Raphson ...")

        # First load the model and set up some training params.
        # A `persisitor` (NewtonRaphsonModelPersistor) will load
        # the model in `ModelLearnable` format, then will be
        # converted `FLModel` by `ModelController`.
        #
        model = self.load_model()

        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        self.info("Server side model loader: {}".format(model))

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            self.info(f"Round {self.current_round} started.")

            # Get the list of clients.
            clients = self.sample_clients(self.num_clients)

            model.current_round = self.current_round

            # Send training task and current global model to clients.
            #
            # A `task` instance will be created, and sent
            # to clients, the model is first converted to a shareable
            # and is attached to the task.
            #
            # After the task is finished, the result (shareable) received
            # from the task is converted to FLModel, and is returned to the
            # server. The `results` below is a list with result (FLModel)
            # from all clients.
            #
            # The full logic of `task` is implemented in:
            # https://github.com/NVIDIA/NVFlare/blob/d6827bca96d332adb3402ceceb4b67e876146067/nvflare/app_common/workflows/model_controller.py#L178
            #
            self.info("sending server side global model to clients")
            results = self.send_model_and_wait(targets=clients, data=model)

            # Aggregate results received from clients.
            aggregate_results = self.aggregate(results, aggregate_fn=self.newton_raphson_aggregator_fn)

            # Update global model based on the following formula:
            # weights = weights + updates, where
            # updates = -damping_factor * Hessian^{-1} . Gradient
            self.update_model(model, aggregate_results)

            # Save global model.
            self.save_model(model)

        self.info("Finished FedAvg.")

    def newton_raphson_aggregator_fn(self, results: List[FLModel]):
        """
        Custom aggregator function for second order Newton-Raphson
        optimization.

        This uses the default thread-safe WeightedAggregationHelper,
        which implement a weighted average of all values received from
        a `result` dictionary.

        Args:
            results: a list of `FLModel`s. Each `FLModel` is received
                from a client. The field `params` is a dictionary that
                contains values to be aggregated: the gradient and hessian.
        """
        self.info("received results from clients: {}".format(results))

        # On client side the `NUM_STEPS_CURRENT_ROUND` key
        # is used to track the number of samples for each client.
        for curr_result in results:
            self.aggregator.add(
                data=curr_result.params,
                weight=curr_result.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0),
                contributor_name=curr_result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN),
                contribution_round=curr_result.current_round,
            )

        aggregated_dict = self.aggregator.get_result()
        # self.info("aggregated result: {}".format(aggregated_dict))

        # Compute global model update:
        # update = - damping_factor * Hessian^{-1} . Gradient
        # A regularization is added to avoid empty hessian.
        #
        reg = self.epsilon * np.eye(aggregated_dict["hessian"].shape[0])
        newton_raphson_updates = self.damping_factor * np.linalg.solve(
            aggregated_dict["hessian"] + reg, aggregated_dict["gradient"]
        )
        self.info("Newton-Raphson updates: {}".format(newton_raphson_updates))

        # Convert the aggregated result to `FLModel`, this `FLModel`
        # will then be used by `update_model` method from the base class,
        # to update the global model weights.
        #
        aggr_result = FLModel(
            params={"newton_raphson_updates": newton_raphson_updates},
            params_type=results[0].params_type,
            meta={
                "nr_aggregated": len(results),
                AppConstants.CURRENT_ROUND: results[0].current_round,
                AppConstants.NUM_ROUNDS: self.num_rounds,
            },
        )
        return aggr_result

    def update_model(self, model, model_update, replace_meta=True) -> FLModel:
        """
        Update logistic regression parameters based on
        aggregated gradient and hessian.

        """
        if model_update is None:
            return model
        if model_update.params is None:
            raise ValueError("model params is None")

        if replace_meta:
            model.meta = model_update.meta
        else:
            model.meta.update(model_update.meta)

        model.metrics = model_update.metrics
        model.params[NPConstants.NUMPY_KEY] += model_update.params["newton_raphson_updates"]

        return model

    def load_model(self) -> FLModel:
        if self.persistor is None:
            self.persistor = self._default_persistor

        model = super().load_model()
        if model is None:
            raise ValueError("model can't be None")

        if model.params is None:
            raise ValueError("model params is None")

        return model

    def save_model(self, model: FLModel) -> None:
        if self.persistor is None:
            self.persistor = self._default_persistor

        super().save_model(model)
