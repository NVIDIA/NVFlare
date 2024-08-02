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


import time
from typing import Dict

import tensorflow as tf

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.security.logging import secure_format_exception


class FedOpt(FedAvg):
    def __init__(
        self,
        *args,
        optimizer_args: dict = {
            "path": "tensorflow.keras.optimizers.SGD",
            "args": {"learning_rate": 1.0, "momentum": 0.6},
        },
        lr_scheduler_args: dict = {
            "path": "tensorflow.keras.optimizers.schedules.CosineDecay",
            "args": {"initial_learning_rate": 1.0, "decay_steps": None, "alpha": 0.9},
        },
        **kwargs,
    ):
        """Implement the FedOpt algorithm. Based on FedAvg ModelController.

        The algorithm is proposed in Reddi, Sashank, et al. "Adaptive federated optimization." arXiv preprint arXiv:2003.00295 (2020).
        After each round, update the global model's trainable variables using the specified optimizer and learning rate scheduler,
        in this case, SGD with momentum & CosineDecay.

        Args:
            optimizer_args: dictionary of optimizer arguments, with keys of 'optimizer_path' and 'args.
            lr_scheduler_args: dictionary of server-side learning rate scheduler arguments, with keys of 'lr_scheduler_path' and 'args.

        Raises:
            TypeError: when any of input arguments does not have correct type
        """
        super().__init__(*args, **kwargs)

        self.optimizer_args = optimizer_args
        self.lr_scheduler_args = lr_scheduler_args

        # Set "decay_steps" arg to num_rounds
        if lr_scheduler_args["args"]["decay_steps"] is None:
            lr_scheduler_args["args"]["decay_steps"] = self.num_rounds

        self.keras_model = None
        self.optimizer = None
        self.lr_scheduler = None

    def run(self):
        """
        Override run method to add set-up for FedOpt specific optimizer
        and LR scheduler.
        """
        # set up optimizer
        try:
            if "args" not in self.optimizer_args:
                self.optimizer_args["args"] = {}
            self.optimizer = self.build_component(self.optimizer_args)
        except Exception as e:
            error_msg = f"Exception while constructing optimizer: {secure_format_exception(e)}"
            self.exception(error_msg)
            self.panic(error_msg)
            return

        # set up lr scheduler
        try:
            if "args" not in self.lr_scheduler_args:
                self.lr_scheduler_args["args"] = {}
            self.lr_scheduler = self.build_component(self.lr_scheduler_args)
            self.optimizer.learning_rate = self.lr_scheduler
        except Exception as e:
            error_msg = f"Exception while constructing lr_scheduler: {secure_format_exception(e)}"
            self.exception(error_msg)
            self.panic(error_msg)
            return

        super().run()

    def _to_tf_params_list(self, params: Dict, negate: bool = False):
        """
        Convert FLModel params to a list of tf.Variables.
        Optionally negate the values of weights, needed
        to apply gradients.
        """
        tf_params_list = []
        for k, v in params.items():
            if negate:
                v = -1 * v
            tf_params_list.append(tf.Variable(v))
        return tf_params_list

    def update_model(self, global_model: FLModel, aggr_result: FLModel):
        """
        Override the default version of update_model
        to perform update with Keras Optimizer on the
        global model stored in memory in persistor, instead of
        creating new temporary model on-the-fly.

        Creating a new model would not work for Keras
        Optimizers, since an optimizer is bind to
        specific set of Variables.

        """
        # Get the Keras model stored in memory in persistor.
        global_model_tf = self.persistor.model
        global_params = global_model_tf.trainable_weights

        # Compute model diff: need to use model diffs as
        # gradients to be applied by the optimizer.
        model_diff_params = {k: aggr_result.params[k] - global_model.params[k] for k in global_model.params}
        model_diff = self._to_tf_params_list(model_diff_params, negate=True)

        # Apply model diffs as gradients, using the optimizer.
        start = time.time()
        self.optimizer.apply_gradients(zip(model_diff, global_params))
        secs = time.time() - start

        # Convert updated global model weights to
        # numpy format for FLModel.
        start = time.time()
        weights = global_model_tf.get_weights()
        w_idx = 0
        new_weights = {}
        for key in global_model.params:
            w = weights[w_idx]
            while global_model.params[key].shape != w.shape:
                w_idx += 1
                w = weights[w_idx]
            new_weights[key] = w
        secs_detach = time.time() - start

        self.info(
            f"FedOpt ({type(self.optimizer)}) server model update "
            f"round {self.current_round}, "
            f"{type(self.lr_scheduler)} "
            f"lr: {self.optimizer.learning_rate}, "
            f"update: {secs} secs., detach: {secs_detach} secs.",
        )

        global_model.params = new_weights
        global_model.meta = aggr_result.meta

        return global_model
