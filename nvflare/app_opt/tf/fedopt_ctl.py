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
from typing import Union, Dict

import tensorflow as tf

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.security.logging import secure_format_exception


class FedOpt(FedAvg):
    def __init__(
        self,
        *args,
        #source_model: Union[str, tf.keras.models.Model],
        optimizer_args: dict = {
            "path": "tensorflow.keras.optimizers.SGD",
            "args": {"learning_rate": 1.0, "momentum": 0.6},
        },
        #lr_scheduler_args: dict = {
        #    "path": "torch.optim.lr_scheduler.CosineAnnealingLR",
        #    "args": {"T_max": 3, "eta_min": 0.9},
        #},
        **kwargs,
    ):
        """Implement the FedOpt algorithm. Based on FedAvg ModelController.

        The algorithm is proposed in Reddi, Sashank, et al. "Adaptive federated optimization." arXiv preprint arXiv:2003.00295 (2020).
        After each round, update the global model using the specified PyTorch optimizer and learning rate scheduler.
        Note: This class will use FedOpt to optimize the global trainable parameters (i.e. `self.keras_model.named_parameters()`)
        but use FedAvg to update any other layers such as batch norm statistics.

        Args:
            source_model: component id of torch model object or a valid Keras model object
            optimizer_args: dictionary of optimizer arguments, with keys of 'optimizer_path' and 'args.
            lr_scheduler_args: dictionary of server-side learning rate scheduler arguments, with keys of 'lr_scheduler_path' and 'args.

        Raises:
            TypeError: when any of input arguments does not have correct type
        """
        super().__init__(*args, **kwargs)

        #self.source_model = source_model
        self.optimizer_args = optimizer_args
        #self.lr_scheduler_args = lr_scheduler_args

        self.keras_model = None
        self.optimizer = None
        self.lr_scheduler = None

    def run(self):
        # set up source model
        """
        if isinstance(self.source_model, str):
            self.keras_model = self.get_component(self.source_model)
        else:
            self.keras_model = self.source_model

        if self.keras_model is None:
            self.panic("Model is not available")
            return
        elif not isinstance(self.keras_model, tf.keras.models.Model):
            self.panic(f"expect model to be tf.keras.models.Model but got {type(self.keras_model)}")
            return
        else:
            print("server model", self.keras_model)
        self.keras_model.to(self.device)
        """

        # set up optimizer
        try:
            if "args" not in self.optimizer_args:
                self.optimizer_args["args"] = {}
            #self.optimizer_args["args"]["params"] = self.keras_model.parameters()
            self.optimizer = self.build_component(self.optimizer_args)
        except Exception as e:
            error_msg = f"Exception while constructing optimizer: {secure_format_exception(e)}"
            self.exception(error_msg)
            self.panic(error_msg)
            return

        # set up lr scheduler
        """
        try:
            if "args" not in self.lr_scheduler_args:
                self.lr_scheduler_args["args"] = {}
            self.lr_scheduler_args["args"]["optimizer"] = self.optimizer
            self.lr_scheduler = self.build_component(self.lr_scheduler_args)
        except Exception as e:
            error_msg = f"Exception while constructing lr_scheduler: {secure_format_exception(e)}"
            self.exception(error_msg)
            self.panic(error_msg)
            return
        """
        super().run()

    def _to_tf_params_list(self, params: Dict, negate: bool = False):
        tf_params_list = []
        for k, v in params.items():
            if negate:
                v = -1 * v
            tf_params_list.append(tf.Variable(v))
        return tf_params_list

    def update_model(self, global_model: FLModel, aggr_result: FLModel):
        optim_cnt = tf.Variable(global_model.current_round)
        model_diff = self._to_tf_params_list(aggr_result.params, negate=True)
        global_params = self._to_tf_params_list(global_model.params)

        # set current global model weights
        #if not self.optimizer.built:
        #    self.optimizer.build([optim_cnt, self.optimizer.learning_rate, tf.Variable(self.optimizer.momentum)] + global_params)
        #self.optimizer.set_weights([optim_cnt, self.optimizer.learning_rate, tf.Variable(self.optimizer.momentum)] + global_params)

        start = time.time()
        #weights, updated_params = self.optimizer_update(model_diff)
        self.optimizer.apply_gradients(zip(model_diff, global_params))
        secs = time.time() - start

        # convert to numpy dict of weights
        start = time.time()
        weights = self.optimizer.variables
        w_idx = 0
        new_weights = {}
        for key in global_model.params:
            w = weights[w_idx].numpy()
            while global_model.params[key].shape != w.shape:
                w_idx += 1
                w = weights[w_idx].numpy()
            new_weights[key] = w
        secs_detach = time.time() - start

        # update unnamed parameters such as batch norm layers if there are any using the averaged update
        #n_fedavg = 0
        #for key, value in model_diff.items():
        #    if key not in updated_params:
        #        weights[key] = global_model.params[key] + value
        #        n_fedavg += 1

        self.info(
            f"FedOpt ({type(self.optimizer)}) server model update "
            f"round {self.current_round}, "
            f"{type(self.lr_scheduler)} "
            f"lr: {self.optimizer.learning_rate}, "
            #f"fedopt layers: {len(updated_params)}, "
            #f"fedavg layers: {n_fedavg}, "
            f"update: {secs} secs., detach: {secs_detach} secs.",
        )

        global_model.params = new_weights
        global_model.meta = aggr_result.meta

        # Rebuild optimizer
        self.optimizer = self.build_component(self.optimizer_args)

        return global_model
