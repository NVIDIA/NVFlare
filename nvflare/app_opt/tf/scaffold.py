# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import tensorflow as tf

from .utils import flat_layer_weights_dict

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

tf.debugging.enable_check_numerics()


def get_lr_values(optimizer):
    """
    This function is used to get the learning rates of the optimizer.
    """
    return optimizer.learning_rate


class TFScaffoldHelper(object):
    """Helper to be used with SCAFFOLD components."""

    def __init__(self):
        self.cnt = 0
        self.c_global = None
        self.c_local = None
        self.c_delta_para = None
        self.global_keys = None

    # self.clip_norm = 1.0

    def init(self, model):
        self.c_global = tf.keras.models.clone_model(model)
        self.c_local = tf.keras.models.clone_model(model)
        # Initialize correction term with zeros
        c_init_para = {v.name: np.zeros_like(v.numpy()) for v in model.variables}
        self.c_global.set_weights([c_init_para[k] for k in c_init_para])
        self.c_local.set_weights([c_init_para[k] for k in c_init_para])

        # Generate a list of the flattened layers
        layer_weights_dict = {layer.name: layer.get_weights() for layer in self.c_global.layers}
        flattened_layer_weights_dict = flat_layer_weights_dict(layer_weights_dict)
        self.global_keys = [key for key, _ in flattened_layer_weights_dict.items()]
        print("Gloabl")
        print(self.global_keys)

    def get_params(self):
        self.cnt = 0
        c_global_para = self.c_global.variables
        c_local_para = self.c_local.variables
        return c_global_para, c_local_para

    def model_update(self, model, curr_lr, c_global_para, c_local_para):
        net_para = model.variables  # Access only trainable variables
        trainable_var_names = [var.name for var in model.trainable_variables]
        model_difference = tf.nest.map_structure(
            lambda a, b: tf.multiply(curr_lr, a - b),
            c_global_para,
            c_local_para,
        )
        new_weights = tf.nest.map_structure(lambda a, b: a - b, net_para, model_difference)
        # print('the length of the weights are:',(new_weights))
        for var, new_weight in zip(net_para, new_weights):
            if var.name in trainable_var_names:
                var.assign(new_weight)

        self.cnt += 1

    def terms_update(
        self,
        model,
        curr_lr,
        c_global_para,
        c_local_para,
        model_global,
    ):
        c_new_para = self.c_local.variables
        self.c_delta_para = copy.deepcopy(self.c_local.variables)
        global_model_para = model_global.variables
        net_para = model.variables
        scaler = 1 / (self.cnt * curr_lr)

        c_new_para = tf.nest.map_structure(
            lambda a, b, c, d: (a - b) + tf.multiply(scaler, c - d),
            c_new_para,
            c_global_para,
            global_model_para,
            net_para,
        )

        # if tf.less(tf.constant(0, tf.float32), self.clip_norm):
        #    flatten_weights_delta = tf.nest.flatten(c_new_para)
        #   clipped_flatten_weights_delta, _ = tf.clip_by_global_norm(
        #         flatten_weights_delta, self.clip_norm)
        #  c_new_para = tf.nest.pack_sequence_as(c_new_para,
        #          clipped_flatten_weights_delta)

        c_delta_para_value_new = tf.nest.map_structure(lambda a, b: (a - b), c_new_para, c_local_para)

        self.c_delta_para = {var.name: c_delta_para_value_new[i].numpy() for i, var in enumerate(net_para)}

        self.c_local.set_weights(c_new_para)

    def load_global_controls(self, weights):
        weights_values = [v for _, v in weights.items()]
        self.c_global.set_weights(weights_values)

    def get_delta_controls(self):
        if self.c_delta_para is None:
            raise ValueError("c_delta_para hasn't been computed yet!")

        print(type(self.c_delta_para))

        # print(self.c_delta_para)

        c_delta_para_new = {self.global_keys[i]: value for i, (key, value) in enumerate(self.c_delta_para.items())}

        self.c_delta_para = c_delta_para_new

        return self.c_delta_para


class ScaffoldCallback(tf.keras.callbacks.Callback):
    def __init__(self, scaffold_helper):
        super(ScaffoldCallback, self).__init__()
        self.scaffold_helper = scaffold_helper
        self.c_global_para, self.c_local_para = self.scaffold_helper.get_params()

    def on_epoch_end(self, epoch, logs=None):
        curr_lr = self.model.optimizer.learning_rate
        self.scaffold_helper.model_update(self.model, curr_lr, self.c_global_para, self.c_local_para)
        print(f"SCAFFOLD model updated at end of epoch {epoch + 1}")
