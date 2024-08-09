# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import tensorflow as tf

from .utils import flat_layer_weights_dict

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

tf.debugging.enable_check_numerics()


def optimize_weights(model, c_delta_para_value):
    """
    Efficiently assigns weights to `self.c_delta_para` based on trainability or non tainability.

    Args:
        model: The TensorFlow model containing layers.
        c_delta_para_value: Delta values for trainable variables.

    Returns:
        None. Modifies the `self.c_delta_para` attribute in-place.
    """
    c_delta_para = {}
    layer_weights_dict = {layer.name: layer.get_weights() for layer in model.layers}
    trainable_layers_dict = {layer.name: layer.trainable_weights for layer in model.layers if layer.trainable_weights}
    flatten_layer_weights_dict = flat_layer_weights_dict(layer_weights_dict)
    flatten_layer_weights_dict = flat_layer_weights_dict(layer_weights_dict)
    flatten_trainable_layers_dict = flat_layer_weights_dict(trainable_layers_dict)

    trainable_layer_names = list(flatten_trainable_layers_dict.keys())
    j = 0
    for layer_name, weight in flatten_layer_weights_dict.items():
        if layer_name in trainable_layer_names:
            c_delta_para[layer_name] = c_delta_para_value[j].numpy()
            j += 1
        else:
            c_delta_para[layer_name] = weight

    return c_delta_para


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

    def init(self, model):
        self.c_global = tf.keras.models.clone_model(model)
        self.c_local = tf.keras.models.clone_model(model)
        # Initialize correction term with zeros
        c_init_para = [np.zeros(shape) for shape in [w.shape for w in model.get_weights()]]
        self.c_global.set_weights(c_init_para)
        self.c_local.set_weights(c_init_para)

    def get_params(self):
        self.cnt = 0
        c_global_para = self.c_global.trainable_variables
        c_local_para = self.c_local.trainable_variables
        return c_global_para, c_local_para

    def model_update(self, model, curr_lr, c_global_para, c_local_para):
        net_para = model.trainable_variables  # Access only trainable trainable_variables
        trainable_var_names = [var.name for var in model.trainable_variables]
        model_difference = tf.nest.map_structure(
            lambda a, b: tf.multiply(tf.cast(curr_lr, a.dtype), a - b),
            c_global_para,
            c_local_para,
        )
        new_weights = tf.nest.map_structure(lambda a, b: a - b, net_para, model_difference)
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
        c_new_para = self.c_local.trainable_variables
        global_model_para = model_global.trainable_variables
        net_para = model.trainable_variables
        scaler = 1 / (self.cnt * curr_lr)

        c_new_para_c_global = tf.nest.map_structure(lambda a, b: a - b, c_new_para, c_global_para)

        global_model_para_net_para = tf.nest.map_structure(
            lambda a, b: tf.multiply(tf.cast(scaler, a.dtype), a - b),
            global_model_para,
            net_para,
        )

        c_new_para = tf.nest.map_structure(
            lambda a, b: a + b,
            c_new_para_c_global,
            global_model_para_net_para,
        )

        c_delta_para_value = tf.nest.map_structure(lambda a, b: a - b, c_new_para, c_local_para)
        self.c_delta_para = optimize_weights(model, c_delta_para_value)

        for var, new_weight in zip(self.c_local.trainable_variables, c_new_para):
            var.assign(new_weight)

    def load_global_controls(self, weights):
        weights_values = [v for _, v in weights.items()]
        self.c_global.set_weights(weights_values)

    def get_delta_controls(self):
        if self.c_delta_para is None:
            raise ValueError("c_delta_para hasn't been computed yet!")

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
