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


import tensorflow as tf


class TFFedProxLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        local_model_weights: tf.Tensor,
        global_model_weights: tf.Tensor,
        mu: float,
        loss_fun: tf.keras.losses.Loss,
        *args,
        **kwargs,
    ):
        """
        Initialize the TFFedProxLoss class.

        Parameters:
        local_model_weights: The local model trainable weights.
        global_model_weights: The global model trainable weights.
        mu: The regularization parameter for the FedProx term.
        loss_fun: The base loss function to be used
        (e.g., tf.keras.losses.SparseCategoricalCrossentropy).

        Returns:
        A loss function that includes the FedProx regularization term.
        """
        super().__init__(*args, **kwargs)
        self.local_model_weights = local_model_weights
        self.global_model_weights = global_model_weights
        self.mu = mu
        self.loss_fun = loss_fun

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate the FedProx loss.

        Parameters:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

        Returns:
        The loss value including the FedProx regularization term.
        """
        original_loss = self.loss_fun(y_true, y_pred)
        fedprox_term = (self.mu / 2) * self.model_difference_norm(self.global_model_weights, self.local_model_weights)
        return original_loss + fedprox_term

    def model_difference_norm(self, global_model: tf.Tensor, local_model: tf.Tensor) -> tf.Tensor:
        """
        Calculate the squared l2 norm of the model difference.

        Parameters:
        global_model: The trainable variables of the global model
        local_model: The trainable variables of the local model.

        Returns:
        The squared norm of the difference between the global and local models.
        """
        model_difference = tf.nest.map_structure(lambda a, b: a - b, local_model, global_model)
        squared_norm = tf.reduce_sum([tf.reduce_sum(tf.square(diff)) for diff in model_difference])
        return squared_norm
