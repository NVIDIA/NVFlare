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

import tensorflow as tf

from nvflare.app_opt.tf.model_persistor import TFModelPersistor


class TFModel:
    def __init__(self, model):
        """TensorFLow model wrapper.

        If model is a tf.keras.Model, add a TFModelPersistor with the model.

        Args:
            model (any): model
        """
        self.model = model

    def add_to_fed_job(self, job, ctx):
        """This method is used by Job API.

        Args:
            job: the Job object to add to
            ctx: Job Context

        Returns:
            dictionary of ids of component added
        """
        if isinstance(self.model, tf.keras.Model):  # if model, create a TF persistor
            persistor = TFModelPersistor(model=self.model)
            persistor_id = job.add_component(comp_id="persistor", obj=persistor, ctx=ctx)
            return persistor_id
        else:
            raise ValueError(
                f"Unable to add {self.model} to job with TFModelPersistor. Expected tf.keras.Model but got {type(self.model)}."
            )
