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

from typing import Optional

import tensorflow as tf

from nvflare.app_opt.tf.model_persistor import TFModelPersistor


class TFModelPersistorArgs:
    def __init__(self, save_name="tf_model.weights.h5", filter_id: str = None):
        self.save_name = save_name
        self.filter_id = filter_id


class TFModel:
    def __init__(self, model, model_persistor_args: Optional[TFModelPersistorArgs] = None):
        """TensorFlow model wrapper.

        If model is a tf.keras.Model, add a TFModelPersistor with the model.

        Args:
            model (any): model
            model_persistor_args (TFModelPersistorArgs): args for TFModelPersistor
        """
        self.model = model
        self.model_persistor_args = model_persistor_args if model_persistor_args else TFModelPersistorArgs()

    def add_to_fed_job(self, job, ctx):
        """This method is used by Job API.

        Args:
            job: the Job object to add to
            ctx: Job Context

        Returns:
            dictionary of ids of component added
        """
        if isinstance(self.model, tf.keras.Model):  # if model, create a TF persistor
            persistor = TFModelPersistor(
                model=self.model,
                save_name=self.model_persistor_args.save_name,
                filter_id=self.model_persistor_args.filter_id,
            )
            persistor_id = job.add_component(comp_id="persistor", obj=persistor, ctx=ctx)
            return persistor_id
        else:
            raise ValueError(f"Unsupported type for model: {type(self.model)}.")
