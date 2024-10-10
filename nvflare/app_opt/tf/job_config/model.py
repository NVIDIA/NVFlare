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

from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_opt.tf.model_persistor import TFModelPersistor
from nvflare.job_config.api import validate_object_for_job


class TFModel:
    def __init__(self, model, persistor: Optional[ModelPersistor] = None):
        """TensorFlow model wrapper.

        If persistor is provided, use it.
        Else if model is a tf.keras.Model, add a TFModelPersistor with the model.

        Args:
            model (any): model
            persistor (Optional[ModelPersistor]): A ModelPersistor,
                if provided will ignore argument `model`, defaults to None.
        """
        self.model = model

        if persistor:
            validate_object_for_job("persistor", persistor, ModelPersistor)
        self.persistor = persistor

    def add_to_fed_job(self, job, ctx):
        """This method is used by Job API.

        Args:
            job: the Job object to add to
            ctx: Job Context

        Returns:
            dictionary of ids of component added
        """
        if self.persistor:
            persistor = self.persistor
        elif isinstance(self.model, tf.keras.Model):
            # if model is a tf.keras.Model, creates a TFModelPersistor
            persistor = TFModelPersistor(model=self.model)
        else:
            raise ValueError(f"Unsupported type for model: {type(self.model)}.")
        persistor_id = job.add_component(comp_id="persistor", obj=persistor, ctx=ctx)
        return persistor_id
