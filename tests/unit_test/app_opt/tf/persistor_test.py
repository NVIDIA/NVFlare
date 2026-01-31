# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for TFModelPersistor source_ckpt_file_full_name support."""

import pytest


class TestTFModelPersistorInit:
    """Tests for TFModelPersistor initialization with source_ckpt_file_full_name."""

    def test_init_without_source_ckpt(self):
        """Init without source_ckpt should work."""
        try:
            import tensorflow as tf

            from nvflare.app_opt.tf.model_persistor import TFModelPersistor

            model = tf.keras.Sequential([tf.keras.layers.Dense(2)])
            persistor = TFModelPersistor(model=model)

            assert persistor.source_ckpt_file_full_name is None
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_init_with_source_ckpt_stores_path(self):
        """Init should store the source_ckpt path."""
        try:
            import tensorflow as tf

            from nvflare.app_opt.tf.model_persistor import TFModelPersistor

            model = tf.keras.Sequential([tf.keras.layers.Dense(2)])
            persistor = TFModelPersistor(
                model=model,
                source_ckpt_file_full_name="/data/pretrained/model.h5",
            )

            assert persistor.source_ckpt_file_full_name == "/data/pretrained/model.h5"
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_init_without_model_with_ckpt(self):
        """Init without model but with source_ckpt should work (TF can load full model)."""
        try:
            from nvflare.app_opt.tf.model_persistor import TFModelPersistor

            persistor = TFModelPersistor(
                model=None,
                source_ckpt_file_full_name="/data/pretrained/model.h5",
            )

            assert persistor.model is None
            assert persistor.source_ckpt_file_full_name == "/data/pretrained/model.h5"
        except ImportError:
            pytest.skip("TensorFlow not installed")
