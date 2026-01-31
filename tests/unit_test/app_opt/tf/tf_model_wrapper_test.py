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

"""Unit tests for TFModel dict config and initial_ckpt support."""

import pytest


class TestTFModelInit:
    """Tests for TFModel initialization."""

    def test_init_with_model(self):
        """Init with tf.keras.Model should work."""
        try:
            import tensorflow as tf

            from nvflare.app_opt.tf.job_config.model import TFModel

            model = tf.keras.Sequential([tf.keras.layers.Dense(2)])
            tf_model = TFModel(model=model)

            assert tf_model.model is model
            assert tf_model.initial_ckpt is None
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_init_with_dict_config(self):
        """Init with dict config should parse correctly."""
        try:
            from nvflare.app_opt.tf.job_config.model import TFModel

            model_dict = {"path": "my_module.models.Net", "args": {"num_classes": 10}}
            tf_model = TFModel(model=model_dict)

            assert tf_model.model == model_dict
            assert tf_model.model_class_path == "my_module.models.Net"
            assert tf_model.model_args == {"num_classes": 10}
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_init_dict_config_missing_path_raises(self):
        """Dict config without 'path' key should raise."""
        try:
            from nvflare.app_opt.tf.job_config.model import TFModel

            with pytest.raises(ValueError, match="must have 'path' key"):
                TFModel(model={"args": {}})
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_init_with_initial_ckpt_only(self):
        """Init with initial_ckpt only should work (TF can load full model)."""
        try:
            from nvflare.app_opt.tf.job_config.model import TFModel

            tf_model = TFModel(model=None, initial_ckpt="/data/pretrained.h5")

            assert tf_model.model is None
            assert tf_model.initial_ckpt == "/data/pretrained.h5"
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_init_with_model_and_ckpt(self):
        """Init with both model and initial_ckpt should work."""
        try:
            import tensorflow as tf

            from nvflare.app_opt.tf.job_config.model import TFModel

            model = tf.keras.Sequential([tf.keras.layers.Dense(2)])
            tf_model = TFModel(model=model, initial_ckpt="/data/pretrained.h5")

            assert tf_model.model is model
            assert tf_model.initial_ckpt == "/data/pretrained.h5"
        except ImportError:
            pytest.skip("TensorFlow not installed")
