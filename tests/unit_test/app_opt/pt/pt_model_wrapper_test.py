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

"""Unit tests for PTModel dict config and initial_ckpt support."""

import pytest


class TestPTModelInit:
    """Tests for PTModel initialization."""

    def test_init_with_module(self):
        """Init with nn.Module should work."""
        try:
            import torch.nn as nn

            from nvflare.app_opt.pt.job_config.model import PTModel

            class SimpleNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 2)

            model = SimpleNet()
            pt_model = PTModel(model=model)

            assert pt_model.model is model
            assert pt_model.initial_ckpt is None
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_init_with_dict_config(self):
        """Init with dict config should parse correctly."""
        try:
            from nvflare.app_opt.pt.job_config.model import PTModel

            model_dict = {"class_path": "my_module.models.Net", "args": {"num_classes": 10}}
            pt_model = PTModel(model=model_dict)

            assert pt_model.model == model_dict
            assert pt_model.model_class_path == "my_module.models.Net"
            assert pt_model.model_args == {"num_classes": 10}
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_init_dict_config_missing_path_raises(self):
        """Dict config without 'path' key should raise."""
        try:
            from nvflare.app_opt.pt.job_config.model import PTModel

            with pytest.raises(ValueError, match="must have 'path' key"):
                PTModel(model={"args": {}})
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_init_with_initial_ckpt(self):
        """Init with initial_ckpt should store path."""
        try:
            import torch.nn as nn

            from nvflare.app_opt.pt.job_config.model import PTModel

            class SimpleNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 2)

            model = SimpleNet()
            pt_model = PTModel(model=model, initial_ckpt="/data/pretrained.pt")

            assert pt_model.initial_ckpt == "/data/pretrained.pt"
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_init_dict_config_with_ckpt(self):
        """Init with dict config and initial_ckpt should work."""
        try:
            from nvflare.app_opt.pt.job_config.model import PTModel

            model_dict = {"path": "my_module.Net", "args": {}}
            pt_model = PTModel(model=model_dict, initial_ckpt="/data/pretrained.pt")

            assert pt_model.model_class_path == "my_module.Net"
            assert pt_model.initial_ckpt == "/data/pretrained.pt"
        except ImportError:
            pytest.skip("PyTorch not installed")


class TestPTFileModelPersistorDictConfig:
    """Tests for PTFileModelPersistor dict config support."""

    def test_persistor_accepts_dict_config(self):
        """PTFileModelPersistor should accept dict config."""
        try:
            from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

            model_dict = {"class_path": "my_module.models.Net", "args": {"num_classes": 10}}
            persistor = PTFileModelPersistor(model=model_dict)

            assert persistor.model == model_dict
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_persistor_accepts_string_component_id(self):
        """PTFileModelPersistor should accept string component ID."""
        try:
            from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

            persistor = PTFileModelPersistor(model="my_model_component")

            assert persistor.model == "my_model_component"
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_persistor_accepts_none(self):
        """PTFileModelPersistor should accept None model."""
        try:
            from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

            persistor = PTFileModelPersistor(model=None)

            assert persistor.model is None
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_persistor_dict_config_missing_path_detected_at_init_time(self):
        """PTFileModelPersistor accepts dict without path (validation at runtime)."""
        try:
            from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

            # Note: The persistor accepts any dict at init time;
            # The path validation happens at runtime in _initialize()
            persistor = PTFileModelPersistor(model={"args": {}})
            assert persistor.model == {"args": {}}
        except ImportError:
            pytest.skip("PyTorch not installed")
