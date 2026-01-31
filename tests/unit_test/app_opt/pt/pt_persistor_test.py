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

"""Unit tests for PTFileModelPersistor source_ckpt_file_full_name support."""

import pytest


class TestPTFileModelPersistorInit:
    """Tests for PTFileModelPersistor initialization with source_ckpt_file_full_name."""

    def test_init_without_source_ckpt(self):
        """Init without source_ckpt should work."""
        try:
            import torch.nn as nn

            from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

            class SimpleNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 2)

            model = SimpleNet()
            persistor = PTFileModelPersistor(model=model)

            assert persistor.source_ckpt_file_full_name is None
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_init_with_source_ckpt_no_existence_check(self):
        """Init with non-existent source_ckpt should NOT raise error (deferred to runtime)."""
        try:
            import torch.nn as nn

            from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

            class SimpleNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 2)

            model = SimpleNet()
            # This path doesn't exist, but should NOT raise error at init time
            persistor = PTFileModelPersistor(
                model=model,
                source_ckpt_file_full_name="/nonexistent/path/model.pt",
            )

            assert persistor.source_ckpt_file_full_name == "/nonexistent/path/model.pt"
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_init_with_source_ckpt_stores_path(self):
        """Init should store the source_ckpt path."""
        try:
            import torch.nn as nn

            from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

            class SimpleNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 2)

            model = SimpleNet()
            persistor = PTFileModelPersistor(
                model=model,
                source_ckpt_file_full_name="/data/pretrained/model.pt",
            )

            assert persistor.source_ckpt_file_full_name == "/data/pretrained/model.pt"
        except ImportError:
            pytest.skip("PyTorch not installed")
