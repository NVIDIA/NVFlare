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

"""Unit tests for NPModelPersistor source_ckpt_file_full_name support."""


class TestNPModelPersistorInit:
    """Tests for NPModelPersistor initialization with source_ckpt_file_full_name."""

    def test_init_without_source_ckpt(self):
        """Init without source_ckpt should work."""
        from nvflare.app_common.np.np_model_persistor import NPModelPersistor

        persistor = NPModelPersistor()

        assert persistor.source_ckpt_file_full_name is None

    def test_init_with_source_ckpt_stores_path(self):
        """Init should store the source_ckpt path."""
        from nvflare.app_common.np.np_model_persistor import NPModelPersistor

        persistor = NPModelPersistor(
            source_ckpt_file_full_name="/data/pretrained/model.npy",
        )

        assert persistor.source_ckpt_file_full_name == "/data/pretrained/model.npy"

    def test_init_with_initial_model_and_ckpt(self):
        """Init with both model and source_ckpt should work."""
        from nvflare.app_common.np.np_model_persistor import NPModelPersistor

        persistor = NPModelPersistor(
            model=[[1, 2], [3, 4]],
            source_ckpt_file_full_name="/data/pretrained/model.npy",
        )

        assert persistor.model == [[1, 2], [3, 4]]
        assert persistor.source_ckpt_file_full_name == "/data/pretrained/model.npy"


class TestNPFileModelPersistorInit:
    """Tests for NPFileModelPersistor initialization with source_ckpt_file_full_name."""

    def test_init_without_source_ckpt(self):
        """Init without source_ckpt should work."""
        from nvflare.app_common.ccwf.comps.np_file_model_persistor import NPFileModelPersistor

        persistor = NPFileModelPersistor()

        assert persistor.source_ckpt_file_full_name is None

    def test_init_with_source_ckpt_stores_path(self):
        """Init should store the source_ckpt path."""
        from nvflare.app_common.ccwf.comps.np_file_model_persistor import NPFileModelPersistor

        persistor = NPFileModelPersistor(
            source_ckpt_file_full_name="/data/pretrained/model.npy",
        )

        assert persistor.source_ckpt_file_full_name == "/data/pretrained/model.npy"
