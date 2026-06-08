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
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.fl_constant import ConfigVarName
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs import Decomposer
from nvflare.fuel.utils.fobs.datum import DatumManager
from nvflare.fuel.utils.fobs.fobs import register_custom_folder
from nvflare.private.fed.utils.fed_utils import (
    create_job_processing_context_properties,
    custom_fobs_initialize,
    extract_participants,
)


class ExampleTestClass:
    def __init__(self, number):
        self.number = number


class ExampleTestClassDecomposer(Decomposer):
    def __init__(self):
        super().__init__()
        self.type = ExampleTestClass.__name__

    def __eq__(self, other):
        return hasattr(other, "type") and self.type == other.type

    def supported_type(self):
        return ExampleTestClass

    def decompose(self, target: ExampleTestClass, manager: DatumManager = None) -> Any:
        return target.number

    def recompose(self, data: Any, manager: DatumManager = None) -> ExampleTestClass:
        return ExampleTestClass(data)


class TestFedUtils:
    def test_custom_fobs_initialize(self):
        pwd = os.path.dirname(os.path.realpath(__file__))
        register_custom_folder(pwd)
        decomposer = ExampleTestClassDecomposer()
        decomposers = fobs.fobs._decomposers
        assert decomposer in list(decomposers.values())

    def test_extract_participants(self):
        participants = ["site-1", "site-2"]
        results = extract_participants(participants)
        expected = ["site-1", "site-2"]
        assert results == expected

        participants = ["@ALL"]
        results = extract_participants(participants)
        expected = ["@ALL"]
        assert results == expected

    def test_extract_participants_with_image(self):
        participants = [
            "site-1",
            "site-2",
            {"sites": ["site-3", "site-4"], "image": "image1"},
            {"sites": ["site-5"], "image": "image2"},
        ]
        results = extract_participants(participants)
        expected = ["site-1", "site-2", "site-3", "site-4", "site-5"]
        assert results == expected

    def test_create_job_processing_context_properties_rejects_non_dict_job_meta(self):
        workspace = MagicMock()

        with patch("nvflare.private.fed.utils.fed_utils.get_job_meta_from_workspace", return_value=[]):
            with pytest.raises(RuntimeError, match="job_meta must be dict"):
                create_job_processing_context_properties(workspace, "job-1")

    @staticmethod
    def _make_workspace():
        workspace = MagicMock()
        workspace.get_client_custom_dir.return_value = os.path.join("site", "custom")
        workspace.get_app_custom_dir.return_value = os.path.join("job", "custom")
        workspace.get_app_config_dir.return_value = os.path.join("job", "config")
        return workspace

    @patch("nvflare.private.fed.utils.fed_utils.register_custom_folder")
    @patch("nvflare.private.fed.utils.fed_utils.get_job_meta_from_workspace")
    @patch("nvflare.private.fed.utils.fed_utils.os.path.exists", return_value=True)
    def test_job_config_decomposers_are_ignored(self, mock_exists, mock_get_meta, mock_register):
        # even for a BYOC job, decomposers are only loaded from the job custom dir, never from config
        mock_get_meta.return_value = {AppValidationKey.BYOC: True}
        workspace = self._make_workspace()

        custom_fobs_initialize(workspace, job_id="job-1")

        registered_dirs = [call.args[0] for call in mock_register.call_args_list]
        config_decomposer_dir = os.path.join("job", "config", ConfigVarName.DECOMPOSER_MODULE)
        custom_decomposer_dir = os.path.join("job", "custom", ConfigVarName.DECOMPOSER_MODULE)

        # config-shipped decomposers must never be loaded (FLARE-2977)
        assert config_decomposer_dir not in registered_dirs
        # decomposers shipped under the job custom dir are loaded for a BYOC job
        assert custom_decomposer_dir in registered_dirs

    @patch("nvflare.private.fed.utils.fed_utils.register_custom_folder")
    @patch("nvflare.private.fed.utils.fed_utils.get_job_meta_from_workspace")
    @patch("nvflare.private.fed.utils.fed_utils.os.path.exists", return_value=True)
    def test_job_custom_decomposers_ignored_when_not_byoc(self, mock_exists, mock_get_meta, mock_register):
        # job meta without the BYOC flag => job is not allowed to bring custom code
        mock_get_meta.return_value = {}
        workspace = self._make_workspace()

        custom_fobs_initialize(workspace, job_id="job-1")

        registered_dirs = [call.args[0] for call in mock_register.call_args_list]
        custom_decomposer_dir = os.path.join("job", "custom", ConfigVarName.DECOMPOSER_MODULE)
        site_decomposer_dir = os.path.join("site", "custom", ConfigVarName.DECOMPOSER_MODULE)

        # the job's custom/nvflare_decomposers must be ignored for a non-BYOC job
        assert custom_decomposer_dir not in registered_dirs
        # site decomposers are installed by the admin and must still be loaded
        assert site_decomposer_dir in registered_dirs

    @patch("nvflare.private.fed.utils.fed_utils.register_custom_folder")
    @patch("nvflare.private.fed.utils.fed_utils.get_job_meta_from_workspace")
    @patch("nvflare.private.fed.utils.fed_utils.os.path.exists", return_value=True)
    def test_job_decomposers_ignored_when_meta_unavailable(self, mock_exists, mock_get_meta, mock_register):
        # if the job meta cannot be read, default to not loading the job decomposers
        mock_get_meta.side_effect = FileNotFoundError("missing job meta")
        workspace = self._make_workspace()

        custom_fobs_initialize(workspace, job_id="job-1")

        registered_dirs = [call.args[0] for call in mock_register.call_args_list]
        custom_decomposer_dir = os.path.join("job", "custom", ConfigVarName.DECOMPOSER_MODULE)

        assert custom_decomposer_dir not in registered_dirs
