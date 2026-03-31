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

import tempfile
from unittest.mock import patch

import pytest

from nvflare.recipe.prod_env import ProdEnv


def test_prod_env_session_manager_passes_study():
    with tempfile.TemporaryDirectory() as startup_kit_location:
        env = ProdEnv(startup_kit_location=startup_kit_location, study="cancer-research")
        with patch("nvflare.recipe.prod_env.SessionManager") as mock_session_manager:
            env._get_session_manager()
            session_params = mock_session_manager.call_args[0][0]
            assert session_params["study"] == "cancer-research"


def test_prod_env_rejects_invalid_study_name():
    with tempfile.TemporaryDirectory() as startup_kit_location:
        with pytest.raises(ValueError):
            ProdEnv(startup_kit_location=startup_kit_location, study="Bad Study")


def test_prod_env_defaults_study():
    with tempfile.TemporaryDirectory() as startup_kit_location:
        env = ProdEnv(startup_kit_location=startup_kit_location)
        with patch("nvflare.recipe.prod_env.SessionManager") as mock_session_manager:
            env._get_session_manager()
            session_params = mock_session_manager.call_args[0][0]
            assert session_params["study"] == "default"
