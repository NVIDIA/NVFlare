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

from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest

from nvflare.app_opt.flower.recipe import FlowerRecipe
from nvflare.client.api import ClientAPIType
from nvflare.client.api_spec import CLIENT_API_TYPE_KEY


@pytest.mark.parametrize("flwr_version", ["1.15.9", "1.26.0"])
def test_flower_recipe_rejects_incompatible_flwr_version(flwr_version):
    with patch("nvflare.app_opt.flower.recipe.get_package_version", return_value=flwr_version):
        with patch("nvflare.app_opt.flower.recipe._create_flower_job") as mock_flower_job:
            with pytest.raises(RuntimeError, match=r"requires 'flwr>=1\.16,<1\.26'"):
                FlowerRecipe(flower_content="mock_flower_content")

            mock_flower_job.assert_not_called()


def test_flower_recipe_rejects_missing_flwr_package():
    with patch("nvflare.app_opt.flower.recipe.get_package_version", side_effect=PackageNotFoundError):
        with patch("nvflare.app_opt.flower.recipe._create_flower_job") as mock_flower_job:
            with pytest.raises(RuntimeError, match=r"requires 'flwr>=1\.16,<1\.26'"):
                FlowerRecipe(flower_content="mock_flower_content")

            mock_flower_job.assert_not_called()


@pytest.mark.parametrize("flwr_version", ["1.16.0", "1.25.9"])
def test_flower_recipe_accepts_compatible_flwr_version(flwr_version):
    fake_job = object()
    with patch("nvflare.app_opt.flower.recipe.get_package_version", return_value=flwr_version):
        with patch("nvflare.app_opt.flower.recipe._create_flower_job", return_value=fake_job) as mock_flower_job:
            recipe = FlowerRecipe(flower_content="mock_flower_content")

    assert recipe.job is fake_job
    kwargs = mock_flower_job.call_args.kwargs
    assert kwargs["extra_env"] == {CLIENT_API_TYPE_KEY: ClientAPIType.EX_PROCESS_API.value}
