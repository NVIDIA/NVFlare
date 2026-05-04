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

import os
import tempfile
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest

from nvflare.app_opt.flower.recipe import FlowerRecipe
from nvflare.client.api import ClientAPIType
from nvflare.client.api_spec import CLIENT_API_TYPE_KEY


@pytest.mark.parametrize("flwr_version", ["1.15.9", "1.16rc0", "1.25.9", "1.26.0rc0"])
def test_flower_recipe_rejects_incompatible_flwr_version(flwr_version):
    with patch("nvflare.app_opt.flower.recipe.get_package_version", return_value=flwr_version):
        with patch("nvflare.app_opt.flower.recipe._create_flower_job") as mock_flower_job:
            with pytest.raises(RuntimeError, match=r"requires 'flwr>=1\.26'"):
                FlowerRecipe(flower_content="mock_flower_content")

            mock_flower_job.assert_not_called()


def test_flower_recipe_rejects_missing_flwr_package():
    with patch("nvflare.app_opt.flower.recipe.get_package_version", side_effect=PackageNotFoundError):
        with patch("nvflare.app_opt.flower.recipe._create_flower_job") as mock_flower_job:
            with pytest.raises(RuntimeError, match=r"requires 'flwr>=1\.26'"):
                FlowerRecipe(flower_content="mock_flower_content")

            mock_flower_job.assert_not_called()


@pytest.mark.parametrize("flwr_version", ["1.26.0", "1.26.1", "1.27.5"])
def test_flower_recipe_accepts_compatible_flwr_version(flwr_version):
    fake_job = object()
    with patch("nvflare.app_opt.flower.recipe.get_package_version", return_value=flwr_version):
        with patch("nvflare.app_opt.flower.recipe._create_flower_job", return_value=fake_job) as mock_flower_job:
            recipe = FlowerRecipe(flower_content="mock_flower_content")

    assert recipe.job is fake_job
    kwargs = mock_flower_job.call_args.kwargs
    assert kwargs["extra_env"] == {CLIENT_API_TYPE_KEY: ClientAPIType.EX_PROCESS_API.value}


@pytest.mark.parametrize("flwr_version", ["1.26.0", "1.26.1", "1.27.5"])
def test_flower_recipe_merges_extra_env(flwr_version):
    fake_job = object()
    user_env = {"MY_VAR": "123"}

    with patch("nvflare.app_opt.flower.recipe.get_package_version", return_value=flwr_version):
        with patch("nvflare.app_opt.flower.recipe._create_flower_job", return_value=fake_job) as mock_flower_job:
            recipe = FlowerRecipe(flower_content="mock_flower_content", extra_env=user_env)

    assert recipe.job is fake_job
    kwargs = mock_flower_job.call_args.kwargs
    assert kwargs["extra_env"]["MY_VAR"] == "123"
    assert kwargs["extra_env"][CLIENT_API_TYPE_KEY] == ClientAPIType.EX_PROCESS_API.value


@pytest.mark.parametrize("flwr_version", ["1.26.0", "1.26.1", "1.27.5"])
def test_flower_recipe_rejects_extra_env_with_wrong_client_api_type(flwr_version):
    bad_value = "wrong_api_type"
    user_env = {CLIENT_API_TYPE_KEY: bad_value, "MY_VAR": "123"}

    with patch("nvflare.app_opt.flower.recipe.get_package_version", return_value=flwr_version):
        with patch("nvflare.app_opt.flower.recipe._create_flower_job") as mock_flower_job:
            with pytest.raises(
                ValueError,
                match=rf"'extra_env\[{CLIENT_API_TYPE_KEY}\]' must be '"
                rf"{ClientAPIType.EX_PROCESS_API.value}' for the Flower integration; got '{bad_value}'\.",
            ):
                FlowerRecipe(flower_content="mock_flower_content", extra_env=user_env)

    mock_flower_job.assert_not_called()


@pytest.mark.parametrize("flwr_version", ["1.26.0", "1.26.1", "1.27.5"])
def test_flower_recipe_with_predeployed_path(flwr_version):
    fake_job = object()
    with patch("nvflare.app_opt.flower.recipe.get_package_version", return_value=flwr_version):
        with patch("nvflare.app_opt.flower.recipe._create_flower_job", return_value=fake_job) as mock_flower_job:
            recipe = FlowerRecipe(flower_app_path="/opt/flower_apps/my_app")

    assert recipe.job is fake_job
    kwargs = mock_flower_job.call_args.kwargs
    assert kwargs["flower_app_path"] == "/opt/flower_apps/my_app"
    assert kwargs["flower_content"] is None


@pytest.mark.parametrize("flwr_version", ["1.26.0", "1.26.1", "1.27.5"])
def test_flower_recipe_passes_content_and_path_through(flwr_version):
    fake_job = object()
    with patch("nvflare.app_opt.flower.recipe.get_package_version", return_value=flwr_version):
        with patch("nvflare.app_opt.flower.recipe._create_flower_job", return_value=fake_job) as mock_flower_job:
            recipe = FlowerRecipe(flower_content="mock_flower_content", flower_app_path="/opt/flower_apps/my_app")

    kwargs = mock_flower_job.call_args.kwargs
    assert kwargs["flower_content"] == "mock_flower_content"
    assert kwargs["flower_app_path"] == "/opt/flower_apps/my_app"


def test_flower_recipe_rejects_both_content_and_path():
    with patch("nvflare.app_opt.flower.recipe.get_package_version", return_value="1.26.0"):
        with patch(
            "nvflare.app_opt.flower.recipe._create_flower_job",
            side_effect=ValueError("Specify either 'flower_content'"),
        ):
            with pytest.raises(ValueError, match="Specify either 'flower_content'"):
                FlowerRecipe(flower_content="mock_flower_content", flower_app_path="/opt/flower_apps/my_app")


def test_flower_recipe_rejects_neither_content_nor_path():
    with patch("nvflare.app_opt.flower.recipe.get_package_version", return_value="1.26.0"):
        with patch(
            "nvflare.app_opt.flower.recipe._create_flower_job",
            side_effect=ValueError("One of 'flower_content' or 'flower_app_path' must be provided"),
        ):
            with pytest.raises(ValueError, match="One of 'flower_content' or 'flower_app_path' must be provided"):
                FlowerRecipe()


def test_flower_job_no_byoc_when_predeployed():
    """Verify that a FlowerJob with flower_app_path exports without any custom/ content (no BYOC)."""
    from nvflare.app_opt.flower.flower_job import FlowerJob

    with tempfile.TemporaryDirectory() as app_dir:
        with tempfile.TemporaryDirectory() as job_root:
            job = FlowerJob(
                name="test_no_byoc_job",
                flower_app_path=app_dir,
                min_clients=1,
            )
            job.export_job(job_root)

            job_dir = os.path.join(job_root, "test_no_byoc_job")
            assert os.path.isdir(job_dir), "Job directory was not created"

            for app_name in os.listdir(job_dir):
                custom_dir = os.path.join(job_dir, app_name, "custom")
                if os.path.isdir(custom_dir):
                    custom_contents = os.listdir(custom_dir)
                    assert (
                        len(custom_contents) == 0
                    ), f"Expected empty custom/ in {app_name}, but found: {custom_contents}"


def test_flower_job_has_byoc_when_content_provided():
    """Verify that a FlowerJob with flower_content exports with custom/ content (BYOC mode)."""
    from nvflare.app_opt.flower.flower_job import FlowerJob

    with tempfile.TemporaryDirectory() as app_dir:
        test_file = os.path.join(app_dir, "pyproject.toml")
        with open(test_file, "w") as f:
            f.write("[tool.poetry]\nname = 'test'\n")

        with tempfile.TemporaryDirectory() as job_root:
            job = FlowerJob(
                name="test_byoc_job",
                flower_content=app_dir,
                min_clients=1,
            )
            job.export_job(job_root)

            job_dir = os.path.join(job_root, "test_byoc_job")
            assert os.path.isdir(job_dir), "Job directory was not created"

            server_custom = os.path.join(job_dir, "app", "custom")
            assert os.path.isdir(server_custom), "Server custom/ directory was not created"
            assert "pyproject.toml" in os.listdir(
                server_custom
            ), "Expected pyproject.toml in server custom/ for BYOC mode"
