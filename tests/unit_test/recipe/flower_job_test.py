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
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nvflare.app_opt.flower.flower_job import FlowerJob


@pytest.fixture
def tmp_flower_content_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        content_dir = Path(tmpdir) / "flower_content"
        content_dir.mkdir()
        (content_dir / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        yield str(content_dir)


def _make_mock_component(target_type):
    mock = MagicMock()
    mock.get_job_target_type.return_value = target_type
    return mock


class TestFlowerJob:
    def test_flower_job_with_byoc_content(self, tmp_flower_content_dir):
        job = FlowerJob(
            name="test_job",
            flower_content=tmp_flower_content_dir,
        )
        assert job.job.fed_apps is not None
        for app_name, fed_app in job.job.fed_apps.items():
            if fed_app.server_app:
                assert tmp_flower_content_dir in fed_app.server_app.ext_dirs
            if fed_app.client_app:
                assert tmp_flower_content_dir in fed_app.client_app.ext_dirs

    def test_flower_job_no_ext_dirs_when_predeployed(self):
        with (
            patch("nvflare.app_opt.flower.flower_job.FlowerController", return_value=_make_mock_component("server")),
            patch("nvflare.app_opt.flower.flower_job.FlowerExecutor", return_value=_make_mock_component("client")),
        ):
            job = FlowerJob(
                name="test_job_predeployed",
                flower_app_path="/opt/flower_apps/my_app",
            )
            for app_name, fed_app in job.job.fed_apps.items():
                if fed_app.server_app:
                    assert fed_app.server_app.ext_dirs == []
                if fed_app.client_app:
                    assert fed_app.client_app.ext_dirs == []

    def test_flower_job_rejects_both_content_and_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Specify either 'flower_content'"):
                FlowerJob(
                    name="test_job",
                    flower_content=tmpdir,
                    flower_app_path="/opt/flower_apps/my_app",
                )

    def test_flower_job_rejects_neither_content_nor_path(self):
        with pytest.raises(ValueError, match="One of 'flower_content' or 'flower_app_path' must be provided"):
            FlowerJob(name="test_job")

    def test_flower_job_rejects_invalid_content_dir(self):
        with pytest.raises(ValueError, match="is not a valid directory"):
            FlowerJob(
                name="test_job",
                flower_content="/nonexistent/path/flower/content",
            )

    def test_flower_job_propagates_flower_app_path_to_controller(self):
        with (
            patch("nvflare.app_opt.flower.flower_job.FlowerController") as mock_controller,
            patch("nvflare.app_opt.flower.flower_job.FlowerExecutor", return_value=_make_mock_component("client")),
        ):
            mock_controller.return_value = _make_mock_component("server")
            FlowerJob(
                name="test_job",
                flower_app_path="/opt/flower_apps/my_app",
            )
            call_kwargs = mock_controller.call_args.kwargs
            assert call_kwargs["flower_app_path"] == "/opt/flower_apps/my_app"

    def test_flower_job_propagates_flower_app_path_to_executor(self):
        with (
            patch("nvflare.app_opt.flower.flower_job.FlowerController", return_value=_make_mock_component("server")),
            patch("nvflare.app_opt.flower.flower_job.FlowerExecutor") as mock_executor,
        ):
            mock_executor.return_value = _make_mock_component("client")
            FlowerJob(
                name="test_job",
                flower_app_path="/opt/flower_apps/my_app",
            )
            call_kwargs = mock_executor.call_args.kwargs
            assert call_kwargs["flower_app_path"] == "/opt/flower_apps/my_app"
