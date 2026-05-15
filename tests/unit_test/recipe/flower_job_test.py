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

import json
import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.app_validation import AppValidationKey
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
                flower_app_path="local/custom/my_app",
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
                flower_app_path="local/custom/my_app",
            )
            call_kwargs = mock_controller.call_args.kwargs
            assert call_kwargs["flower_app_path"] == "local/custom/my_app"

    def test_flower_job_meta_contains_predeployed_flag(self):
        """flower_app_path sets FLOWER_PREDEPLOYED in meta_props."""
        job = FlowerJob(
            name="test_predeployed_job",
            flower_app_path="local/custom/my_app",
            min_clients=1,
        )
        assert hasattr(job, "job")
        assert hasattr(job.job, "meta_props") or hasattr(job.job, "meta")
        meta = job.job.meta_props or job.job.meta
        assert AppValidationKey.FLOWER_PREDEPLOYED in meta
        assert meta[AppValidationKey.FLOWER_PREDEPLOYED] is True

    def test_flower_job_meta_no_predeployed_flag(self):
        """flower_content does NOT set FLOWER_PREDEPLOYED flag."""
        with tempfile.TemporaryDirectory() as flower_content_dir:
            with open(os.path.join(flower_content_dir, "pyproject.toml"), "w") as f:
                f.write("[project]\nname = 'test'\n")

            job = FlowerJob(
                name="test_byoc_job",
                flower_content=flower_content_dir,
                min_clients=1,
            )
            # When flower_content is used, meta_props is None
            assert job.job.meta_props is None

    def test_flower_job_meta_props_exported_to_zip(self):
        """Exported job ZIP contains meta.json with FLOWER_PREDEPLOYED flag."""
        with tempfile.TemporaryDirectory() as job_root:
            job = FlowerJob(
                name="test_export_job",
                flower_app_path="local/custom/my_app",
                min_clients=1,
            )
            job.export_job(job_root)

            job_dir = os.path.join(job_root, "test_export_job")
            assert os.path.isdir(job_dir)

            for root, dirs, files in os.walk(job_dir):
                for file in files:
                    if file.endswith(".zip"):
                        zip_path = os.path.join(root, file)
                        with zipfile.ZipFile(zip_path, "r") as z:
                            with z.open("meta.json") as f:
                                meta = json.load(f)
                                assert AppValidationKey.FLOWER_PREDEPLOYED in meta
                                assert meta[AppValidationKey.FLOWER_PREDEPLOYED] is True

    def test_flower_job_accepts_local_custom_path(self):
        """Valid local/custom/ paths should be accepted."""
        with (
            patch("nvflare.app_opt.flower.flower_job.FlowerController", return_value=_make_mock_component("server")),
            patch("nvflare.app_opt.flower.flower_job.FlowerExecutor", return_value=_make_mock_component("client")),
        ):
            job = FlowerJob(
                name="test_job",
                flower_app_path="local/custom/flwr_pt_tb",
            )
            assert job is not None

    @pytest.mark.parametrize(
        "invalid_path",
        [
            "/absolute/path/to/app",
            "app/path",
            "../../etc/passwd",
            "C:\\windows\\system32",
            "local/apps/my_app",
        ],
    )
    def test_flower_job_rejects_invalid_prefix(self, invalid_path):
        """Paths not starting with 'local/custom/' should be rejected."""
        with (
            patch("nvflare.app_opt.flower.flower_job.FlowerController", return_value=_make_mock_component("server")),
            patch("nvflare.app_opt.flower.flower_job.FlowerExecutor", return_value=_make_mock_component("client")),
        ):
            with pytest.raises(ValueError, match="flower_app_path must start with 'local/custom/'"):
                FlowerJob(
                    name="test_job",
                    flower_app_path=invalid_path,
                )

    @pytest.mark.parametrize(
        "malicious_path",
        [
            "local/custom/../../../etc/passwd",
            "local/custom/..\\..\\windows\\system32",
            "local/custom/../../secret",
            "local/custom/..",
            "local/custom/../secret",
        ],
    )
    def test_flower_job_rejects_path_traversal(self, malicious_path):
        """Path traversal attempts should be rejected."""
        with (
            patch("nvflare.app_opt.flower.flower_job.FlowerController", return_value=_make_mock_component("server")),
            patch("nvflare.app_opt.flower.flower_job.FlowerExecutor", return_value=_make_mock_component("client")),
        ):
            with pytest.raises(ValueError, match="flower_app_path contains invalid path traversal"):
                FlowerJob(
                    name="test_job",
                    flower_app_path=malicious_path,
                )

    def test_flower_job_handles_mixed_separators(self):
        """Backslashes should also be checked for path traversal."""
        with (
            patch("nvflare.app_opt.flower.flower_job.FlowerController", return_value=_make_mock_component("server")),
            patch("nvflare.app_opt.flower.flower_job.FlowerExecutor", return_value=_make_mock_component("client")),
        ):
            # Path with backslash traversal after valid prefix should be rejected
            with pytest.raises(ValueError, match="invalid path traversal"):
                FlowerJob(
                    name="test_job",
                    flower_app_path="local/custom/..\\secret",
                )

    def test_flower_job_accepts_nested_local_custom_paths(self):
        """Nested paths under local/custom/ should be accepted."""
        with (
            patch("nvflare.app_opt.flower.flower_job.FlowerController", return_value=_make_mock_component("server")),
            patch("nvflare.app_opt.flower.flower_job.FlowerExecutor", return_value=_make_mock_component("client")),
        ):
            job = FlowerJob(
                name="test_job",
                flower_app_path="local/custom/deep/nested/app/path",
            )
            assert job is not None
