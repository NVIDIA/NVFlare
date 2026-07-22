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
import sys
from unittest.mock import Mock, patch

import pytest

from nvflare.job_config.fed_job_config import FedJobConfig


class TestFedJobConfig:
    def test_locate_imports(self):
        job_config = FedJobConfig(job_name="job_name", min_clients=1)
        cwd = os.path.dirname(__file__)
        source_file = os.path.join(cwd, "../data/job_config/sample_code.data")
        expected = [
            ("typing", 0),
            ("nvflare.fuel.f3.drivers.base_driver", 0),
            ("nvflare.fuel.f3.drivers.connector_info", 0),
            ("nvflare.fuel.f3.drivers.driver_params", 0),
        ]
        with open(source_file, "r") as sf:
            imports = list(job_config.locate_imports(sf))
        assert imports == expected

    @pytest.mark.parametrize("script", ["client.py", "./client.py"])
    def test_copy_ext_script_finds_top_level_import_in_parent_directory(self, tmp_path, monkeypatch, script):
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (tmp_path / "custom_layers.py").write_text("class PlainAdder:\n    pass\n", encoding="utf-8")
        (project_dir / "client.py").write_text("from custom_layers import PlainAdder\n", encoding="utf-8")
        monkeypatch.chdir(project_dir)

        custom_dir = tmp_path / "exported" / "custom"
        job_config = FedJobConfig(job_name="job_name", min_clients=1)
        job_config._copy_ext_scripts(str(custom_dir), [script])

        assert (custom_dir / "client.py").is_file()
        assert (custom_dir / "custom_layers.py").is_file()

    def test_copy_ext_script_finds_top_level_import_in_same_directory(self, tmp_path, monkeypatch):
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
        (project_dir / "client.py").write_text("import helper\n", encoding="utf-8")
        monkeypatch.chdir(project_dir)

        custom_dir = tmp_path / "exported" / "custom"
        job_config = FedJobConfig(job_name="job_name", min_clients=1)
        job_config._copy_ext_scripts(str(custom_dir), ["client.py"])

        assert (custom_dir / "helper.py").is_file()

    def test_copy_ext_scripts_reject_distinct_absolute_sources_with_same_destination(self, tmp_path, monkeypatch):
        first_dir = tmp_path / "first"
        second_dir = tmp_path / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        first_script = first_dir / "client.py"
        second_script = second_dir / "client.py"
        first_script.write_text("SOURCE = 'first'\n", encoding="utf-8")
        second_script.write_text("SOURCE = 'second'\n", encoding="utf-8")
        monkeypatch.setattr(sys, "path", [])

        custom_dir = tmp_path / "exported" / "custom"
        job_config = FedJobConfig(job_name="job_name", min_clients=1)

        with pytest.raises(ValueError, match="map to the same destination"):
            job_config._copy_ext_scripts(str(custom_dir), [str(first_script), str(second_script)])
        assert (custom_dir / "client.py").read_text(encoding="utf-8") == "SOURCE = 'first'\n"

    def test_copy_ext_script_accepts_absolute_symlink_alias_within_sys_path(self, tmp_path, monkeypatch):
        source_root = tmp_path / "source"
        source_root.mkdir()
        source_file = source_root / "client.py"
        source_file.write_text("VALUE = 1\n", encoding="utf-8")
        source_alias = tmp_path / "source_alias"
        try:
            source_alias.symlink_to(source_root, target_is_directory=True)
        except OSError as e:
            pytest.skip(f"symlinks are not available: {e}")
        monkeypatch.setattr(sys, "path", [str(source_root)])

        custom_dir = tmp_path / "exported" / "custom"
        job_config = FedJobConfig(job_name="job_name", min_clients=1)
        job_config._copy_ext_scripts(str(custom_dir), [str(source_alias / "client.py")])

        assert (custom_dir / "client.py").read_text(encoding="utf-8") == "VALUE = 1\n"

    def test_absolute_import_does_not_resolve_to_package_sibling(self, tmp_path, monkeypatch):
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        (package_dir / "traceback.py").write_text("from ._compatibility import helper\n", encoding="utf-8")
        (package_dir / "client.py").write_text("import traceback\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        custom_dir = tmp_path / "exported" / "custom"
        job_config = FedJobConfig(job_name="job_name", min_clients=1)
        job_config._copy_ext_scripts(str(custom_dir), ["pkg/client.py"])

        assert not (custom_dir / "traceback.py").exists()

    def test_copy_ext_script_resolves_valid_multi_level_relative_import(self, tmp_path, monkeypatch):
        package_dir = tmp_path / "pkg"
        script_dir = package_dir / "sub"
        script_dir.mkdir(parents=True)
        (package_dir / "helper.py").write_text("class Helper:\n    pass\n", encoding="utf-8")
        (script_dir / "client.py").write_text("from ..helper import Helper\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        custom_dir = tmp_path / "exported" / "custom"
        job_config = FedJobConfig(job_name="job_name", min_clients=1)
        job_config._copy_ext_scripts(str(custom_dir), ["pkg/sub/client.py"])

        assert (custom_dir / "pkg" / "helper.py").is_file()

    @pytest.mark.parametrize("import_statement", ["from ..outside import Secret\n", "from .. import *\n"])
    def test_copy_ext_script_rejects_relative_import_above_source_root(self, tmp_path, monkeypatch, import_statement):
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        (package_dir / "client.py").write_text(import_statement, encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        custom_dir = tmp_path / "exported" / "custom"
        job_config = FedJobConfig(job_name="job_name", min_clients=1)

        with pytest.raises(ValueError, match="escapes the allowed source root"):
            job_config._copy_ext_scripts(str(custom_dir), ["pkg/client.py"])

    def test_copy_source_file_rejects_source_outside_allowed_root(self, tmp_path):
        source_root = tmp_path / "source"
        source_root.mkdir()
        outside_source = tmp_path / "outside.py"
        outside_source.write_text("SECRET = True\n", encoding="utf-8")
        custom_dir = tmp_path / "exported" / "custom"
        dest_file = custom_dir / "outside.py"
        job_config = FedJobConfig(job_name="job_name", min_clients=1)

        with pytest.raises(ValueError, match="outside the allowed source root"):
            job_config._copy_source_file(
                str(custom_dir),
                "outside",
                str(outside_source),
                str(dest_file),
                source_root=str(source_root),
            )

    def test_copy_source_file_rejects_destination_outside_custom_dir(self, tmp_path):
        source_root = tmp_path / "source"
        source_root.mkdir()
        source_file = source_root / "client.py"
        source_file.write_text("VALUE = 1\n", encoding="utf-8")
        custom_dir = tmp_path / "exported" / "custom"
        outside_dest = tmp_path / "outside.py"
        job_config = FedJobConfig(job_name="job_name", min_clients=1)

        with pytest.raises(ValueError, match="outside the custom directory"):
            job_config._copy_source_file(
                str(custom_dir),
                "client",
                str(source_file),
                str(outside_dest),
                source_root=str(source_root),
            )

    def test_copy_ext_script_rejects_source_symlink_escape(self, tmp_path, monkeypatch):
        source_root = tmp_path / "source"
        project_dir = source_root / "proj"
        project_dir.mkdir(parents=True)
        outside_source = tmp_path / "outside.py"
        outside_source.write_text("SECRET = True\n", encoding="utf-8")
        helper_link = source_root / "helper.py"
        try:
            helper_link.symlink_to(outside_source)
        except OSError as e:
            pytest.skip(f"symlinks are not available: {e}")
        (project_dir / "client.py").write_text("from helper import SECRET\n", encoding="utf-8")
        monkeypatch.chdir(project_dir)

        custom_dir = tmp_path / "exported" / "custom"
        job_config = FedJobConfig(job_name="job_name", min_clients=1)

        with pytest.raises(ValueError, match="outside the allowed source root"):
            job_config._copy_ext_scripts(str(custom_dir), ["client.py"])

    def test_copy_ext_script_rejects_destination_symlink_escape(self, tmp_path, monkeypatch):
        source_root = tmp_path / "source"
        project_dir = source_root / "proj"
        project_dir.mkdir(parents=True)
        (source_root / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
        (project_dir / "client.py").write_text("from helper import VALUE\n", encoding="utf-8")
        monkeypatch.chdir(project_dir)

        custom_dir = tmp_path / "exported" / "custom"
        custom_dir.mkdir(parents=True)
        outside_dest = tmp_path / "outside.py"
        outside_dest.write_text("DO NOT OVERWRITE\n", encoding="utf-8")
        try:
            (custom_dir / "helper.py").symlink_to(outside_dest)
        except OSError as e:
            pytest.skip(f"symlinks are not available: {e}")
        job_config = FedJobConfig(job_name="job_name", min_clients=1)

        with pytest.raises(ValueError, match="outside the custom directory"):
            job_config._copy_ext_scripts(str(custom_dir), ["client.py"])
        assert outside_dest.read_text(encoding="utf-8") == "DO NOT OVERWRITE\n"

    def test_copy_ext_script_rejects_source_destination_alias(self, tmp_path, monkeypatch):
        source_file = tmp_path / "client.py"
        source_file.write_text("VALUE = 1\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        job_config = FedJobConfig(job_name="job_name", min_clients=1)

        with pytest.raises(ValueError, match="same file"):
            job_config._copy_ext_scripts(str(tmp_path), ["client.py"])
        assert source_file.read_text(encoding="utf-8") == "VALUE = 1\n"

    @pytest.mark.parametrize(
        ("script", "content"),
        [
            ("run-script.sh", "#!/bin/sh\nexit 0\n"),
            ("train-script.py", "VALUE = 1\n"),
        ],
    )
    def test_copy_ext_script_preserves_non_module_filenames(self, tmp_path, monkeypatch, script, content):
        (tmp_path / script).write_text(content, encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        custom_dir = tmp_path / "exported" / "custom"
        job_config = FedJobConfig(job_name="job_name", min_clients=1)

        job_config._copy_ext_scripts(str(custom_dir), [script])

        assert (custom_dir / script).read_text(encoding="utf-8") == content

    def test_trim_whitespace(self):
        job_config = FedJobConfig(job_name="job_name", min_clients=1)
        expected = "site-0,site-1"
        assert expected == job_config._trim_whitespace("site-0,site-1")
        assert expected == job_config._trim_whitespace("site-0, site-1")
        assert expected == job_config._trim_whitespace(" site-0,site-1 ")
        assert expected == job_config._trim_whitespace(" site-0, site-1 ")

    def test_simulator_run_returns_process_returncode(self, tmp_path):
        job_config = FedJobConfig(job_name="test_job", min_clients=1)
        process = Mock()
        process.wait.return_value = 0

        with patch.object(job_config, "generate_job_config"):
            with patch("nvflare.job_config.fed_job_config.subprocess.Popen", return_value=process):
                result = job_config.simulator_run(str(tmp_path), n_clients=1)

        assert result == 0

    def test_simulator_run_returns_nonzero_process_returncode(self, tmp_path):
        job_config = FedJobConfig(job_name="job_name", min_clients=1)
        process = Mock()
        process.wait.return_value = 2

        with patch.object(job_config, "generate_job_config"):
            with patch("nvflare.job_config.fed_job_config.subprocess.Popen", return_value=process):
                result = job_config.simulator_run(workspace=str(tmp_path), clients="site-1", threads=1)

        assert result == 2
