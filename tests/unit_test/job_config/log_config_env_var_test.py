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
import json
from unittest.mock import patch

from nvflare.fuel.utils.log_utils import FL_LOG_LEVEL, configure_logging
from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner


class TestSimulatorRunnerFlLogLevel:
    def test_env_var_used_when_no_param(self, tmp_path):
        job_folder = str(tmp_path / "job")
        os.makedirs(job_folder, exist_ok=True)
        workspace = str(tmp_path / "workspace")

        with patch.dict(os.environ, {FL_LOG_LEVEL: "error"}):
            runner = SimulatorRunner(job_folder=job_folder, workspace=workspace)
        assert runner.log_config == "error"

    def test_explicit_param_overrides_env_var(self, tmp_path):
        job_folder = str(tmp_path / "job")
        os.makedirs(job_folder, exist_ok=True)
        workspace = str(tmp_path / "workspace")

        with patch.dict(os.environ, {FL_LOG_LEVEL: "error"}):
            runner = SimulatorRunner(job_folder=job_folder, workspace=workspace, log_config="concise")
        assert runner.log_config == "concise"


def _extract_filenames(obj):
    filenames = []
    stack = [obj]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if key == "filename":
                    filenames.append(value)
                elif isinstance(value, dict):
                    stack.append(value)
    return filenames


def _write_log_config(path):
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"simple": {"format": "%(message)s"}},
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "simple",
                "filename": "log.txt",
            }
        },
        "root": {"level": "INFO", "handlers": ["file"]},
    }
    with open(path, "w") as f:
        json.dump(config, f)


class _FakeWorkspace:
    def __init__(self, log_config_file_path, log_root):
        self.log_config_file_path = log_config_file_path
        self.log_root = log_root

    def get_log_config_file_path(self):
        return self.log_config_file_path

    def get_log_root(self, job_id=None):
        return self.log_root


class TestConfigureLoggingFlLogLevel:
    def test_env_log_mode_keeps_file_prefix(self, tmp_path):
        config_path = tmp_path / "log_config.json"
        _write_log_config(config_path)
        log_root = tmp_path / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        workspace = _FakeWorkspace(str(config_path), str(log_root))
        dict_configs = []

        with patch("logging.config.dictConfig", side_effect=lambda x: dict_configs.append(x)):
            with patch.dict(os.environ, {FL_LOG_LEVEL: "concise"}):
                configure_logging(workspace, file_prefix="applet")

        assert dict_configs
        filenames = _extract_filenames(dict_configs[-1])
        assert filenames
        assert all(os.path.basename(f).startswith("applet_") for f in filenames)

    def test_env_log_path_keeps_file_prefix(self, tmp_path):
        config_path = tmp_path / "log_config.json"
        _write_log_config(config_path)
        env_config_path = tmp_path / "env_log_config.json"
        _write_log_config(env_config_path)
        log_root = tmp_path / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        workspace = _FakeWorkspace(str(config_path), str(log_root))
        dict_configs = []

        with patch("logging.config.dictConfig", side_effect=lambda x: dict_configs.append(x)):
            with patch.dict(os.environ, {FL_LOG_LEVEL: str(env_config_path)}):
                configure_logging(workspace, file_prefix="applet")

        assert dict_configs
        filenames = _extract_filenames(dict_configs[-1])
        assert filenames
        assert all(os.path.basename(f).startswith("applet_") for f in filenames)
