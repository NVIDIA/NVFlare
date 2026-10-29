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

import pytest

from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.job_config_converter import convert_job_configs

HOCON_META = """
name = hocon_job
deploy_map { app = ["@ALL"] }
min_clients = ${rounds}
rounds = 2
"""

HOCON_CLIENT = """
include "../../../shared.conf"
format_version = 2
executors = [{ tasks = ["train"], executor { path = "a.B", args { lr = ${lr}, home = ${?CONVERTER_TEST_HOME} } } }]
"""

YAML_SERVER = """
base: &base {path: a.B}
components:
  - <<: *base
    id: c1
rounds: ${num_rounds}
num_rounds: 3
"""


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


@pytest.fixture
def legacy_job(tmp_path):
    job_dir = tmp_path / "jobs" / "job1"
    _write(str(tmp_path / "jobs" / "shared.conf"), "lr = 0.01")
    _write(str(job_dir / "meta.conf"), HOCON_META)
    _write(str(job_dir / "app" / "config" / "config_fed_client.conf"), HOCON_CLIENT)
    _write(str(job_dir / "app" / "config" / "config_fed_server.yml"), YAML_SERVER)
    _write(str(job_dir / "app" / "custom" / "logging.conf"), "[loggers]\nkeys=root\n")
    return job_dir


def test_legacy_configs_render_to_inert_data_without_touching_source(legacy_job, tmp_path, monkeypatch):
    monkeypatch.setenv("CONVERTER_TEST_HOME", "/submitter/home")
    original = (legacy_job / "meta.conf").read_text()

    converted = convert_job_configs(str(legacy_job), str(tmp_path / "work"))

    assert converted == str(tmp_path / "work" / "job1")
    assert (legacy_job / "meta.conf").read_text() == original
    assert json.loads(open(os.path.join(converted, "meta.conf")).read())["min_clients"] == 2
    config_dir = os.path.join(converted, "app", "config")
    client = ConfigService.load_config_dict("config_fed_client.json", [config_dir])
    assert client["executors"][0]["executor"]["args"] == {"lr": 0.01, "home": "/submitter/home"}
    server = ConfigService.load_config_dict("config_fed_server.json", [config_dir])
    assert server["rounds"] == 3
    assert server["components"] == [{"path": "a.B", "id": "c1"}]
    assert open(os.path.join(converted, "app", "custom", "logging.conf")).read().startswith("[loggers]")


def test_json_job_is_submitted_unchanged(tmp_path):
    job_dir = tmp_path / "job2"
    _write(str(job_dir / "meta.conf"), '{"name": "job2"}')
    _write(str(job_dir / "app" / "config" / "config_fed_client.yml"), "lr: 1e-5\n")

    assert convert_job_configs(str(job_dir), str(tmp_path / "work")) == str(job_dir)
    assert not (tmp_path / "work").exists()


def test_unresolvable_substitution_names_the_file(tmp_path):
    job_dir = tmp_path / "job3"
    _write(str(job_dir / "meta.conf"), "name = ${CONVERTER_TEST_MISSING}")

    with pytest.raises(ValueError, match="meta.conf"):
        convert_job_configs(str(job_dir), str(tmp_path / "work"))


def test_legacy_app_folder_and_symlinks_are_preserved(tmp_path):
    app_dir = tmp_path / "legacy_app"
    _write(str(app_dir / "config" / "config_fed_client.conf"), "format_version = 2\nexecutors = []\n")
    os.symlink("/nonexistent-target", str(app_dir / "dangling"))

    converted = convert_job_configs(str(app_dir), str(tmp_path / "work"))

    assert json.loads(open(os.path.join(converted, "config", "config_fed_client.conf")).read())["format_version"] == 2
    assert os.readlink(os.path.join(converted, "dangling")) == "/nonexistent-target"


@pytest.mark.parametrize("link", ["file", "folder"])
def test_linked_configs_are_rejected_without_touching_their_targets(tmp_path, link):
    outside = tmp_path / "outside" / "config"
    _write(str(outside / "config_fed_client.conf"), "format_version = 2\n")
    job_dir = tmp_path / "job4"
    if link == "file":
        (job_dir / "app" / "config").mkdir(parents=True)
        os.symlink(str(outside / "config_fed_client.conf"), str(job_dir / "app" / "config" / "config_fed_client.conf"))
    else:
        (job_dir / "app").mkdir(parents=True)
        os.symlink(str(outside), str(job_dir / "app" / "config"))

    with pytest.raises(ValueError, match="linked job config"):
        convert_job_configs(str(job_dir), str(tmp_path / "work"))
    assert (outside / "config_fed_client.conf").read_text() == "format_version = 2\n"


def test_default_fallback_configs_are_rendered(tmp_path):
    job_dir = tmp_path / "job5"
    _write(str(job_dir / "meta.yml.default"), "base: &b {n: 1}\nother: *b\n")
    _write(str(job_dir / "app" / "config" / "config_fed_client.conf.default"), "format_version = 2\n")

    converted = convert_job_configs(str(job_dir), str(tmp_path / "work"))

    assert ConfigService.load_config_dict(os.path.join(converted, "meta.yml.default")) == {
        "base": {"n": 1},
        "other": {"n": 1},
    }
    client = os.path.join(converted, "app", "config", "config_fed_client.conf.default")
    assert ConfigService.load_config_dict(client) == {"format_version": 2}


def test_configs_shadowed_by_higher_precedence_files_are_not_rendered(tmp_path):
    job_dir = tmp_path / "job6"
    config_dir = job_dir / "app" / "config"
    _write(str(config_dir / "config_fed_client.json"), '{"format_version": 2}')
    _write(str(config_dir / "config_fed_client.conf.default"), "data_root = ${CONVERTER_TEST_MISSING}\n")
    _write(str(config_dir / "config_fed_server.conf"), "format_version = 2\n")
    _write(str(config_dir / "config_fed_server.yml"), "rounds: ${CONVERTER_TEST_MISSING}\n")

    converted = convert_job_configs(str(job_dir), str(tmp_path / "work"))

    converted_config = os.path.join(converted, "app", "config")
    assert json.loads(open(os.path.join(converted_config, "config_fed_server.conf")).read()) == {"format_version": 2}
    assert open(os.path.join(converted_config, "config_fed_client.conf.default")).read().startswith("data_root = ${")
