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
import logging

import pytest

from nvflare.app_opt.job_launcher.study_runtime import (
    load_study_runtime_file,
    resolve_study_runtime,
    study_runtime_file_path,
)


def _write(tmp_path, text):
    file_path = tmp_path / "study_runtime.yaml"
    file_path.write_text(text, encoding="utf-8")
    return str(file_path)


FULL_EXAMPLE = """
format_version: 2

studies:
  lung-cancer:
    container:
      name: lung-cancer-job

    pod_template: lung_cancer_pod.yaml

    datasets:
      reference:
        source: nvfldata
        mode: ro
      scratch:
        type: mount
        source: scratch-pvc
        mode: rw

    env:
      DB_HOST: postgres.svc
      DB_PORT: 5432

    secret_env:
      DB_USER: {source: study-db, key: username}
      DB_PASSWORD: {source: study-db, key: password}

    secret_mounts:
      db-ca:
        source: study-db-ca
        mount_path: /var/run/nvflare/secrets/db-ca
        mode: ro
        items:
          ca.crt: ca.crt
"""


class TestLoadStudyRuntimeFile:
    def test_parses_full_example(self, tmp_path):
        (tmp_path / "lung_cancer_pod.yaml").write_text(
            "apiVersion: v1\nkind: Pod\nspec:\n  serviceAccountName: study-sa\n", encoding="utf-8"
        )
        runtime_map = load_study_runtime_file(_write(tmp_path, FULL_EXAMPLE))

        runtime = runtime_map["lung-cancer"]
        assert runtime.container_name == "lung-cancer-job"
        assert runtime.pod_template["spec"]["serviceAccountName"] == "study-sa"
        datasets = {d.dataset: d for d in runtime.datasets}
        assert datasets["reference"].source == "nvfldata"
        assert datasets["reference"].read_only is True
        assert datasets["reference"].mount_path == "/data/lung-cancer/reference"
        assert datasets["scratch"].read_only is False
        assert runtime.env == {"DB_HOST": "postgres.svc", "DB_PORT": "5432"}
        secret_env = {ref.name: ref for ref in runtime.secret_env}
        assert secret_env["DB_PASSWORD"].source == "study-db"
        assert secret_env["DB_PASSWORD"].key == "password"
        (mount,) = runtime.secret_mounts
        assert mount.source == "study-db-ca"
        assert mount.mount_path == "/var/run/nvflare/secrets/db-ca"
        assert mount.items == (("ca.crt", "ca.crt"),)

    def test_inline_pod_template(self, tmp_path):
        runtime_map = load_study_runtime_file(
            _write(
                tmp_path,
                "format_version: 2\n"
                "studies:\n"
                "  study-a:\n"
                "    pod_template:\n"
                "      spec:\n"
                "        nodeSelector: {workload: gpu}\n",
            )
        )
        assert runtime_map["study-a"].pod_template == {"spec": {"nodeSelector": {"workload": "gpu"}}}

    def test_missing_format_version_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="format_version"):
            load_study_runtime_file(_write(tmp_path, "studies: {}\n"))

    def test_wrong_format_version_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="format_version"):
            load_study_runtime_file(_write(tmp_path, "format_version: 3\nstudies: {}\n"))

    def test_unknown_top_level_key_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="resources"):
            load_study_runtime_file(_write(tmp_path, "format_version: 2\nstudies: {}\nresources: {}\n"))

    def test_unknown_study_key_rejected(self, tmp_path):
        # a typo'd secret_env must not silently drop a secret
        with pytest.raises(ValueError, match="secret_evn"):
            load_study_runtime_file(_write(tmp_path, "format_version: 2\nstudies:\n  study-a:\n    secret_evn: {}\n"))

    def test_databricks_type_reserved(self, tmp_path):
        with pytest.raises(ValueError, match="not yet supported"):
            load_study_runtime_file(
                _write(
                    tmp_path,
                    "format_version: 2\n"
                    "studies:\n"
                    "  study-a:\n"
                    "    datasets:\n"
                    "      training:\n"
                    "        type: databricks\n"
                    "        workspace: hospital-databricks\n",
                )
            )

    def test_unknown_dataset_type_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown type"):
            load_study_runtime_file(
                _write(
                    tmp_path,
                    "format_version: 2\n"
                    "studies:\n"
                    "  study-a:\n"
                    "    datasets:\n"
                    "      training:\n"
                    "        type: nfs\n"
                    "        source: /data\n"
                    "        mode: ro\n",
                )
            )

    def test_dataset_requires_source_and_mode(self, tmp_path):
        with pytest.raises(ValueError, match="mode"):
            load_study_runtime_file(
                _write(
                    tmp_path,
                    "format_version: 2\nstudies:\n  study-a:\n    datasets:\n      training:\n        source: pvc\n",
                )
            )

    def test_env_and_secret_env_collision_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="both env and secret_env"):
            load_study_runtime_file(
                _write(
                    tmp_path,
                    "format_version: 2\n"
                    "studies:\n"
                    "  study-a:\n"
                    "    env:\n"
                    "      DB_USER: admin\n"
                    "    secret_env:\n"
                    "      DB_USER: {source: study-db, key: username}\n",
                )
            )

    @pytest.mark.parametrize(
        "section, entry",
        [
            ("env", "PYTHONPATH: /opt/site"),
            ("env", "NVFL_WORKSPACE_TRANSFER_TOKEN: site-value"),
            ("secret_env", "NVFL_WORKSPACE_TRANSFER_TOKEN: {source: study-db, key: token}"),
            ("secret_env", "NVFL_WORKSPACE_OWNER_FQCN: {source: study-db, key: fqcn}"),
        ],
    )
    def test_launcher_owned_env_names_rejected(self, tmp_path, section, entry):
        with pytest.raises(ValueError, match="launcher-owned"):
            load_study_runtime_file(
                _write(
                    tmp_path,
                    "format_version: 2\n" "studies:\n" "  study-a:\n" f"    {section}:\n" f"      {entry}\n",
                )
            )

    def test_env_value_must_be_scalar(self, tmp_path):
        with pytest.raises(ValueError, match="scalar"):
            load_study_runtime_file(
                _write(tmp_path, "format_version: 2\nstudies:\n  study-a:\n    env:\n      DB: {host: x}\n")
            )

    def test_container_name_must_be_rfc1123(self, tmp_path):
        with pytest.raises(ValueError, match="RFC 1123"):
            load_study_runtime_file(
                _write(tmp_path, "format_version: 2\nstudies:\n  study-a:\n    container:\n      name: bad_name\n")
            )

    def test_secret_mount_mode_rw_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="read-only"):
            load_study_runtime_file(
                _write(
                    tmp_path,
                    "format_version: 2\n"
                    "studies:\n"
                    "  study-a:\n"
                    "    secret_mounts:\n"
                    "      db-ca:\n"
                    "        source: study-db-ca\n"
                    "        mount_path: /secrets\n"
                    "        mode: rw\n",
                )
            )

    def test_secret_mount_relative_mount_path_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="absolute"):
            load_study_runtime_file(
                _write(
                    tmp_path,
                    "format_version: 2\n"
                    "studies:\n"
                    "  study-a:\n"
                    "    secret_mounts:\n"
                    "      db-ca:\n"
                    "        source: study-db-ca\n"
                    "        mount_path: secrets/db-ca\n",
                )
            )

    def test_pod_template_rejected_when_not_allowed(self, tmp_path):
        with pytest.raises(ValueError, match="Kubernetes-only"):
            load_study_runtime_file(
                _write(tmp_path, "format_version: 2\nstudies:\n  study-a:\n    pod_template:\n      spec: {}\n"),
                allow_pod_template=False,
            )

    def test_pod_template_path_escape_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="relative path"):
            load_study_runtime_file(
                _write(
                    tmp_path,
                    "format_version: 2\nstudies:\n  study-a:\n    pod_template: ../outside/pod.yaml\n",
                )
            )

    def test_pod_template_missing_file_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            load_study_runtime_file(
                _write(tmp_path, "format_version: 2\nstudies:\n  study-a:\n    pod_template: missing.yaml\n")
            )

    def test_non_pod_template_rejected(self, tmp_path):
        (tmp_path / "job.yaml").write_text("kind: Job\n", encoding="utf-8")
        with pytest.raises(ValueError, match="kind: Pod"):
            load_study_runtime_file(
                _write(tmp_path, "format_version: 2\nstudies:\n  study-a:\n    pod_template: job.yaml\n")
            )

    def test_invalid_study_name_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="study name"):
            load_study_runtime_file(_write(tmp_path, "format_version: 2\nstudies:\n  Bad/Name: {}\n"))

    def test_empty_studies_warns(self, tmp_path, caplog):
        logger = logging.getLogger("study-runtime-test")
        with caplog.at_level(logging.WARNING):
            runtime_map = load_study_runtime_file(_write(tmp_path, "format_version: 2\n"), logger=logger)
        assert runtime_map == {}
        assert "no study entries" in caplog.text


class TestResolveStudyRuntime:
    def test_returns_entry_for_known_study(self, tmp_path):
        runtime_map = load_study_runtime_file(
            _write(tmp_path, "format_version: 2\nstudies:\n  study-a:\n    env:\n      A: b\n")
        )
        runtime = resolve_study_runtime(runtime_map, "study-a", "study_runtime.yaml")
        assert runtime.env == {"A": "b"}

    def test_returns_empty_runtime_and_warns_for_unknown_study(self, tmp_path, caplog):
        logger = logging.getLogger("study-runtime-test")
        runtime_map = load_study_runtime_file(
            _write(tmp_path, "format_version: 2\nstudies:\n  other:\n    env:\n      A: b\n")
        )
        with caplog.at_level(logging.WARNING):
            runtime = resolve_study_runtime(runtime_map, "study-a", "study_runtime.yaml", logger=logger)
        assert runtime.env == {}
        assert runtime.datasets == []
        assert runtime.pod_template is None
        assert "has no entry for study 'study-a'" in caplog.text


def test_study_runtime_file_path():
    assert study_runtime_file_path("/ws") == "/ws/local/study_runtime.yaml"
