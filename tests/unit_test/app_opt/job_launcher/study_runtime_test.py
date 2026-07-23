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

from nvflare.app_opt.job_launcher.study_runtime import load_study_runtime_file as _load_study_runtime_file
from nvflare.app_opt.job_launcher.study_runtime import resolve_study_runtime, study_runtime_file_path


def _write(tmp_path, text):
    file_path = tmp_path / "study_runtime.yaml"
    file_path.write_text(text, encoding="utf-8")
    return str(file_path)


def load_study_runtime_file(file_path, launcher_mode="k8s", logger=None):
    """Keep individual parser tests concise while selecting their normal K8s context."""
    return _load_study_runtime_file(file_path, launcher_mode=launcher_mode, logger=logger)


FULL_EXAMPLE = """
format_version: 2

studies:
  lung-cancer:
    container:
      image: registry.example.com/lung-cancer:v3

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

SLURM_EXAMPLE = """
format_version: 2
studies:
  pathology:
    container:
      image: /lustre/images/pathology.sif
    datasets:
      training: {source: /lustre/data/pathology, mode: ro}
    env:
      EMPTY_VALUE: ""
      MULTILINE: |-
        first line
        second line
    secret_env:
      DB_PASSWORD: {source: PATH_DB_PASSWORD}
      LEGACY_SECRET: {source: PATH_LEGACY_SECRET, key: ignored-on-slurm}
    secret_mounts:
      tls:
        source: /lustre/secrets/pathology
        mount_path: /run/secrets/pathology
    slurm:
      sandbox: apptainer
      setup: |
        module load cuda
      partition: gpu
      account: 1234
      qos: normal
"""


class TestLoadStudyRuntimeFile:
    def test_parses_full_example(self, tmp_path):
        (tmp_path / "lung_cancer_pod.yaml").write_text(
            "apiVersion: v1\nkind: Pod\nspec:\n  serviceAccountName: study-sa\n", encoding="utf-8"
        )
        runtime_map = load_study_runtime_file(_write(tmp_path, FULL_EXAMPLE))

        runtime = runtime_map["lung-cancer"]
        assert runtime.container_image == "registry.example.com/lung-cancer:v3"
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

    def test_parses_slurm_example(self, tmp_path):
        runtime = load_study_runtime_file(_write(tmp_path, SLURM_EXAMPLE), launcher_mode="slurm")["pathology"]

        assert runtime.container_image == "/lustre/images/pathology.sif"
        assert runtime.datasets[0].source == "/lustre/data/pathology"
        assert runtime.env == {"EMPTY_VALUE": "", "MULTILINE": "first line\nsecond line"}
        secret_env = {ref.name: ref for ref in runtime.secret_env}
        assert secret_env["DB_PASSWORD"].key is None
        assert secret_env["LEGACY_SECRET"].key == "ignored-on-slurm"
        assert runtime.secret_mounts[0].source == "/lustre/secrets/pathology"
        assert runtime.slurm == {
            "sandbox": "apptainer",
            "setup": "module load cuda\n",
            "partition": "gpu",
            "account": 1234,
            "qos": "normal",
        }

    @pytest.mark.parametrize(
        "body, error",
        [
            ("pod_template:\n      spec: {}", "Kubernetes-only"),
            ("docker_kwargs:\n      shm_size: 8g", "Docker-only"),
            (
                "secret_mounts:\n"
                "      tls:\n"
                "        source: /secrets/tls\n"
                "        mount_path: /run/secrets/tls\n"
                "        items: {cert: cert}",
                "Kubernetes-only",
            ),
        ],
    )
    def test_slurm_rejects_other_launcher_fields(self, tmp_path, body, error):
        with pytest.raises(ValueError, match=error):
            load_study_runtime_file(
                _write(tmp_path, f"format_version: 2\nstudies:\n  study-a:\n    {body}\n"),
                launcher_mode="slurm",
            )

    @pytest.mark.parametrize("launcher_mode", ["k8s", "docker"])
    def test_existing_launchers_still_require_secret_key(self, tmp_path, launcher_mode):
        with pytest.raises(ValueError, match=r"secret_env\.PASSWORD\.key"):
            load_study_runtime_file(
                _write(
                    tmp_path,
                    "format_version: 2\n"
                    "studies:\n"
                    "  study-a:\n"
                    "    secret_env:\n"
                    "      PASSWORD: {source: PASSWORD_SOURCE}\n",
                ),
                launcher_mode=launcher_mode,
            )

    @pytest.mark.parametrize("launcher_mode", ["process", "k8s", "docker"])
    def test_slurm_block_rejected_by_other_launchers(self, tmp_path, launcher_mode):
        with pytest.raises(ValueError, match="Slurm-only"):
            load_study_runtime_file(
                _write(
                    tmp_path,
                    "format_version: 2\nstudies:\n  study-a:\n    slurm:\n      partition: gpu\n",
                ),
                launcher_mode=launcher_mode,
            )

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

    def test_launcher_mode_is_required(self, tmp_path):
        with pytest.raises(TypeError):
            _load_study_runtime_file(_write(tmp_path, "format_version: 2\nstudies: {}\n"))

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
            ("env", "NVFLARE_JOB_AUTH_TOKEN: site-value"),
            ("secret_env", "NVFLARE_JOB_TOKEN_SIGNATURE: {source: study-db, key: sig}"),
            ("secret_env", "NVFLARE_JOB_SSID: {source: study-db, key: ssid}"),
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

    def test_docker_kwargs_parsed(self, tmp_path):
        runtime_map = load_study_runtime_file(
            _write(
                tmp_path,
                "format_version: 2\n"
                "studies:\n"
                "  study-a:\n"
                "    docker_kwargs:\n"
                "      shm_size: 8g\n"
                "      device_requests:\n"
                "        - Count: 1\n"
                "          Capabilities: [[gpu]]\n",
            ),
            launcher_mode="docker",
        )
        assert runtime_map["study-a"].docker_kwargs == {
            "shm_size": "8g",
            "device_requests": [{"Count": 1, "Capabilities": [["gpu"]]}],
        }

    def test_docker_kwargs_reserved_key_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="launcher-owned"):
            load_study_runtime_file(
                _write(
                    tmp_path,
                    "format_version: 2\nstudies:\n  study-a:\n    docker_kwargs:\n      image: sneaky:v1\n",
                ),
                launcher_mode="docker",
            )

    def test_docker_kwargs_rejected_by_other_launcher(self, tmp_path):
        with pytest.raises(ValueError, match="Docker-only"):
            load_study_runtime_file(
                _write(tmp_path, "format_version: 2\nstudies:\n  study-a:\n    docker_kwargs:\n      shm_size: 8g\n"),
                launcher_mode="k8s",
            )

    def test_env_value_must_be_scalar(self, tmp_path):
        with pytest.raises(ValueError, match="scalar"):
            load_study_runtime_file(
                _write(tmp_path, "format_version: 2\nstudies:\n  study-a:\n    env:\n      DB: {host: x}\n")
            )

    def test_env_value_must_not_be_empty(self, tmp_path):
        with pytest.raises(ValueError, match="must not be empty"):
            load_study_runtime_file(
                _write(tmp_path, 'format_version: 2\nstudies:\n  study-a:\n    env:\n      FOO: ""\n')
            )

    @pytest.mark.parametrize(
        "section, entry, error",
        [
            ("env", "BAD-NAME: value", "POSIX environment variable"),
            ("env", "_nvfl_secret: value", "launcher-owned"),
            ("env", "SLURM_EXPORT_ENV: value", "launcher-owned"),
            ("env", "NVFLARE_SLURM_HELPER_MASTER_PORT: value", "launcher-owned"),
            ("secret_env", "PASSWORD: {source: BAD-SOURCE}", "POSIX environment variable"),
            ("secret_env", "NVFL_APPTAINER: {source: SOURCE}", "launcher-owned"),
            ("secret_env", "NVFLARE_SLURM_CHILD_PROCESS: {source: SOURCE}", "launcher-owned"),
        ],
    )
    def test_slurm_env_names_are_validated(self, tmp_path, section, entry, error):
        with pytest.raises(ValueError, match=error):
            load_study_runtime_file(
                _write(
                    tmp_path,
                    "format_version: 2\n" "studies:\n" "  study-a:\n" f"    {section}:\n" f"      {entry}\n",
                ),
                launcher_mode="slurm",
            )

    @pytest.mark.parametrize("name", ["SLURM_JOB_ID", "APPTAINER_BIND", "NVFLARE_SLURM_CHILD_PROCESS", "UID"])
    def test_slurm_rejects_runtime_owned_environment_names(self, tmp_path, name):
        with pytest.raises(ValueError, match="launcher-owned"):
            load_study_runtime_file(
                _write(
                    tmp_path,
                    "format_version: 2\nstudies:\n  study-a:\n    env:\n" f"      {name}: site-value\n",
                ),
                launcher_mode="slurm",
            )

    @pytest.mark.parametrize(
        "body",
        [
            "container: {image: relative/image.sif}",
            "datasets:\n      training: {source: relative/data, mode: ro}",
            (
                "secret_mounts:\n"
                "      tls:\n"
                "        source: relative/secret\n"
                "        mount_path: /run/secrets/tls"
            ),
        ],
    )
    def test_slurm_sources_must_be_absolute(self, tmp_path, body):
        with pytest.raises(ValueError, match="absolute path"):
            load_study_runtime_file(
                _write(tmp_path, f"format_version: 2\nstudies:\n  study-a:\n    {body}\n"),
                launcher_mode="slurm",
            )

    @pytest.mark.parametrize(
        "slurm_entry, error",
        [
            ("backend: apptainer", "unknown key"),
            ("sandbox: docker", "must be one of"),
            ("sandbox: {name: none}", "must be one of"),
            ("partition: true", "non-empty string or integer"),
            ('partition: "gpu debug"', "must not contain whitespace"),
            ("partition: |\n        gpu\n        debug", "must not contain whitespace"),
        ],
    )
    def test_slurm_overrides_are_strict(self, tmp_path, slurm_entry, error):
        with pytest.raises(ValueError, match=error):
            load_study_runtime_file(
                _write(
                    tmp_path,
                    f"format_version: 2\nstudies:\n  study-a:\n    slurm:\n      {slurm_entry}\n",
                ),
                launcher_mode="slurm",
            )

    @pytest.mark.parametrize(
        "value, error",
        [
            ('"\\0"', "NUL"),
            ('"\\uD800"', "UTF-8"),
        ],
    )
    def test_slurm_env_values_must_be_utf8_text_without_nul(self, tmp_path, value, error):
        with pytest.raises(ValueError, match=error):
            load_study_runtime_file(
                _write(
                    tmp_path,
                    f"format_version: 2\nstudies:\n  study-a:\n    env:\n      VALUE: {value}\n",
                ),
                launcher_mode="slurm",
            )

    @pytest.mark.parametrize(
        "body",
        [
            "container: {image: /images/study.sif}",
            "datasets:\n      training: {source: /data/training, mode: ro}",
            ("secret_mounts:\n" "      tls:\n" "        source: /secrets/tls\n" "        mount_path: /run/secrets/tls"),
        ],
    )
    def test_explicit_slurm_sandbox_none_rejects_mount_bearing_fields(self, tmp_path, body):
        with pytest.raises(ValueError, match="sandbox 'none' does not support"):
            load_study_runtime_file(
                _write(
                    tmp_path,
                    f"format_version: 2\nstudies:\n  study-a:\n    {body}\n    slurm:\n      sandbox: none\n",
                ),
                launcher_mode="slurm",
            )

    def test_container_name_key_rejected(self, tmp_path):
        # the main container is identified via the nvflare_job template sentinel, not config
        with pytest.raises(ValueError, match="name"):
            load_study_runtime_file(
                _write(tmp_path, "format_version: 2\nstudies:\n  study-a:\n    container:\n      name: my-job\n")
            )

    def test_container_requires_image(self, tmp_path):
        with pytest.raises(ValueError, match="container.image"):
            load_study_runtime_file(_write(tmp_path, "format_version: 2\nstudies:\n  study-a:\n    container: {}\n"))

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

    def test_pod_template_rejected_by_other_launcher(self, tmp_path):
        with pytest.raises(ValueError, match="Kubernetes-only"):
            load_study_runtime_file(
                _write(tmp_path, "format_version: 2\nstudies:\n  study-a:\n    pod_template:\n      spec: {}\n"),
                launcher_mode="docker",
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
