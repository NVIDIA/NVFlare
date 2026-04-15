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

from nvflare.app_opt.job_launcher.k8s_launcher import K8sJobHandle, PvName, _volume_mount_list, uuid4_to_rfc1123


class TestVolumeMountList:
    def test_data_read_only_by_default(self):
        mounts = _volume_mount_list()
        data_mount = next(m for m in mounts if m["name"] == PvName.DATA.value)
        assert data_mount["readOnly"] is True

    def test_data_read_only_true(self):
        mounts = _volume_mount_list(data_read_only=True)
        data_mount = next(m for m in mounts if m["name"] == PvName.DATA.value)
        assert data_mount["readOnly"] is True

    def test_data_read_only_false(self):
        mounts = _volume_mount_list(data_read_only=False)
        data_mount = next(m for m in mounts if m["name"] == PvName.DATA.value)
        assert data_mount["readOnly"] is False

    def test_workspace_always_writable(self):
        mounts = _volume_mount_list()
        ws_mount = next(m for m in mounts if m["name"] == PvName.WORKSPACE.value)
        assert "readOnly" not in ws_mount

    def test_mount_paths(self):
        mounts = _volume_mount_list()
        ws_mount = next(m for m in mounts if m["name"] == PvName.WORKSPACE.value)
        data_mount = next(m for m in mounts if m["name"] == PvName.DATA.value)
        assert ws_mount["mountPath"] == "/var/tmp/nvflare/workspace"
        assert data_mount["mountPath"] == "/var/tmp/nvflare/data"


class TestUuid4ToRfc1123:
    def test_lowercase(self):
        assert uuid4_to_rfc1123("A1B2C3D4") == "a1b2c3d4"

    def test_strips_invalid_chars(self):
        assert uuid4_to_rfc1123("abc_def.ghi") == "abcdefghi"

    def test_prefix_when_starts_with_digit(self):
        assert uuid4_to_rfc1123("123abc")[0] == "j"

    def test_truncates_to_63(self):
        long = "a" * 100
        assert len(uuid4_to_rfc1123(long)) <= 63

    def test_strips_trailing_hyphens_after_truncation(self):
        name = "a" * 62 + "-x"
        result = uuid4_to_rfc1123(name)
        assert not result.endswith("-")
        assert len(result) <= 63


class TestJobHandleSecurityContext:
    def _make_handle(self, security_context=None):
        job_config = {
            "name": "test-job",
            "image": "test:latest",
            "command": "nvflare.test.module",
            "volume_mount_list": _volume_mount_list(),
            "volume_list": [],
        }
        if security_context:
            job_config["security_context"] = security_context
        return K8sJobHandle("test-job", None, job_config, namespace="test")

    def test_no_security_context_by_default(self):
        handle = self._make_handle()
        manifest = handle.get_manifest()
        assert "securityContext" not in manifest["spec"]

    def test_security_context_applied(self):
        ctx = {"seLinuxOptions": {"type": "spc_t"}}
        handle = self._make_handle(security_context=ctx)
        manifest = handle.get_manifest()
        assert manifest["spec"]["securityContext"] == ctx

    def test_security_context_empty_dict_not_applied(self):
        handle = self._make_handle(security_context={})
        manifest = handle.get_manifest()
        assert "securityContext" not in manifest["spec"]
