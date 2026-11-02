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

import copy

import pytest

from nvflare.app_opt.job_launcher.k8s_pod import build_pod_manifest


@pytest.mark.parametrize("use_template", [False, True])
def test_manifests_own_their_configuration_without_mutating_site_policy(use_template):
    config = {
        "name": "worker",
        "image": "site/image",
        "command": "worker.module",
        "resources": {"limits": {"nvidia.com/gpu": "1"}},
        "volume_list": [{"name": "workspace", "persistentVolumeClaim": {"claimName": "site-data"}}],
        "volume_mount_list": [{"name": "workspace", "mountPath": "/workspace"}],
        "security_context": {"runAsUser": 1000},
    }
    template = {"spec": {"containers": [{"name": "nvflare_job"}, {"name": "sidecar", "image": "site/sidecar"}]}}
    template = template if use_template else None
    original_config, original_template = copy.deepcopy(config), copy.deepcopy(template)
    manifest = build_pod_manifest(config, "/image/python", template)
    original_manifest = copy.deepcopy(manifest)
    manifest["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = "4"
    manifest["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"] = "other"
    manifest["spec"]["securityContext"]["runAsUser"] = 2000
    if use_template:
        manifest["spec"]["containers"][1]["image"] = "other/sidecar"
    assert config == original_config
    assert template == original_template
    assert build_pod_manifest(config, "/image/python", template) == original_manifest
