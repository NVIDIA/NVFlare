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
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
COCO_DIR = REPO_ROOT / "examples" / "devops" / "CoCo"
COMMON_SCRIPT = COCO_DIR / "lib" / "common.sh"


def _run_common(script: str, **environment: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(environment)
    env["COMMON_SCRIPT"] = str(COMMON_SCRIPT)
    return subprocess.run(
        ["bash", "-c", 'source "$COMMON_SCRIPT"\n' + script],
        capture_output=True,
        env=env,
        text=True,
        timeout=5,
    )


def test_managed_namespace_accepts_existing_owned_namespace():
    result = _run_common(
        """
kctl() {
  if [[ "$*" == *jsonpath* ]]; then
    printf '%s' "$FAKE_LABEL"
  else
    return 0
  fi
}
ensure_coco_managed_namespace nvflare-coco
""",
        FAKE_LABEL="true",
    )

    assert result.returncode == 0, result.stderr


def test_managed_namespace_rejects_existing_unowned_namespace():
    result = _run_common(
        """
kctl() {
  if [[ "$*" == *jsonpath* ]]; then
    printf '%s' "$FAKE_LABEL"
  else
    return 0
  fi
}
ensure_coco_managed_namespace shared-namespace
""",
        FAKE_LABEL="",
    )

    assert result.returncode != 0
    assert "refusing to adopt a potentially shared namespace" in result.stderr


def test_managed_namespace_creates_new_namespace_with_ownership_label():
    result = _run_common(
        """
kctl() {
  if [[ "$1" == get ]]; then
    return 1
  fi
  if [[ "$1" == create ]]; then
    cat
    return 0
  fi
  return 2
}
ensure_coco_managed_namespace nvflare-coco
"""
    )

    assert result.returncode == 0, result.stderr
    assert "name: nvflare-coco" in result.stdout
    assert 'nvflare.nvidia.com/coco-managed: "true"' in result.stdout


def test_credential_and_deployment_stages_use_namespace_ownership_guard():
    stage_30 = (COCO_DIR / "30-deploy-security-services.sh").read_text()
    stage_50 = (COCO_DIR / "50-nvflare-deploy.sh").read_text()

    assert 'ensure_coco_managed_namespace "$NVFLARE_NAMESPACE"' in stage_30
    assert 'ensure_coco_managed_namespace "$NVFLARE_NAMESPACE"' in stage_50
    assert 'create namespace "$NVFLARE_NAMESPACE"' not in stage_30
    assert 'create namespace "$NVFLARE_NAMESPACE"' not in stage_50
