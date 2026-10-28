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

"""Keep published confidential-computing examples aligned with executable inputs."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[5]
BUILDER = ROOT / "nvflare/lighter/cc/image_builder"
CVM_EXAMPLES = ROOT / "examples/advanced/cvm_builder"
COCO_EXAMPLES = ROOT / "examples/devops/coco/provision"


def test_vault_example_matches_the_checked_in_profile_output():
    profile = yaml.safe_load((BUILDER / "config/cvm_profile.yml").read_text())
    vault = yaml.safe_load((BUILDER / "config/vault_build.yml").read_text())
    assert vault["cvm_image"] == f"../target/cvm_{profile['profile_version']}"
    assert vault["docker_archive"].endswith("/application.tar")
    assert profile["kbs_url"].startswith("https://trustee.example.org:")
    assert "kbs.example.org" not in "".join(
        path.read_text() for path in BUILDER.rglob("*") if path.is_file() and path.suffix in {".md", ".py", ".yml"}
    )


def test_docker_example_emits_the_archive_referenced_by_participants():
    script = (CVM_EXAMPLES / "docker/docker_build.sh").read_text()
    guide = (CVM_EXAMPLES / "docker/README.md").read_text()
    assert 'docker save --output "$OUTPUT_ARCHIVE"' in script
    assert "gzip" not in script
    assert "output_application.tar" in script
    assert "/srv/nvflare/images/application.tar" in guide
    for path in CVM_EXAMPLES.rglob("cc_*.yml"):
        config = yaml.safe_load(path.read_text())
        if "workload" in config:
            assert config["workload"]["source"]["path"] == "/srv/nvflare/images/application.tar"


def test_all_published_cc_yaml_is_parseable_and_uses_closed_enums():
    paths = sorted(CVM_EXAMPLES.rglob("*.yml")) + sorted(COCO_EXAMPLES.rglob("*.yml"))
    assert paths
    for path in paths:
        value = yaml.safe_load(path.read_text())
        assert isinstance(value, dict), path
        if "cc_deployment_mode" not in value:
            continue
        assert value["cc_deployment_mode"] in {"bare_metal_cvm", "coco", "azure_cc"}, path
        assert value["cpu_tee"] in {"intel_tdx", "amd_sev_snp"}, path
        assert value["gpu_tee"] in {"none", "nvidia_cc"}, path
        source = value["workload"]["source"]
        assert source["type"] in {"docker_archive", "docker_build", "external"}, path
        if source["type"] == "docker_build":
            assert (path.parent / source["context"] / source["dockerfile"]).is_file(), path


def test_rendered_guide_includes_validated_participant_examples():
    guide = (ROOT / "docs/user_guide/confidential_computing/deployment.rst").read_text()
    includes = re.findall(r"^\.\. literalinclude:: (.+)$", guide, re.MULTILINE)
    assert includes
    for include in includes:
        path = (ROOT / "docs/user_guide/confidential_computing" / include).resolve()
        assert path.is_file(), path
        assert isinstance(yaml.safe_load(path.read_text()), dict)


def test_cpu_only_example_is_the_minimal_same_profile_topology():
    root = CVM_EXAMPLES / "cpu_only"
    server = yaml.safe_load((root / "cc_server.yml").read_text())
    client = yaml.safe_load((root / "cc_site-1.yml").read_text())
    assert server["cpu_tee"] == client["cpu_tee"] == "amd_sev_snp"
    assert server["gpu_tee"] == client["gpu_tee"] == "none"
    assert server["bare_metal_cvm"]["cvm_image"] == client["bare_metal_cvm"]["cvm_image"]
