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

"""Exercise the documented launch-file preparation without calling Azure."""

import base64
import json
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
GUIDE = ROOT / "docs/user_guide/confidential_computing/azure/confidential_azure_container_instances_deployment.rst"


def _launch_script():
    guide = GUIDE.read_text()
    command = guide.index("   python3 - aci-template.json /path/to/startup-kit.tar")
    script = guide[command:].split("<<'PY'\n", 1)[1].split("\n   PY", 1)[0]
    return compile(textwrap.dedent(script), str(GUIDE), "exec")


def test_launch_preparation_keeps_credentials_in_secure_parameters(tmp_path, monkeypatch):
    template = tmp_path / "aci-template.json"
    source = (ROOT / "docs/resources/ccprep.json").read_bytes()
    template.write_bytes(source)
    archive = tmp_path / "startup-kit.tar"
    archive.write_bytes(b"private startup archive containing server.key\x00\xff")
    signature = tmp_path / "startup-kit.sig"
    signature.write_bytes(b"detached deployment signature\x00\xff")
    policy = tmp_path / "approved_cce_policy.txt"
    policy.write_bytes(b"exact approved policy\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prepare-launch", *map(str, (template, archive, signature, policy))])

    exec(_launch_script(), {"__name__": "__main__"})

    launch = tmp_path / "aci-launch.json"
    document = json.loads(launch.read_text())
    parameters_file = tmp_path / "aci-launch.parameters.json"
    parameters = json.loads(parameters_file.read_text())["parameters"]
    properties = document["resources"][0]["properties"]
    volume = next(item for item in properties["volumes"] if item["name"] == "nvflare-startup")
    for name, parameter, path in (
        ("startup-kit.tar", "startupKitArchive", archive),
        ("startup-kit.sig", "startupKitSignature", signature),
    ):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        assert volume["secret"][name] == f"[parameters('{parameter}')]"
        assert document["parameters"][parameter]["type"] == "secureString"
        assert "defaultValue" not in document["parameters"][parameter]
        assert parameters[parameter]["value"] == encoded
        assert encoded not in launch.read_text()
        assert encoded not in json.dumps(document.get("outputs", {}))
    assert base64.b64decode(properties["confidentialComputeProperties"]["ccePolicy"]) == policy.read_bytes()
    assert parameters_file.stat().st_mode & 0o777 == 0o600
    assert template.read_bytes() == source


@pytest.mark.parametrize("unsafe", ["literal", "ordinary_parameter", "secret_default"])
def test_launch_preparation_refuses_templates_that_expose_credentials(tmp_path, monkeypatch, unsafe):
    document = json.loads((ROOT / "docs/resources/ccprep.json").read_text())
    if unsafe == "literal":
        document["resources"][0]["properties"]["volumes"][0]["secret"]["startup-kit.tar"] = "literal-secret"
    elif unsafe == "ordinary_parameter":
        document["parameters"]["startupKitArchive"]["type"] = "string"
    else:
        document["parameters"]["startupKitArchive"]["defaultValue"] = "literal-secret"
    template = tmp_path / "aci-template.json"
    template.write_text(json.dumps(document))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prepare-launch", str(template), "archive", "signature", "policy"])

    with pytest.raises(ValueError, match="secure"):
        exec(_launch_script(), {"__name__": "__main__"})

    assert not (tmp_path / "aci-launch.json").exists()
    assert not (tmp_path / "aci-launch.parameters.json").exists()
