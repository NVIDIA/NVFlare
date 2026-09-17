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

"""Security regressions against callable package APIs; no new dependencies."""

import base64
import copy
import gzip
import hashlib
import json
import subprocess
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
import yaml

from nvflare.lighter.cc_provision import kbs_audience
from nvflare.lighter.cc_provision.impl.coco import validate_coco_config
from nvflare.lighter.cc_provision.impl.coco_packager import CoCoPackager
from nvflare.lighter.cc_provision.workload_security import HANDOFF_FILES, authenticate_handoff, validate_policy
from tests.unit_test.lighter.cc_provision.impl.test_coco import setup_project, write_fake_result
from tests.unit_test.lighter.cc_provision.impl.workload_security_context_test import (
    IMAGE,
    context,
    policy,
    policy_data,
)


@pytest.fixture
def handoff(tmp_path):
    source = tmp_path / "incoming"
    source.mkdir()
    for name in HANDOFF_FILES:
        path = source / name
        path.write_bytes(b"k" * 32 if name == "image_key" else b"reviewed")
        path.chmod(0o600)
    manifest = "".join(
        hashlib.sha256((source / name).read_bytes()).hexdigest() + "  " + name + "\n" for name in sorted(HANDOFF_FILES)
    )
    (source / "SHA256SUMS").write_text(manifest)
    return source, hashlib.sha256(manifest.encode()).hexdigest()


def test_handoff_uses_private_authenticated_snapshot(tmp_path, handoff):
    source, pin = handoff
    dest = tmp_path / "checked"
    authenticate_handoff(source, dest, pin)
    (source / "cosign.pub").write_bytes(b"replaced later")
    assert (dest / "cosign.pub").read_bytes() == b"reviewed"
    assert dest.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in dest.iterdir())


@pytest.mark.parametrize("attack", ["manifest", "payload", "extra", "symlink", "keymode", "missing"])
def test_handoff_rejects_untrusted_changes(tmp_path, handoff, attack):
    source, pin = handoff
    if attack == "manifest":
        (source / "SHA256SUMS").write_text("self-approved\n")
    elif attack == "payload":
        (source / "cosign.pub").write_text("changed")
    elif attack == "extra":
        (source / "unapproved").touch()
    elif attack == "symlink":
        (source / "cosign.pub").unlink()
        (source / "cosign.pub").symlink_to(source / "image_key")
    elif attack == "keymode":
        (source / "image_key").chmod(0o644)
    else:
        (source / "image_key").unlink()
    with pytest.raises((ValueError, OSError)):
        authenticate_handoff(source, tmp_path / "checked", pin)
    assert not (tmp_path / "checked").exists()


@pytest.mark.parametrize("grant", [["*"], ["nvflare."], [""], [None], "*", None, [1], ["a.*"]])
def test_unbounded_class_grants_rejected(tmp_path, grant):
    _, config = setup_project(tmp_path)
    config["class_allow_list"] = grant
    with pytest.raises(ValueError, match="class_allow_list"):
        validate_coco_config(config)


def test_reviewed_explicit_class_is_allowed(tmp_path):
    _, config = setup_project(tmp_path)
    config["class_allow_list"] = ["my_app.executor.ReviewedExecutor"]
    validate_coco_config(config)


@pytest.mark.parametrize(
    "attack",
    [
        "hostNetwork",
        "hostPID",
        "hostIPC",
        "volumes",
        "serviceAccountName",
        "privileged",
        "uid",
        "capabilities",
        "annotation",
        "extra_container",
        "initdata",
        "command",
        "extra_oci",
        "commented_guard",
        "dead_guard",
    ],
)
def test_packager_rejects_unapproved_builder_output(tmp_path, attack):
    _, config = setup_project(tmp_path)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"result_file": str(tmp_path / "result.json")}))
    write_fake_result(request)
    path = tmp_path / "protected-pod.yaml"
    pod = yaml.safe_load(path.read_text())
    CoCoPackager.validate_pod(path, config)
    spec = pod["spec"]
    container = spec["containers"][0]
    annotation = "io.katacontainers.config.hypervisor.cc_init_data"
    if attack in ("hostNetwork", "hostPID", "hostIPC"):
        spec[attack] = True
    elif attack in ("volumes", "serviceAccountName"):
        spec[attack] = []
    elif attack == "privileged":
        container["securityContext"]["privileged"] = True
    elif attack == "uid":
        container["securityContext"]["runAsUser"] = 0
    elif attack == "capabilities":
        container["securityContext"]["capabilities"]["add"] = ["SYS_ADMIN"]
    elif attack == "annotation":
        pod["metadata"]["annotations"]["io.katacontainers.config.hypervisor.kernel_params"] = "debug"
    elif attack == "extra_container":
        spec["containers"].append(copy.deepcopy(container))
    elif attack == "initdata":
        pod["metadata"]["annotations"][annotation] = "not-initdata"
    elif attack == "command":
        container["command"] = ["/bin/sh"]
    else:
        from nvflare.lighter.cc_provision.workload_security import decode_initdata

        rules = decode_initdata(pod["metadata"]["annotations"][annotation])["data"]["policy.rego"]
        if attack == "extra_oci":
            preamble, raw = rules.split("policy_data := ")
            data = json.loads(raw)
            data["containers"].append(copy.deepcopy(data["containers"][1]))
            rules = preamble + "policy_data := " + json.dumps(data)
        else:
            guard = "p_oci.Root.Readonly == i_oci.Root.Readonly"
            rules = rules.replace(guard, "# " + guard if attack == "commented_guard" else "true # " + guard)
        raw = '[data]\n"policy.rego" = ' + "'''\n" + rules + "\n'''\n"
        pod["metadata"]["annotations"][annotation] = base64.b64encode(gzip.compress(raw.encode())).decode()
    path.write_text(yaml.safe_dump(pod))
    with pytest.raises((ValueError, KeyError)):
        CoCoPackager.validate_pod(path, config)


@pytest.mark.parametrize("attack", ["uid", "args", "caps", "nnp", "root", "image", "expansion"])
def test_non_application_oci_is_validated(attack):
    data = policy_data()
    pause = data["containers"][1]["OCI"]
    if attack == "uid":
        pause["Process"]["User"]["UID"] = 0
    elif attack == "args":
        pause["Process"]["Args"] = ["/bin/sh"]
    elif attack == "caps":
        pause["Process"]["Capabilities"]["Effective"] = ["CAP_SYS_ADMIN"]
    elif attack == "nnp":
        pause["Process"]["NoNewPrivileges"] = False
    elif attack == "root":
        pause["Root"]["Readonly"] = False
    elif attack == "image":
        data["cluster_config"]["pause_container_image"] = "untrusted"
    else:
        data["common"]["default_caps"] = ["CAP_SYS_ADMIN"]
    with pytest.raises(ValueError):
        validate_policy(policy(data), IMAGE, context())


def test_audience_probe_signs_tokens_and_checks_all_four_http_results(tmp_path):
    key = tmp_path / "admin.key"
    subprocess.run(["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(key)], check=True, capture_output=True)
    observed = []

    def respond(request, timeout):
        header = request.get_header("Authorization")
        claims = json.loads(base64.urlsafe_b64decode(header.split(".")[1] + "===")) if header else {}
        observed.append(claims)
        if claims.get("aud") != ["KBS"]:
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)
        response = MagicMock()
        response.__enter__.return_value.status = 200
        return response

    opener = MagicMock()
    opener.open.side_effect = respond
    with (
        patch.object(kbs_audience.ssl, "create_default_context"),
        patch.object(kbs_audience.urllib.request, "build_opener", return_value=opener),
    ):
        kbs_audience.probe("https://secure.example", "ca.pem", key)
    assert [item.get("aud") for item in observed] == [["KBS"], ["not-KBS"], None, None]
    assert observed[2]["iss"] == "TrusteeInDocker"


def test_audience_probe_fails_when_kbs_accepts_wrong_audience():
    opener = MagicMock()
    opener.open.return_value.__enter__.return_value.status = 200
    with (
        patch.object(kbs_audience.ssl, "create_default_context"),
        patch.object(kbs_audience, "admin_token", return_value="token"),
        patch.object(kbs_audience.urllib.request, "build_opener", return_value=opener),
    ):
        with pytest.raises(RuntimeError, match="wrong audience"):
            kbs_audience.probe("https://secure.example", "ca.pem", "key")
