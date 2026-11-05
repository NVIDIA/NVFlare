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

"""CoCo external-runtime contracts; run serially outside unit workers."""

import hashlib
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.coco_test_utils.process import COMMAND_TIMEOUT, run_coco_command
from tests.coco_test_utils.shell import require_coco_bash
from tests.coco_test_utils.workload_context import contract, final_workload_pod

ROOT = Path(__file__).resolve().parents[4] / "examples/devops/coco"

PASSWORD = "fixture-password-not-for-production"


@pytest.fixture(scope="module")
def credential_bash():
    bash = require_coco_bash()
    # The deployment target is Ubuntu; do not confuse absent GNU host tools
    # on a developer's machine with a failure of release configuration.
    for tool in ("realpath", "find", "install", "sort", "tr", "hostname", "mkdir", "chmod"):
        if not shutil.which(tool):
            pytest.skip(f"Publisher credential tests require {tool}")
    probe = run_coco_command(
        [bash, "-c", 'realpath -e -- . >/dev/null && find . -maxdepth 0 -printf "%f\\n" >/dev/null'],
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
    )
    if probe.returncode:
        pytest.skip(f"Publisher credential tests require GNU realpath/find: {probe.stderr.strip()}")
    return bash


@pytest.fixture
def credential_kit(tmp_path, credential_bash):
    kit = tmp_path / "admin"
    # Copy the real admin/shared sources so the test exercises configuration
    # loading and path derivation, not a replacement platform helper.
    shutil.copytree(ROOT / "admin", kit)
    shutil.copytree(ROOT / "shared", tmp_path / "shared")
    work_root = tmp_path / "custom state"
    config = (ROOT / "admin/platform.env.example").read_text() + (f"WORK_ROOT={shlex.quote(str(work_root))}\n")
    (kit / "platform.env").write_text(config)
    credentials = work_root / "secrets/registry"
    credentials.mkdir(parents=True, mode=0o700)
    (credentials / "username").write_text("coco-publisher\n")
    (credentials / "password").write_text(PASSWORD)
    for credential in credentials.iterdir():
        credential.chmod(0o600)
    (kit / "public").mkdir(exist_ok=True)
    for certificate in ("trustee.crt", "registry-ca.crt"):
        (kit / "public" / certificate).write_text("fixture-not-a-real-certificate\n")
    context = tmp_path / "build"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM scratch\n")
    (kit / "workload.env").write_text(
        "RELEASE_NAME=fixture-v1\n"
        f"BUILD_CONTEXT={shlex.quote(str(context))}\n"
        f"DOCKERFILE={shlex.quote(str(context / 'Dockerfile'))}\n"
        "REGISTRY_REPOSITORY=workloads/fixture\n"
        "REGISTRY_ENDPOINT=secure.nvflare.local:5000\n"
        f"REGISTRY_CA_FILE={shlex.quote(str(kit / 'public/registry-ca.crt'))}\n"
        f"REGISTRY_USERNAME_FILE={shlex.quote(str(work_root / 'secrets/registry/username'))}\n"
        f"REGISTRY_PASSWORD_FILE={shlex.quote(str(work_root / 'secrets/registry/password'))}\n"
        "KBS_URL=https://secure.nvflare.local:8443\n"
        "APP_COMMAND_JSON='[\"/coco-app\"]'\n"
        "APP_UID=65532\nAPP_GID=65532\n"
    )
    return kit, work_root


def run_release(credential_bash, credential_kit):
    kit, _ = credential_kit
    # Source the real release helper with supplied credential files. Keep its
    # private state inside the fixture and avoid touching the user's bin path.
    prelude = r"""
mkdir() {
    local item
    for item in "$@"; do
        [[ $item == -* ]] && continue
        [[ $item == "$TEST_ROOT/"* ]] || { echo "write outside test root: $item" >&2; return 99; }
        command mkdir -p -- "$item"
    done
}
chmod() {
    local item
    for item in "${@:2}"; do
        [[ $item == "$TEST_ROOT/"* ]] || { echo "write outside test root: $item" >&2; return 99; }
    done
    command chmod "$@"
}
source "$RELEASE_HELPER"
need_file "$REGISTRY_USERNAME_PATH"
need_file "$REGISTRY_PASSWORD_PATH"
printf 'RELEASE_USERNAME_PATH=%s\nRELEASE_PASSWORD_PATH=%s\n' "$REGISTRY_USERNAME_PATH" "$REGISTRY_PASSWORD_PATH"
printf 'RELEASE_REGISTRY_BASE=%s\nRELEASE_IMAGE_REPOSITORY=%s\n' "$(registry_base)" "$IMAGE_REPOSITORY"
for name in GENPOLICY RULES SETTINGS WORKLOAD_LAUNCH_PROFILE; do printf 'RELEASE_%s=%s\n' "$name" "${!name}"; done
"""
    env = dict(
        os.environ,
        TEST_ROOT=str(kit.parent),
        RELEASE_HELPER=str(kit / "lib/release.sh"),
        OWNER_CONFIG=str(kit / "workload.env"),
        PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
    )
    return run_coco_command(
        [credential_bash, "-c", prelude], env=env, capture_output=True, text=True, timeout=COMMAND_TIMEOUT
    )


def _inline_python(path, marker):
    source = path.read_text().split(marker, 1)[1]
    return source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]


@pytest.mark.parametrize("endpoint", ["registry.nvflare.local", "registry.nvflare.local:5443"])
@pytest.mark.skipif(sys.version_info < (3, 11), reason="CoCo InitData launch checks require Python 3.11+")
def test_release_authority_matches_launcher_and_trustee(credential_bash, credential_kit, tmp_path, endpoint):
    kit, _ = credential_kit
    config = kit / "workload.env"
    config.write_text(config.read_text().replace("secure.nvflare.local:5000", endpoint))
    result = run_release(credential_bash, credential_kit)
    assert result.returncode == 0, result.stdout + result.stderr
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if line.startswith("RELEASE_"))
    assert values["RELEASE_REGISTRY_BASE"] == endpoint
    repository = values["RELEASE_IMAGE_REPOSITORY"]
    assert repository == endpoint + "/workloads/fixture"
    image = repository + "@sha256:" + "a" * 64
    runtime = "kata-qemu-snp"

    # Exercise the launcher's real static checker with the real release identity.
    pod = final_workload_pod(runtime, image=image)
    pod["spec"]["containers"][0]["image"] = image
    pod_file = tmp_path / "pod.json"
    pod_file.write_text(json.dumps(pod))
    launcher = ROOT / "coco/50-launch-handoff.sh"
    launch = run_coco_command(
        [
            sys.executable,
            "-",
            str(pod_file),
            str(tmp_path / "snapshot.yaml"),
            str(tmp_path / "captured.json"),
            "",
            str(ROOT / "coco/lib/workload-security-context.py"),
        ],
        input=_inline_python(launcher, 'python3 - "$1"'),
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
    )
    assert launch.returncode == 0, launch.stderr
    assert "InitData policy checks" in launch.stdout

    # Exercise Trustee's full release-metadata reconstruction, including its
    # exact-authority restriction and service-owned policy template.
    from nvflare.lighter.cc_provision import workload_release

    incoming = tmp_path / "service-handoff"
    incoming.mkdir()
    paths = [f"default/{kind}/demo" for kind in ("image-key", "sig-public-key", "security-policy")]
    auth = workload_release.build_authorization(runtime, "demo", image, ["/coco-app"], "ab" * 32, paths)
    policies = ROOT / "service/policies"
    template = policies / "workload-resource-policy.rego.template"
    (incoming / "release-authorization.json").write_text(json.dumps(auth))
    (incoming / "resource-policy-fragment.rego").write_text(
        workload_release.render_fragment(auth, template.read_text())
    )
    (incoming / "cosign.pub").write_text("-----BEGIN PUBLIC KEY-----\nTEST\n-----END PUBLIC KEY-----\n")
    (incoming / "image-security-policy.json").write_text(
        json.dumps(
            {
                "default": [{"type": "reject"}],
                "transports": {
                    "docker": {
                        repository: [
                            {
                                "type": "sigstoreSigned",
                                "keyPath": "kbs:///default/sig-public-key/demo",
                                "signedIdentity": {"type": "matchRepository"},
                            }
                        ]
                    }
                },
            }
        )
    )
    service = runpy.run_path(str(ROOT / "service/lib/install-workload-release.py"))
    accepted, resources, fragment = service["release_inputs"](incoming)
    assert accepted["encrypted_image"] == image
    assert resources["image-key"] == "default/image-key/demo"
    assert fragment == (incoming / "resource-policy-fragment.rego").read_text()


@pytest.mark.parametrize(
    "endpoint",
    [
        "registry.nvflare.local/path",
        "registry.nvflare.local/",
        "https://registry.nvflare.local",
        "user@registry.nvflare.local",
        "user:password@registry.nvflare.local",
    ],
)
def test_release_rejects_registry_path_or_embedded_credentials(credential_bash, credential_kit, endpoint):
    kit, _ = credential_kit
    config = kit / "workload.env"
    config.write_text(config.read_text().replace("secure.nvflare.local:5000", endpoint))
    result = run_release(credential_bash, credential_kit)
    assert result.returncode != 0
    assert "RELEASE_IMAGE_REPOSITORY=" not in result.stdout


@pytest.mark.parametrize("override_request", [False, True])
def test_preinstalled_tool_paths_are_platform_inputs(credential_bash, credential_kit, override_request):
    kit, _ = credential_kit
    paths = {
        "GENPOLICY": kit.parent / "existing tools/genpolicy",
        "RULES": kit.parent / "existing tools/rules.rego",
        "SETTINGS": kit.parent / "existing tools/settings.json",
        "WORKLOAD_LAUNCH_PROFILE": kit.parent / "received launch profile.json",
    }
    config = kit / "platform.env"
    config.write_text(
        config.read_text() + "".join(f"{name}={shlex.quote(str(path))}\n" for name, path in paths.items())
    )
    if override_request:
        workload = kit / "workload.env"
        workload.write_text(workload.read_text() + "GENPOLICY=/unapproved/tool\n")
    result = run_release(credential_bash, credential_kit)
    if override_request:
        assert result.returncode != 0
        assert "readonly variable" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        values = dict(line.split("=", 1) for line in result.stdout.splitlines() if line.startswith("RELEASE_"))
        assert {name: values["RELEASE_" + name] for name in paths} == {name: str(path) for name, path in paths.items()}


def test_builder_rejects_missing_preinstalled_genpolicy_before_building(credential_bash, credential_kit, tmp_path):
    kit, _ = credential_kit
    config = kit / "platform.env"
    config.write_text(config.read_text() + f"GENPOLICY={shlex.quote(str(tmp_path / 'missing genpolicy'))}\n")
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema": "nvflare-coco-build-request/v1",
                "workload_env": str(kit / "workload.env"),
                "admin_dir": str(kit),
                "result_file": str(tmp_path / "result.json"),
            }
        )
    )
    tools = tmp_path / "bin"
    tools.mkdir()
    marker = tmp_path / "unexpected execution"
    for executable in ("docker", "skopeo", "cosign", "jq", "openssl"):
        command = tools / executable
        command.write_text("#!/bin/sh\nprintf 'unexpected' >> " + shlex.quote(str(marker)) + "\nexit 99\n")
        command.chmod(0o700)
    python = tools / "python3"
    python.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    python.chmod(0o700)
    result = run_coco_command(
        [credential_bash, str(kit / "build_coco_image.sh"), str(request)],
        env={**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
    )
    assert result.returncode != 0
    assert "GENPOLICY must name the preinstalled executable" in result.stderr
    assert not marker.exists()
    assert not (tmp_path / "result.json").exists()


def test_image_runner_proceeds_noninteractively_with_existing_tools(credential_bash, credential_kit, tmp_path):
    kit, work_root = credential_kit
    received_profile = tmp_path / "launch-profile.json"
    received_profile.write_text(json.dumps(contract()))
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("genpolicy", "rules.rego", "settings.json"):
        (tools / name).write_text("fixture")
    (tools / "genpolicy").chmod(0o700)
    platform = kit / "platform.env"
    platform.write_text(
        platform.read_text()
        + f"GENPOLICY={shlex.quote(str(tools / 'genpolicy'))}\n"
        + f"RULES={shlex.quote(str(tools / 'rules.rego'))}\n"
        + f"SETTINGS={shlex.quote(str(tools / 'settings.json'))}\n"
        + f"WORKLOAD_LAUNCH_PROFILE={shlex.quote(str(received_profile))}\n"
    )
    stage_calls = tmp_path / "stage-calls"
    stages = [
        "10-build-plaintext.sh",
        "20-encrypt-sign-publish.sh",
        "25-verify-published-image.sh",
        "30-generate-pod-and-policies.sh",
        "40-create-handoffs.sh",
    ]
    for name in stages:
        script = kit / name
        script.write_text(f'#!/bin/sh\nprintf \'{name}:%s\\n\' "$#" >> "$TEST_STAGE_CALLS"\n')
        if name == "40-create-handoffs.sh":
            script.write_text(
                script.read_text()
                + 'mkdir -p "$TEST_HANDOFF/coco-it" "$TEST_HANDOFF/trusted-service"\n'
                + 'printf fixture > "$TEST_HANDOFF/coco-it/fixture-v1-pod.yaml"\n'
                + 'printf fixture-key > "$TEST_HANDOFF/trusted-service/image_key"\n'
            )
        script.chmod(0o700)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    unexpected = tmp_path / "unexpected tool execution"
    for name in ("docker", "skopeo", "cosign", "jq", "openssl"):
        binary = binaries / name
        binary.write_text("#!/bin/sh\nprintf unexpected >> " + shlex.quote(str(unexpected)) + "\nexit 99\n")
        binary.chmod(0o700)
    python = binaries / "python3"
    python.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    python.chmod(0o700)
    result_file = tmp_path / "result.json"
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema": "nvflare-coco-build-request/v1",
                "workload_env": str(kit / "workload.env"),
                "admin_dir": str(kit),
                "result_file": str(result_file),
            }
        )
    )
    result = run_coco_command(
        [credential_bash, str(kit / "build_coco_image.sh"), str(request)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
            "TEST_STAGE_CALLS": str(stage_calls),
            "TEST_HANDOFF": str(work_root / "releases/fixture-v1/handoff"),
        },
        timeout=COMMAND_TIMEOUT,
    )
    assert result.returncode == 0, result.stderr
    assert stage_calls.read_text().splitlines() == [name + ":1" for name in stages]
    assert not unexpected.exists(), "all build and publishing operations must remain mocked"
    generated = json.loads(result_file.read_text())
    assert generated["schema"] == "nvflare-coco-build-result/v1"
    assert Path(generated["pod_yaml"]).is_file()
    assert (Path(generated["trusted_service"]) / "image_key").is_file()


@pytest.mark.parametrize("legacy_flag", [False, True])
def test_publish_script_accepts_workload_alone_without_an_approval_gate(credential_bash, tmp_path, legacy_flag):
    kit = tmp_path / "admin"
    (kit / "lib").mkdir(parents=True)
    script = kit / "20-encrypt-sign-publish.sh"
    shutil.copyfile(ROOT / "admin/20-encrypt-sign-publish.sh", script)
    (kit / "lib/release.sh").write_text("echo REACHED_RELEASE_CONFIGURATION\nexit 67\n")
    command = [credential_bash, str(script), "workload.env"]
    if legacy_flag:
        command.append("--approve-reviewed-plaintext")
    result = run_coco_command(
        command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=COMMAND_TIMEOUT
    )
    assert result.returncode == 67
    assert "REACHED_RELEASE_CONFIGURATION" in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [[], ["workload.env", "--unexpected"], ["workload.env", "--approve-reviewed-plaintext", "extra"]],
)
def test_publish_script_rejects_malformed_flags_before_loading_configuration(credential_bash, arguments):
    result = run_coco_command(
        [credential_bash, str(ROOT / "admin/20-encrypt-sign-publish.sh"), *arguments],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr


PUBLISH_TOOLS = r"""
import base64
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

tool = Path(sys.argv[0]).name
args = sys.argv[1:]
if tool == "sudo":
    if args[0] == "chown":
        sys.exit(0)
    os.execvp(args[0], args)
entry = {"tool": tool, "args": args}
if tool == "skopeo":
    remote = args[0] in ("login", "inspect") or "--digestfile" in args
    if remote:
        flag = "--dest-cert-dir" if args[0] == "copy" else "--cert-dir"
        assert flag in args
        directory = Path(args[args.index(flag) + 1])
        expected = Path(os.environ["TEST_REGISTRY_CA"]).read_bytes()
        assert (directory / "ca.crt").read_bytes() == expected
        tls = "--dest-tls-verify=true" if args[0] == "copy" else "--tls-verify=true"
        assert tls in args
        entry.update(
            cert_dir=str(directory),
            cert_mode=stat.S_IMODE((directory / "ca.crt").stat().st_mode),
            directory_mode=stat.S_IMODE(directory.stat().st_mode),
        )
with Path(os.environ["TEST_TOOL_CALLS"]).open("a") as stream:
    stream.write(json.dumps(entry) + "\n")
if tool == "docker":
    if args[:3] == ["image", "inspect", "--format"]:
        print(os.environ["TEST_PLAIN_IMAGE_ID"])
    elif args[:2] == ["image", "inspect"]:
        assert args[-1] == os.environ["TEST_PROVIDER_IMAGE"]
        present = os.environ["TEST_PROVIDER_CACHED"] == "1" or Path(os.environ["TEST_PROVIDER_CACHE"]).exists()
        if os.environ["TEST_POST_PULL_INSPECT_FAILURE"] == "1":
            present = False
        sys.exit(0 if present else 1)
    elif args[0] == "pull":
        assert args[1] == os.environ["TEST_PROVIDER_IMAGE"]
        status = int(os.environ["TEST_PULL_STATUS"])
        if not status:
            Path(os.environ["TEST_PROVIDER_CACHE"]).write_text("cached")
        sys.exit(status)
    elif args[0] == "run":
        assert "--pull=never" in args and os.environ["TEST_PROVIDER_IMAGE"] in args
    else:
        raise AssertionError("Unexpected Docker call")
elif tool == "skopeo":
    if args[0] == "login":
        assert sys.stdin.read() == os.environ["TEST_REGISTRY_PASSWORD"]
        Path(args[args.index("--authfile") + 1]).write_text('{"auths":{}}')
    elif args[0] == "copy":
        if "--digestfile" in args:
            Path(args[args.index("--digestfile") + 1]).write_text("sha256:" + "a" * 64)
        else:
            assert args[-2].startswith("docker-daemon:") and args[-1].startswith("dir:")
            assert "--cert-dir" not in args and "--dest-cert-dir" not in args
    elif args[0] == "inspect":
        annotation = base64.urlsafe_b64encode(
            b'{"kid":"kbs:///default/image-key/fixture-v1"}'
        ).decode()
        print(json.dumps({"layers": [{
            "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip+encrypted",
            "annotations": {"org.opencontainers.image.enc.keys.provider.attestation-agent": annotation},
        }]}))
    else:
        raise AssertionError("Unexpected Skopeo call")
elif tool == "cosign":
    assert args[0] in ("sign", "verify")
    print("{}")
elif tool == "curl":
    assert "--cacert" in args and os.environ["TEST_REGISTRY_CA"] in args
    sys.stdout.write("403")
else:
    raise AssertionError("Unexpected external tool execution")
"""


@pytest.fixture
def publishing(credential_kit, credential_bash, tmp_path):
    kit, work_root = credential_kit
    release = work_root / "releases/fixture-v1"
    private = release / "private"
    for directory in (private, release / "output", release / "review"):
        directory.mkdir(parents=True, mode=0o700)
    signing = work_root / "secrets/signing"
    signing.mkdir()
    (private / "image_key").write_bytes(b"k" * 32)
    (signing / "cosign.password").write_text("fixture-signing-password")
    (signing / "cosign.key").write_text("fixture-private-key")
    (signing / "cosign.pub").write_text("-----BEGIN PUBLIC KEY-----\nfixture\n-----END PUBLIC KEY-----\n")
    plain_id = "sha256:" + "1" * 64
    config_hash = hashlib.sha256((kit / "workload.env").read_bytes()).hexdigest()
    (private / "build-state.env").write_text(
        f"CONFIG_SHA256={config_hash}\nPLAIN_IMAGE=localhost/fixture:plain\nPLAIN_IMAGE_ID={plain_id}\n"
    )
    calls = tmp_path / "tool-calls.jsonl"
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in ("sudo", "docker", "skopeo", "cosign", "curl", "openssl", "jq"):
        script = binaries / name
        script.write_text(f"#!{sys.executable}\n" + PUBLISH_TOOLS)
        script.chmod(0o700)
    interpreter = binaries / "python3"
    interpreter.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    interpreter.chmod(0o700)
    provider = next(
        line.split("=", 1)[1].strip('"')
        for line in (kit / "platform.env").read_text().splitlines()
        if line.startswith("KEYPROVIDER_IMAGE=")
    )

    def run(cached=True, pull_status=0, post_pull_failure=False, provider_image=None):
        if provider_image is not None:
            config = kit / "platform.env"
            config.write_text(config.read_text() + f"KEYPROVIDER_IMAGE={shlex.quote(provider_image)}\n")
        environment = {
            **os.environ,
            "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
            "TEST_TOOL_CALLS": str(calls),
            "TEST_PLAIN_IMAGE_ID": plain_id,
            "TEST_PROVIDER_IMAGE": provider_image or provider,
            "TEST_PROVIDER_CACHED": str(int(cached)),
            "TEST_PROVIDER_CACHE": str(tmp_path / "provider-cache"),
            "TEST_PULL_STATUS": str(pull_status),
            "TEST_POST_PULL_INSPECT_FAILURE": str(int(post_pull_failure)),
            "TEST_REGISTRY_CA": str(kit / "public/registry-ca.crt"),
            "TEST_REGISTRY_PASSWORD": PASSWORD,
        }
        publish = run_coco_command(
            [credential_bash, str(kit / "20-encrypt-sign-publish.sh"), str(kit / "workload.env")],
            capture_output=True,
            text=True,
            env=environment,
            timeout=COMMAND_TIMEOUT,
            diagnostic_paths=(calls,),
        )
        verify = None
        if publish.returncode == 0:
            verify = run_coco_command(
                [credential_bash, str(kit / "25-verify-published-image.sh"), str(kit / "workload.env")],
                capture_output=True,
                text=True,
                env=environment,
                timeout=COMMAND_TIMEOUT,
                diagnostic_paths=(calls,),
            )
        operations = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        assert PASSWORD not in publish.stdout + publish.stderr
        return publish, verify, operations, release

    return run


@pytest.mark.parametrize("cached", [False, True])
def test_publishing_fetches_only_missing_pinned_tool_and_scopes_registry_ca(publishing, cached):
    publish, verify, operations, release = publishing(cached=cached)
    assert publish.returncode == 0, publish.stderr
    assert verify.returncode == 0, verify.stderr
    pulls = [entry for entry in operations if entry["tool"] == "docker" and entry["args"][0] == "pull"]
    assert len(pulls) == (0 if cached else 1)
    runs = [entry for entry in operations if entry["tool"] == "docker" and entry["args"][0] == "run"]
    assert len(runs) == 1 and "--pull=never" in runs[0]["args"]
    remote = [entry for entry in operations if "cert_dir" in entry]
    assert [entry["args"][0] for entry in remote] == ["login", "copy", "inspect", "inspect"]
    for entry in remote:
        assert entry["cert_dir"] == str(release / "private/registry-certs")
        assert entry["cert_mode"] == 0o600 and entry["directory_mode"] == 0o700
        assert not any(argument.endswith("tls-verify=false") for argument in entry["args"])
    assert (release / "output/encrypted-image-reference.txt").is_file()
    local = [
        entry
        for entry in operations
        if entry["tool"] == "skopeo" and entry["args"][0] == "copy" and "cert_dir" not in entry
    ]
    assert len(local) == 1


@pytest.mark.parametrize(
    "options,message",
    [
        ({"cached": False, "pull_status": 17}, "Could not fetch the configured key-provider image"),
        ({"cached": False, "post_pull_failure": True}, "unavailable after pulling"),
        ({"provider_image": "registry.invalid/keyprovider:latest"}, "must be pinned by a SHA256 digest"),
    ],
)
def test_pinned_tool_failures_stop_before_encryption_or_publication(publishing, options, message):
    publish, verify, operations, release = publishing(**options)
    assert publish.returncode != 0
    assert message in publish.stderr
    assert verify is None
    assert not any(entry["tool"] == "skopeo" for entry in operations)
    assert not any(entry["tool"] == "docker" and entry["args"][0] == "run" for entry in operations)
    assert not (release / "output/encrypted-image-reference.txt").exists()


def test_loading_release_config_does_not_create_the_release_directory(credential_kit, credential_bash):
    result = run_release(credential_bash, credential_kit)
    assert result.returncode == 0, result.stderr
    _, work_root = credential_kit
    assert not (work_root / "releases/fixture-v1").exists()
