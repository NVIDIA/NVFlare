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

"""Host integration fail-closed tests; not a substitute for real quote verification."""

import base64
import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[5]
WORKFLOW = ROOT / "examples/devops/coco/trusted_system/tdx-reference.py"
spec = importlib.util.spec_from_file_location("tdx_reference", WORKFLOW)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
collector_spec = importlib.util.spec_from_file_location(
    "tdx_collector", WORKFLOW.parent / "rehearsal-collector/collect-tdx-evidence.py"
)
collector = importlib.util.module_from_spec(collector_spec)
collector_spec.loader.exec_module(collector)


def verified_claims():
    return {
        "quote": {
            "body": {
                "mr_td": "a" * 96,
                "rtmr_0": "1" * 96,
                "rtmr_1": "b" * 96,
                "rtmr_2": "c" * 96,
                "rtmr_3": "2" * 96,
                "xfam": "d" * 16,
            }
        },
        "uefi_event_logs": [
            {
                "type_name": "EV_EFI_BOOT_SERVICES_APPLICATION",
                "details": {"device_paths": ["File(kernel)"]},
                "digests": [{"alg": "SHA-384", "digest": "e" * 96}],
            },
            {
                "type_name": "EV_EVENT_TAG",
                "details": {"string": "LOADED_IMAGE::LoadOptions"},
                "digests": [{"alg": "SHA-384", "digest": "f" * 96}],
            },
        ],
    }


def test_complete_verified_profile():
    result = workflow.extract_profile(verified_claims(), "tdx-profile-1")
    assert result == {
        "id": "tdx-profile-1",
        "mr_td": "a" * 96,
        "rtmr_0": "1" * 96,
        "rtmr_1": "b" * 96,
        "rtmr_2": "c" * 96,
        "rtmr_3": "2" * 96,
        "xfam": "d" * 16,
        "tdvfkernel": "e" * 96,
        "tdvfkernelparams": "f" * 96,
    }


@pytest.mark.parametrize("field", ["mr_td", "rtmr_0", "rtmr_1", "rtmr_2", "rtmr_3", "xfam"])
@pytest.mark.parametrize("value", [None, 0, "", "A" * 96, "g" * 96, "0" * 64])
def test_invalid_verified_field_rejected(field, value):
    claims = verified_claims()
    claims["quote"]["body"][field] = value
    with pytest.raises(ValueError, match="Invalid verified"):
        workflow.extract_profile(claims, "tdx")


@pytest.mark.parametrize("field", ["mr_td", "rtmr_0", "rtmr_1", "rtmr_2", "rtmr_3", "xfam"])
def test_missing_verified_field_rejected(field):
    claims = verified_claims()
    del claims["quote"]["body"][field]
    with pytest.raises(ValueError, match="Invalid verified"):
        workflow.extract_profile(claims, "tdx")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "sha256", "bad_digest", "tdshim"])
@pytest.mark.parametrize("event_index", [0, 1])
def test_incomplete_or_ambiguous_event_log_rejected(mutation, event_index):
    claims = verified_claims()
    event = claims["uefi_event_logs"][event_index]
    if mutation == "missing":
        claims["uefi_event_logs"].pop(event_index)
    elif mutation == "duplicate":
        claims["uefi_event_logs"].append(copy.deepcopy(event))
    elif mutation == "sha256":
        event["digests"][0]["alg"] = "SHA-256"
    elif mutation == "bad_digest":
        event["digests"][0]["digest"] = "x" * 96
    else:
        event["details"] = {"string": "td_payload"}
    with pytest.raises(ValueError, match="exactly one verified TDVF"):
        workflow.extract_profile(claims, "tdx")


def test_frame_decode_requires_raw_quote_and_ccel():
    evidence = {"quote": "cXVvdGU=", "cc_eventlog": "Y2NlbA=="}
    frame = workflow.FRAME + base64.b64encode(json.dumps(evidence).encode()).decode()
    assert workflow.decode_evidence(frame) == evidence
    with pytest.raises(ValueError):
        workflow.decode_evidence(frame + "\n" + frame)
    with pytest.raises(ValueError):
        workflow.decode_evidence(workflow.FRAME + "notbase64")
    for missing in ("quote", "cc_eventlog"):
        bad = dict(evidence)
        bad.pop(missing)
        with pytest.raises(ValueError):
            workflow.decode_evidence(workflow.FRAME + base64.b64encode(json.dumps(bad).encode()).decode())


def test_verification_failure_cannot_export_decoded_measurement(tmp_path):
    (tmp_path / "request-data.bin").write_bytes(b"a" * 64)
    with (
        patch.object(workflow, "verifier", return_value=Path("pinned-verifier")),
        patch.object(workflow, "output", side_effect=subprocess.CalledProcessError(1, ["pinned-verifier"])),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            workflow.verify_run(tmp_path, tmp_path, "tdx")


def test_verify_uses_raw_report_nonce_and_exact_initdata(tmp_path):
    (tmp_path / "request-data.bin").write_bytes(b"b" * 64)
    with (
        patch.object(workflow, "verifier", return_value=Path("pinned-verifier")),
        patch.object(workflow, "output", return_value=json.dumps(verified_claims())) as verify,
    ):
        profile, _ = workflow.verify_run(tmp_path, tmp_path, "tdx")
    assert profile["mr_td"] == "a" * 96
    assert verify.call_args.args[0] == [
        Path("pinned-verifier"),
        tmp_path / "evidence.json",
        tmp_path / "request-data.bin",
        tmp_path / "initdata.toml",
    ]


@pytest.mark.parametrize("baseline", [None, "", "0", "true", "yes"])
def test_baseline_must_be_explicit(tmp_path, baseline):
    approval = tmp_path / "platform-approval.env"
    approval.touch()
    with patch.object(workflow, "read_env", return_value={"TDX_SECURITY_BASELINE_APPROVED": baseline}):
        with pytest.raises(ValueError, match="baseline"):
            workflow.approval(tmp_path, approval)


def test_finalize_requires_specific_authority_approval(tmp_path):
    candidate = tmp_path / "candidate-tdx-profile.json"
    candidate.write_text(json.dumps({"id": "not-yet-approved"}))
    with (
        patch.object(workflow, "config", return_value=({}, tmp_path)),
        patch.object(workflow, "approval", return_value={"TDX_SECURITY_BASELINE_APPROVED": "1"}),
        patch.object(workflow, "verify_run") as verify,
    ):
        with pytest.raises(ValueError, match="explicitly approve"):
            workflow.finalize("base.env", "approval.env")
    verify.assert_not_called()
    assert not (tmp_path / "platform-reference.final.env").exists()


@pytest.mark.parametrize("field", ["rtmr_0", "rtmr_3"])
@pytest.mark.parametrize("changed_run", ["first", "repeat", "both"])
def test_finalize_rejects_register_change_in_reverified_runs(tmp_path, field, changed_run):
    claims = verified_claims()
    candidate = tmp_path / "candidate-tdx-profile.json"
    candidate.write_text(json.dumps(workflow.extract_profile(claims, "tdx-test")))
    for directory, nonce in (("rehearsal-collector-build", b"a" * 64), ("repeat-rehearsal", b"b" * 64)):
        run_dir = tmp_path / directory
        run_dir.mkdir()
        (run_dir / "request-data.bin").write_bytes(nonce)
    runs = [copy.deepcopy(claims), copy.deepcopy(claims)]
    for index, name in enumerate(("first", "repeat")):
        if changed_run in (name, "both"):
            runs[index]["quote"]["body"][field] = "3" * 96
    with (
        patch.object(workflow, "config", return_value=({"PLATFORM_PROFILE": "tdx-test"}, tmp_path)),
        patch.object(workflow, "approval", return_value={"APPROVED_TDX_PROFILE_SHA256": workflow.sha(candidate)}),
        patch.object(workflow, "verifier", return_value=Path("pinned-verifier")),
        patch.object(workflow, "output", side_effect=[json.dumps(run) for run in runs]) as verify,
    ):
        with pytest.raises(ValueError, match="both newly verified runs"):
            workflow.finalize("base.env", "approval.env")
    assert verify.call_count == 2
    assert not (tmp_path / "approved-tdx-reference-values.json").exists()
    assert not (tmp_path / "platform-reference.final.env").exists()


@pytest.mark.parametrize("omitted", [(), ("rtmr_0",), ("rtmr_3",), ("rtmr_0", "rtmr_3")])
def test_export_requires_complete_finalized_profile(tmp_path, omitted):
    reference = workflow.extract_profile(verified_claims(), "tdx-test")
    for field in omitted:
        del reference[field]
    values = {"schema": "coco-platform-reference-values/v2", "tee": "tdx", "profiles": [reference]}
    source = tmp_path / "approved-tdx-reference-values.json"
    source.write_text(json.dumps(values))
    destination = tmp_path / "handoff.json"
    env = {"TDX_REFERENCE_VALUES_SHA256": workflow.sha(source)}
    with patch.object(workflow, "config", return_value=(env, tmp_path)):
        if omitted:
            with pytest.raises(ValueError, match="all eight measurement fields"):
                workflow.export(tmp_path / "platform-reference.final.env", destination)
            assert not destination.exists()
        else:
            workflow.export(tmp_path / "platform-reference.final.env", destination)
            assert json.loads(destination.read_text()) == values


def test_write_cannot_overwrite_evidence(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text("original")
    with pytest.raises(FileExistsError):
        workflow.write(path, "changed")
    assert path.read_text() == "original"


@pytest.mark.parametrize("nonce", [b"", b"x" * 64, b"a" * 63, b"A" * 64, b"a" * 65, bytes(range(64))])
def test_collector_rejects_invalid_nonce_without_network(tmp_path, nonce):
    path = tmp_path / "nonce"
    path.write_bytes(nonce)
    with patch.object(collector.urllib.request, "build_opener") as opener:
        with pytest.raises(ValueError, match="challenge"):
            collector.collect(path)
        opener.assert_not_called()


def test_collector_redirects_forbidden():
    with pytest.raises(ValueError, match="redirect"):
        collector.NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere")


def test_approval_outside_selected_profile_rejected(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    elsewhere = tmp_path / "elsewhere.env"
    elsewhere.touch()
    with pytest.raises(ValueError, match="selected profile"):
        workflow.approval(profile, elsewhere)


def test_read_launch_rejects_mutated_records_even_if_both_copies_match(tmp_path):
    launch = {"launch_inputs": {"kernel_command_line": "approved", "smp": "4"}}
    launch["launch_inputs_sha256"] = hashlib.sha256(
        json.dumps(launch["launch_inputs"], sort_keys=True).encode()
    ).hexdigest()
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(launch))
    assert workflow.read_launch(path) == launch
    launch["launch_inputs"]["kernel_command_line"] = "changed-in-both-first-and-repeat"
    for name in ("first.json", "repeat.json"):
        path = tmp_path / name
        path.write_text(json.dumps(launch))
        with pytest.raises(ValueError, match="stale or inconsistent"):
            workflow.read_launch(path)


def test_approval_does_not_inherit_previous_shell_approval(tmp_path, monkeypatch):
    path = tmp_path / "approval.env"
    path.write_text("# no approval yet\n")
    monkeypatch.setenv("TDX_SECURITY_BASELINE_APPROVED", "1")
    monkeypatch.setenv("APPROVED_TDX_PROFILE_SHA256", "a" * 64)
    assert workflow.read_env(path) == {}


def test_collector_guest_digest_matches_containerd_cache_alias(tmp_path):
    tls = tmp_path / "registry-tls"
    tls.mkdir()
    (tls / "ca.crt").write_text("public rehearsal CA\n")
    source = tmp_path / "source.yaml"
    source.write_text(
        "spec:\n  runtimeClassName: kata-qemu-tdx\n  containers:\n  - name: app\n    image: example/app\n"
    )
    digest = "sha256:" + "a" * 64
    tagged = "192.0.2.10:5443/coco-tdx-rehearsal:trusted-run"
    expected = "192.0.2.10:5443/coco-tdx-rehearsal@" + digest
    with patch.object(workflow, "run") as fake_runner:
        pinned = workflow.pin_collector_cache(tagged, digest)
    fake_runner.assert_called_once_with(["sudo", "ctr", "--namespace", "k8s.io", "images", "tag", tagged, expected])
    pod = workflow.make_pod(
        {"RUNTIME_CLASS": "kata-qemu-tdx", "REHEARSAL_WORKLOAD_YAML": str(source)},
        tmp_path,
        "trusted-namespace",
        "192.0.2.10:5443",
        pinned,
    )
    assert pod["spec"]["containers"][0]["image"] == expected
    assert pod["spec"]["containers"][0]["imagePullPolicy"] == "Never"


@pytest.mark.parametrize("digest", ["latest", "", "sha256:" + "g" * 64, "sha256:" + "A" * 64])
def test_collector_cache_rejects_nonimmutable_digest(digest):
    with patch.object(workflow, "run") as fake_runner:
        with pytest.raises(ValueError, match="immutable"):
            workflow.pin_collector_cache("192.0.2.10:5443/collector:run", digest)
    fake_runner.assert_not_called()


def test_collector_publishing_isolates_private_keys_from_skopeo_and_build_context(tmp_path):
    digest = "sha256:" + "a" * 64
    first = tmp_path / "rehearsal-collector-build"
    config = tmp_path / "base.env"
    config.touch()
    approval = tmp_path / "approval.env"
    approval.touch()
    publishing_commands = []

    def fake_run(args, **kwargs):
        args = [str(arg) for arg in args]
        if args[0] == "openssl":
            for flag in ("-out", "-keyout"):
                if flag in args:
                    Path(args[args.index(flag) + 1]).write_text("test " + flag)
        elif args[:2] == ["docker", "build"]:
            assert Path(args[-1]) == first
            assert (first / ".dockerignore").read_text().splitlines() == [
                "*",
                "!Dockerfile",
                "!collect-tdx-evidence.py",
            ]
            assert (first / "Dockerfile").is_file()
            assert (first / "collect-tdx-evidence.py").is_file()
            assert (first / "registry-tls/ca.key").is_file()
            assert (first / "registry-tls/server.key").is_file()
        elif args[:2] == ["skopeo", "copy"]:
            publishing_commands.append(args)
            Path(args[args.index("--digestfile") + 1]).write_text(digest)

    def fake_output(args, **kwargs):
        args = [str(arg) for arg in args]
        if args[0] == "ip":
            return json.dumps([{"prefsrc": "192.0.2.10"}])
        if args[:2] == ["skopeo", "inspect"]:
            publishing_commands.append(args)
        return digest

    with (
        patch.object(workflow, "config", return_value=({"RUNTIME_CLASS": "kata-qemu-tdx"}, tmp_path)),
        patch.object(workflow, "approval"),
        patch.object(workflow, "verifier"),
        patch.object(workflow, "preflight"),
        patch.object(workflow, "start_registry"),
        patch.object(workflow, "run", side_effect=fake_run),
        patch.object(workflow, "output", side_effect=fake_output),
        patch.object(workflow, "pin_collector_cache", side_effect=RuntimeError("stop after publishing")),
        patch.object(workflow.subprocess, "run"),
    ):
        with pytest.raises(RuntimeError, match="stop after publishing"):
            workflow.collect(config, approval)

    certs = first / "skopeo-certs"
    assert sorted(path.name for path in certs.iterdir()) == ["ca.crt"]
    assert (certs / "ca.crt").read_bytes() == (first / "registry-tls/ca.crt").read_bytes()
    assert len(publishing_commands) == 2
    for command, flag in zip(publishing_commands, ("--dest-cert-dir", "--cert-dir")):
        assert Path(command[command.index(flag) + 1]) == certs
