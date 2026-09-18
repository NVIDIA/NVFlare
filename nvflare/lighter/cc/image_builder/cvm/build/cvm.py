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

"""Build and finalize generic CVM images on a construction host."""

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

from ..artifacts.bundle import approve_bundle, verify_bundle
from ..artifacts.packaging import package_bundle
from ..common.errors import BuildError, require
from ..common.evidence import serial_evidence, verify_reference
from ..common.firewall import firewall_rules
from ..common.io import digest_file, read_json, write_json
from ..common.linux import lock, run
from ..common.policy import compose
from ..host.launcher import cbit_position, qemu_command, vfio_gpus
from ..host.platforms import host_capabilities, select_platform
from . import config
from .payload import GUEST_MODULES, SOURCE_DIRECTORIES, copy_modules
from .storage import build_verity, linux_root, sidecar

SOURCE = Path(__file__).resolve().parents[2]


def contract(profile, source=SOURCE):
    keys = (
        "guest_release",
        "gpu",
        "gpu_count",
        "kernel_version",
        "python_version",
        "docker_version",
        "containerd_version",
        "cryptsetup_version",
        "root_overlay_max_mib",
        "required_system_packages",
        "bootstrap_egress",
        "vault_header_bytes",
        "vault_storage_profile",
        "trustee_commit",
        "attestation_policy_id",
        "kbs_url",
        "token_algorithm",
        "token_issuer",
    )
    value = {key: profile[key] for key in keys}
    for key in ("base_image", "build_firmware", "kbs_cert", "as_public_key", "attestation_policy", "reference_values"):
        value[key + "_sha256"] = digest_file(profile[key])
    value["runtime_source_sha256"] = hashlib.sha256(
        b"".join(
            str(p.relative_to(source)).encode() + b"\0" + p.read_bytes()
            for directory in SOURCE_DIRECTORIES
            for p in sorted((source / directory).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        )
    ).hexdigest()
    value["layout_version"] = 2
    value["dev_mode"] = profile.get("dev_mode", False)
    if profile["gpu"] == "nvidia_cc":
        for key in ("gpu_policy", "gpu_attestation_library"):
            value[key + "_sha256"] = digest_file(profile[key])
        value["gpu_packages"] = profile["gpu_packages"]
        value["gpu_attestation_url"] = profile["gpu_attestation_url"]
    return value


@contextlib.contextmanager
def owned_process(command, log):
    # Boot logs may contain hardware evidence; never create them world-readable.
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "wb") as output:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def kernel_command_line(roothash, offset, root_overlay_max_mib, gpu="none"):
    """Return the exact measured command line used by every vault on a bundle."""
    gpu_pci = " pci=realloc,nocrs" if gpu == "nvidia_cc" else ""
    return (
        "root=/dev/mapper/verity_root rootfstype=ext4 ro console=ttyS0 "
        "panic=1 oops=panic systemd.verity=no "
        f"cvm.root_overlay_max_mib={root_overlay_max_mib} "
        f"roothash={roothash} verity_hash_offset={offset}{gpu_pci}"
    )


def provisioning_payload(profile, platform, build_id, job, source, runtime):
    """Create the application-free payload consumed by the construction guest."""
    payload = job / "provision-payload"
    (payload / "inputs").mkdir(parents=True, mode=0o755)
    copy_modules(source, payload / "source", GUEST_MODULES)
    for name in ("services", "initramfs"):
        shutil.copytree(source / name, payload / "source" / name)
    shutil.copyfile(source / "cvm/build/provisioning.py", payload / "provision_guest.py")
    inputs = {
        "kbs-client": profile["platforms"][platform]["kbs_client"],
        "kbs-ca.pem": profile["kbs_cert"],
        "as-public.pem": profile["as_public_key"],
        "runtime.json": runtime,
    }
    if profile["gpu"] == "nvidia_cc":
        inputs["libnvat.so.1.2.2"] = profile["gpu_attestation_library"]
    for name, path in inputs.items():
        shutil.copyfile(path, payload / "inputs" / name)
    (payload / "inputs/nftables.conf").write_text("flush ruleset\n" + firewall_rules([], profile["bootstrap_egress"]))
    packages = [*profile["required_system_packages"], *profile.get("gpu_packages", [])]
    write_json(
        payload / "config.json",
        {
            "build_id": build_id,
            "build_user": profile["build_user"],
            "dev_mode": dev_mode(profile),
            "gpu": profile["gpu"],
            "guest_release": profile["guest_release"],
            "kernel_version": profile["kernel_version"],
            "platform": platform,
            "profile_version": profile["profile_version"],
            "required_system_packages": packages,
        },
        mode=0o644,
    )
    archive = job / "provision-payload.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(payload, arcname=".")
    return archive


def dev_mode(profile):
    return bool(profile.get("dev_mode", False))


def ssh_process(argv, *, stdin=None, stdout=None, stderr=None, timeout=3600):
    try:
        return subprocess.run(argv, stdin=stdin, stdout=stdout, stderr=stderr, timeout=timeout).returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BuildError(f"Construction guest command could not complete ({type(exc).__name__})") from None


def plain_build(profile, platform, build_id, job, *, dev=False, source=SOURCE):
    image = job / "construction.qcow2"
    # Full copy: no writable backing image or shared overlay to clean up.
    run(["qemu-img", "convert", "-f", "qcow2", "-O", "qcow2", profile["base_image"], image])
    run(["qemu-img", "resize", image, f'{profile["root_drive_size"] + 8}G'])
    key = job / "build-ssh"
    run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", key])
    user_data = {
        "users": [
            {
                "name": profile["build_user"],
                "sudo": "ALL=(ALL) NOPASSWD:ALL",
                "shell": "/bin/bash",
                "ssh_authorized_keys": [(job / "build-ssh.pub").read_text().strip()],
            }
        ],
        "ssh_pwauth": False,
        "disable_root": True,
    }
    import yaml

    (job / "user-data").write_text("#cloud-config\n" + yaml.safe_dump(user_data))
    (job / "meta-data").write_text("instance-id: " + build_id + "\nlocal-hostname: cvm-build\n")
    run(["cloud-localds", job / "seed.img", job / "user-data", job / "meta-data"])
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = [
        "qemu-system-x86_64",
        "-enable-kvm",
        "-machine",
        "q35",
        "-cpu",
        "host",
        "-smp",
        "4",
        "-m",
        "8G",
        "-nographic",
        "-no-reboot",
        "-bios",
        profile["build_firmware"],
        "-drive",
        f"file={image},format=qcow2,if=virtio",
        "-drive",
        f'file={job / "seed.img"},format=raw,if=virtio,readonly=on',
        "-netdev",
        f"user,id=buildnet,hostfwd=tcp:127.0.0.1:{port}-:22",
        "-device",
        "virtio-net-pci,netdev=buildnet",
    ]
    ssh_options = [
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=3",
        "-o",
        "BatchMode=yes",
        "-i",
        str(key),
        "-p",
        str(port),
    ]
    with contextlib.ExitStack() as cleanup, owned_process(command, job / "construction.log") as process:
        cleanup.callback(key.unlink, missing_ok=True)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            require(process.poll() is None, "Construction VM exited; inspect construction.log")
            result = subprocess.run(
                ["ssh", *ssh_options, profile["build_user"] + "@127.0.0.1", "true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            if result.returncode == 0:
                break
            time.sleep(2)
        else:
            raise BuildError("Construction VM did not become reachable")
        runtime = {
            key: profile[key]
            for key in (
                "profile_version",
                "gpu",
                "gpu_count",
                "bootstrap_egress",
                "kbs_url",
                "token_algorithm",
                "token_issuer",
                "attestation_policy_id",
            )
        }
        runtime.update(
            build_id=build_id,
            platform=platform,
            kbs_client="/usr/lib/cvm/bin/kbs-client",
            kbs_cert="/etc/cvm/kbs-ca.pem",
            as_public_key="/etc/cvm/as-public.pem",
        )
        if dev:
            runtime["platform"] = "none"
        if profile["gpu"] == "nvidia_cc":
            runtime.update(gpu_attestation_url=profile["gpu_attestation_url"])
        write_json(job / "runtime.json", runtime)
        archive = provisioning_payload(profile, platform, build_id, job, source, job / "runtime.json")
        destination = profile["build_user"] + "@127.0.0.1"
        # No application secrets are ever supplied to Stage 1. Its log is useful
        # for dependency errors and cannot contain a vault key or application env.
        with (job / "provision.log").open("wb") as output, archive.open("rb") as content:
            status = ssh_process(
                [
                    "ssh",
                    *ssh_options,
                    destination,
                    "install -d -m 0700 /tmp/cvm-provision && tar -xzf - -C /tmp/cvm-provision",
                ],
                stdin=content,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
            require(status == 0, "Could not transfer the construction payload; inspect provision.log")
            status = ssh_process(
                [
                    "ssh",
                    *ssh_options,
                    destination,
                    "sudo /usr/bin/python3 /tmp/cvm-provision/provision_guest.py install "
                    "/tmp/cvm-provision/config.json",
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            require(status == 0, "Generic provisioning failed; inspect provision.log")
            for name in ("vmlinuz", "initrd.img", "artifacts.json"):
                mode = "wb"
                with (job / name).open(mode) as artifact:
                    status = ssh_process(
                        ["ssh", *ssh_options, destination, f"cat /tmp/cvm-provision/out/{name}"],
                        stdout=artifact,
                        stderr=output,
                        timeout=120,
                    )
                require(status == 0, "Could not retrieve boot artifacts; inspect provision.log")
            expected = read_json(job / "artifacts.json")
            require(
                set(expected) == {"vmlinuz", "initrd.img"}
                and all(digest_file(job / name) == expected[name] for name in expected),
                "Construction boot artifact verification failed",
            )
            status = ssh_process(
                [
                    "ssh",
                    *ssh_options,
                    destination,
                    "sudo /usr/bin/python3 /tmp/cvm-provision/provision_guest.py finalize "
                    "/tmp/cvm-provision/config.json",
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
            require(status == 0, "Construction hardening failed; inspect provision.log")
        # The provisioner removed build access and scheduled a clean shutdown.
        try:
            process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            raise BuildError("Construction VM did not shut down cleanly") from None
    return image


@contextlib.contextmanager
def guest_root(image):
    # libguestfs inspects LVM inside its appliance, never activates the guest VG
    # in the host namespace and never touches another build's volume group.
    # The final fstab describes runtime sidecars, which are intentionally absent
    # here. Inspect only the OS identity, then mount that one root explicitly.
    roots = run(["guestfish", "--ro", "-a", image, "run", ":", "inspect-os"], timeout=90).decode().splitlines()
    require(len(roots) == 1 and roots[0].startswith("/dev/"), "Expected exactly one guest operating system")
    with tempfile.TemporaryDirectory(prefix="cvm-root-", dir="/run") as directory:
        # Keep the FUSE process in the foreground and retain its exact PID.
        # Daemonizing libguestfs can lose its appliance on parent exit.
        with owned_process(
            ["guestmount", "--no-fork", "--ro", "-a", image, "-m", roots[0] + ":/", directory],
            str(image) + ".guestmount.log",
        ) as process:
            try:
                deadline = time.monotonic() + 120
                while not os.path.ismount(directory):
                    require(process.poll() is None and time.monotonic() < deadline, "Guest root mount failed")
                    time.sleep(0.2)
                yield Path(directory)
            finally:
                if os.path.ismount(directory):
                    run(["umount", directory], timeout=30)
                process.wait(timeout=30)


def collect_reference(manifest, directory, *, gpu=None, timeout=300):
    platform = manifest["platform"]
    require(platform in host_capabilities(), "Collect reference measurements on the target TEE host")
    with tempfile.TemporaryDirectory(prefix="cvm-reference-", dir=directory.parent) as temporary:
        job = Path(temporary)
        disks = [directory / "verity_root.qcow2"]
        for name in ("applog", "user_config", "user_data"):
            sidecar(job / (name + ".qcow2"), 1)
            disks.append(job / (name + ".qcow2"))
        log = directory / "reference-boot.log"
        gpu_context = (
            vfio_gpus(gpu, manifest["contract"]["gpu_count"])
            if manifest["contract"]["gpu"] == "nvidia_cc"
            else contextlib.nullcontext(None)
        )
        require(manifest["contract"]["gpu"] == "nvidia_cc" or not gpu, "CPU-only profile cannot add a GPU")
        with gpu_context as selected_gpus:
            command = qemu_command(
                manifest,
                directory,
                disks,
                bytes(32),
                reference=True,
                gpus=selected_gpus,
                cbit=cbit_position() if platform == "amd_sev_snp" else None,
            )
            with owned_process(command, log) as process:
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    text = log.read_text(errors="replace")
                    evidence = serial_evidence(text)
                    if evidence is not None:
                        verify_reference(platform, evidence)
                        write_json(directory / "reference-evidence.json", evidence)
                        return evidence["measurements"]
                    for line in text.splitlines():
                        if "CVM_REFERENCE=" in line:
                            try:
                                evidence = json.loads(line.split("CVM_REFERENCE=", 1)[1])
                            except json.JSONDecodeError:
                                continue
                            verify_reference(platform, evidence)
                            write_json(directory / "reference-evidence.json", evidence)
                            return evidence["measurements"]
                    require(process.poll() is None, "Reference VM exited; inspect reference-boot.log")
                    time.sleep(1)
        raise BuildError("Timed out collecting reference evidence; inspect reference-boot.log")


def finalize(directory, evidence=None, gpu=None, *, package=True):
    directory = Path(directory).resolve()
    with lock(directory / ".finalize.lock"):
        require(not (directory / "cvm_manifest.json").exists(), "Bundle already finalized")
        result = finalize_locked(directory, evidence, gpu)
        if package:
            package_bundle(result)
        return result


def finalize_locked(directory, evidence=None, gpu=None):
    manifest = read_json(directory / "cvm_manifest.pending.json")
    # Validate artifacts before trusting any report collected from them.
    for name, expected in manifest["sha256"].items():
        require(digest_file(directory / name) == expected, "Candidate bundle changed before measurement")
    if manifest.get("dev_mode"):
        manifest["measurements"] = {}
    elif evidence is None:
        manifest["measurements"] = collect_reference(manifest, directory, gpu=gpu)
    else:
        evidence = read_json(evidence)
        verify_reference(manifest["platform"], evidence)
        manifest["measurements"] = evidence["measurements"]
        write_json(directory / "reference-evidence.json", evidence)
    # Hardware references are bundle artifacts. Production approval separately
    # verifies a signed quote, CCEL replay, policy and failure-path acceptance.
    references = read_json(directory / "reference_values.json")
    key_map = {
        "snp.measurement": "snp_launch_measurement",
        "mr_td": "mr_td",
        "rtmr_0": "rtmr_0",
        "rtmr_1": "rtmr_1",
        "rtmr_2": "rtmr_2",
    }
    for key, value in manifest["measurements"].items():
        references[key_map[key]] = [value]
    write_json(directory / "reference_values.json", references, mode=0o644)
    manifest["sha256"]["reference_values.json"] = digest_file(directory / "reference_values.json")
    (directory / "resource_policy.rego").write_text(compose([] if manifest.get("dev_mode") else [manifest]))
    write_json(directory / "cvm_manifest.json", manifest, mode=0o644)
    verify_bundle(directory)
    profiles_path = directory.parent / "profile_set.json"
    with lock(directory.parent / ".profile.lock"):
        if profiles_path.exists():
            profiles = read_json(profiles_path)
            require(
                profiles["profile_version"] == manifest["profile_version"],
                "Profile version differs from existing bundles",
            )
            require(profiles["contract"] == manifest["contract"], "Profile contract differs from existing bundles")
        else:
            profiles = {
                "schema_version": 2,
                "profile_version": manifest["profile_version"],
                "contract": manifest["contract"],
                "bundles": {},
            }
        require(manifest["platform"] not in profiles["bundles"], "Platform already finalized in this profile")
        profiles["bundles"][manifest["platform"]] = {
            "build_id": manifest["build_id"],
            "manifest_sha256": digest_file(directory / "cvm_manifest.json"),
        }
        write_json(profiles_path, profiles, mode=0o644)
    return directory


def resolve_acceptance_runner(runner):
    runner = str(runner)
    resolved = shutil.which(runner) if "/" not in runner else str(Path(runner).resolve())
    require(resolved and os.access(resolved, os.X_OK), f"Acceptance runner is not executable: {runner}")
    return resolved


def select_acceptance_runner(profile, explicit=None, *, defer_measurements=False, dev=False):
    if defer_measurements or dev:
        require(not explicit, "Acceptance requires local production finalization")
        return None
    runner = explicit or profile.get("acceptance_runner")
    return resolve_acceptance_runner(runner or "site_acceptance")


def run_acceptance(runner, directory):
    """Run the trusted site acceptance adapter and approve its exact report."""
    resolved = resolve_acceptance_runner(runner)
    with tempfile.TemporaryDirectory(prefix="cvm-acceptance-", dir=directory.parent) as temporary:
        report = Path(temporary) / "acceptance-report.json"
        run([resolved, str(directory), str(report)], timeout=None)
        require(report.is_file(), "Acceptance runner did not create its requested report")
        approve_bundle(directory, read_json(report))


def build(path, explicit=None, output=None, *, defer_measurements=False, gpu=None, dev=False, acceptance_runner=None):
    linux_root()
    profile = config.profile(path)
    acceptance_runner = select_acceptance_runner(
        profile,
        acceptance_runner,
        defer_measurements=defer_measurements,
        dev=dev,
    )
    require(
        profile["profile_version"].startswith("dev-") == dev,
        "Development builds require --dev and a separate dev- profile version",
    )
    profile["dev_mode"] = dev
    platform = select_platform(profile, explicit)
    root = Path(output or Path("target") / ("cvm_" + profile["profile_version"])).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Provision from the exact source snapshot whose hash enters the contract.
    # Editing the working tree during a long image build cannot mix versions.
    job = Path(tempfile.mkdtemp(prefix="construction-", dir=root))
    source = job / "source"
    for name in SOURCE_DIRECTORIES:
        shutil.copytree(SOURCE / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shared = contract(profile, source)
    directory = root / platform
    with lock(root / ".profile.lock"):
        if (root / "profile_set.json").exists():
            profiles = read_json(root / "profile_set.json")
            require(
                profiles["profile_version"] == profile["profile_version"],
                "Profile version differs from existing bundles",
            )
            require(
                profiles["contract"] == shared,
                "Profile changes require a new profile_version",
            )
        require(not directory.exists(), "Bundle already exists; reuse it or choose a new profile version")
        directory.mkdir(mode=0o755)
    build_id = "cvm-" + uuid.uuid4().hex
    settings = profile["platforms"][platform]
    # Keep failed generic build logs for diagnosis; they never contain vault data.
    try:
        image = plain_build(profile, platform, build_id, job, dev=dev, source=source)
        for name in ("vmlinuz", "initrd.img"):
            shutil.copyfile(job / name, directory / name)
        with guest_root(image) as guest_files:
            roothash, offset = build_verity(guest_files, directory / "verity_root.qcow2", profile["root_drive_size"])
        shutil.copyfile(profile["build_firmware"] if dev else settings["firmware"], directory / "OVMF.fd")
        shutil.copyfile(profile["attestation_policy"], directory / "attestation_policy.rego")
        shutil.copyfile(profile["reference_values"], directory / "reference_values.json")
        for name in ("launch_cvm.sh.tmpl", "shutdown_cvm.sh.tmpl"):
            shutil.copyfile(source / "templates" / name, directory / name)
        # Our initramfs already opened the verified root. Disable systemd's
        # independent GPT-based roothash discovery to avoid a second mapping.
        cmdline = kernel_command_line(roothash, offset, profile["root_overlay_max_mib"], profile["gpu"])
        shape = {"vcpus": profile["vcpus"], "memory_gib": profile["memory_gib"], "cpu_model": settings["cpu_model"]}
        if platform == "intel_tdx":
            shape["quote_generation"] = settings["quote_generation"]
        artifacts = [
            "verity_root.qcow2",
            "OVMF.fd",
            "vmlinuz",
            "initrd.img",
            "attestation_policy.rego",
            "reference_values.json",
            "launch_cvm.sh.tmpl",
            "shutdown_cvm.sh.tmpl",
        ]
        if profile["gpu"] == "nvidia_cc":
            from ..common.gpu_policy import render

            (directory / "gpu_attestation_policy.rego").write_text(render(read_json(profile["gpu_policy"])))
            artifacts.append("gpu_attestation_policy.rego")
        if settings.get("shim") and not dev:
            shutil.copyfile(settings["shim"], directory / "shim.efi")
            artifacts.append("shim.efi")
            shape["shim"] = True
        manifest = {
            "schema_version": 2,
            "build_id": build_id,
            "profile_version": profile["profile_version"],
            "platform": platform,
            "dev_mode": dev,
            "contract": shared,
            "launch_shape": shape,
            "cmdline": cmdline,
            "cmdline_sha256": hashlib.sha256(cmdline.encode()).hexdigest(),
            "root_hash": roothash,
            "hash_offset": offset,
            "attestation_policy_id": profile["attestation_policy_id"],
            "kbs_client_sha256": digest_file(settings["kbs_client"]),
            "sha256": {name: digest_file(directory / name) for name in artifacts},
        }
        write_json(directory / "cvm_manifest.pending.json", manifest, mode=0o644)
        if not defer_measurements:
            finalize(directory, gpu=gpu, package=False)
            if acceptance_runner:
                run_acceptance(acceptance_runner, directory)
        package_bundle(directory)
        shutil.rmtree(job)
        return directory
    except Exception:
        print(f"Generic build retained for diagnosis: {job}", flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(SOURCE / "config/cvm_profile.yml"))
    parser.add_argument("-p", "--platform", choices=("amd_sev_snp", "intel_tdx"))
    parser.add_argument("--output")
    parser.add_argument(
        "--defer-measurements", action="store_true", help="Construct now; collect references on the target host later"
    )
    parser.add_argument("--dev", action="store_true", help="Separate dev- profile with no TEE or KBS authorization")
    parser.add_argument("--finalize", metavar="BUNDLE", help="Finalize a previously constructed bundle")
    parser.add_argument(
        "--reference-evidence", help="Private reference report captured from this bundle on a trusted target host"
    )
    parser.add_argument("--gpu", action="append", help="Repeat once per explicit NVIDIA GPU PCI address")
    parser.add_argument(
        "--acceptance-runner",
        help="Trusted executable called as RUNNER BUNDLE REPORT after finalization; a valid report approves the bundle",
    )
    args = parser.parse_args()
    try:
        result = (
            finalize(args.finalize, args.reference_evidence, args.gpu)
            if args.finalize
            else build(
                args.config,
                args.platform,
                args.output,
                defer_measurements=args.defer_measurements,
                gpu=args.gpu,
                dev=args.dev,
                acceptance_runner=args.acceptance_runner,
            )
        )
        print(f"Generic bundle: {result}")
        manifest_name = (
            "cvm_manifest.json" if (Path(result) / "cvm_manifest.json").is_file() else "cvm_manifest.pending.json"
        )
        manifest = read_json(Path(result) / manifest_name)
        print(
            "OCI artifact: "
            + str(Path(result).parent / f"cvm_{manifest['profile_version']}_{manifest['platform']}.oci.tar")
        )
        if (Path(result) / "approval.json").is_file():
            print("The exact finalized bundle passed the acceptance runner and is approved.")
        elif not args.defer_measurements and not args.dev:
            print("Production approval remains pending; use scripts/admin_approve after acceptance testing.")
    except (BuildError, OSError, ValueError, KeyError, tarfile.TarError) as exc:
        parser.exit(1, f"CVM build failed: {exc}\n")


if __name__ == "__main__":
    main()
