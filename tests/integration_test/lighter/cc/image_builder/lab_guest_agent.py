#!/usr/bin/python3
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

"""Explicit test-vault payload for hardware acceptance; never installed by Stage 1.

Offers fixed fault-injection actions on the isolated lab guest. No command
execution endpoint and no key/token output. Include only in test- vaults.
"""

import http.server
import json
import os
import signal
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, "/usr/lib/cvm")
from cvm.common.contracts import DISK_ROLES, resource_path, validate_resource
from cvm.common.errors import BuildError, require
from cvm.common.io import read_json
from cvm.common.linux import memory_file, protect_process, run
from cvm.runtime.attestation import validate_token
from cvm.runtime.storage import disk_device

CONFIG = read_json("/etc/cvm/runtime.json")
require(CONFIG["profile_version"].startswith("test-"), "Acceptance payload requires a test profile")
protect_process()


def state():
    from cvm.common.measurements import measurements, parse_snp_report, parse_tdx_report
    from cvm.runtime.platforms import local_report

    evidence, nonce = local_report(CONFIG["platform"])
    parse = parse_tdx_report if CONFIG["platform"] == "intel_tdx" else parse_snp_report
    binding = parse(evidence, nonce)
    result = {
        "measurements": measurements(CONFIG["platform"], evidence),
        "binding": binding.hex(),
        "platform": CONFIG["platform"],
        "marker": Path("/vault/docker/image-loaded.json").exists(),
    }
    result["core_dumps_disabled"] = (
        Path("/proc/sys/kernel/core_pattern").read_text().strip() == "/dev/null"
        and Path("/proc/sys/kernel/core_uses_pid").read_text().strip() == "0"
    )
    capacity = os.statvfs("/vault")
    result["vault_bytes"] = capacity.f_frsize * capacity.f_blocks
    result["vault_available_bytes"] = capacity.f_frsize * capacity.f_bavail
    capacity = os.statvfs("/cow")
    result["root_overlay_bytes"] = capacity.f_frsize * capacity.f_blocks
    result["disk_paths"] = {role: os.path.realpath(disk_device(role)) for role in DISK_ROLES}
    result["encrypted_disk_paths"] = {"vault": os.path.realpath("/dev/mapper/vault")}
    result["sidecar_roles_correct"] = all(
        os.path.realpath(run(["findmnt", "-no", "SOURCE", "--mountpoint", mount]).decode().strip())
        == result["disk_paths"][role]
        for role, mount in (("applog", "/applog"), ("user-config", "/user_config"), ("user-data", "/user_data"))
    )
    result["input_mounts_read_only"] = all(
        "ro" in run(["findmnt", "-no", "OPTIONS", "--mountpoint", mount]).decode().strip().split(",")
        for mount in ("/user_config", "/user_data")
    )
    with open("/applog/acceptance-output", "wb") as stream:
        stream.write(b"CVM_APPLOG_WRITE_OK\n")
        stream.flush()
        os.fsync(stream.fileno())
    result["applog_writable"] = True
    for name in ("cvm_bootstrap.service", "cvm_integrity.service", "cvm_app.service", "docker.service"):
        result[name] = run(["systemctl", "show", name, "--property=ActiveState", "--value"]).decode().strip()
    result["periodic"] = read_json("/run/cvm/periodic.json")
    result["cvm_units"] = [
        line.split()[0]
        for line in run(["systemctl", "list-unit-files", "cvm_*", "--no-legend", "--no-pager"]).decode().splitlines()
    ]
    result["docker_socket"] = (
        run(["systemctl", "show", "docker.socket", "--property=UnitFileState", "--value"]).decode().strip()
    )
    result["nftables_enabled"] = run(["systemctl", "is-enabled", "nftables.service"]).decode().strip()
    result["firewall_present"] = bool(run(["nft", "list", "table", "inet", "cvm"]))
    result["ssh_units"] = {
        name: {
            "active": run(["systemctl", "show", name, "--property=ActiveState", "--value"]).decode().strip(),
            "enabled": run(["systemctl", "show", name, "--property=UnitFileState", "--value"]).decode().strip(),
        }
        for name in ("ssh.service", "ssh.socket")
    }
    listening = False
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in Path(table).read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].split(":")[1], 16) == 22:
                listening = True
    result["ssh_port_listening"] = listening
    result["clock_synchronized"] = (
        subprocess.run(
            ["/usr/bin/chronyc", "waitsync", "1", "0.5", "1000", "1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).returncode
        == 0
    )
    p = Path("/vault/acceptance-state")
    result["persisted"] = p.is_file() and p.read_bytes() == b"CVM_PERSISTENT_WRITE\n"
    return result


def cross_vault(path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    validate_resource(path)
    digest = bytes.fromhex(read_json("/run/cvm/binding.json")["digest"])
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()
    )
    command = [CONFIG["kbs_client"], "--url", CONFIG["kbs_url"], "--cert-file", CONFIG["kbs_cert"]]
    with memory_file(pem) as private:
        token = run(
            command + ["attest", "--tee-key-file", f"/proc/self/fd/{private}"],
            pass_fds=(private,),
            timeout=240 if CONFIG["gpu"] == "nvidia_cc" else 60,
            env=dict(os.environ, RUST_LOG="off"),
        ).strip()
        validate_token(token, CONFIG, digest)
        context = ssl.create_default_context(cafile=CONFIG["kbs_cert"])
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context)
        )

        def get(resource):
            req = urllib.request.Request(
                CONFIG["kbs_url"] + "/kbs/v0/resource/" + resource,
                headers={"Authorization": "Bearer " + token.decode("ascii")},
            )
            return opener.open(req, timeout=30)

        # Establish positive authorization with the same fresh token first.
        own = resource_path(CONFIG["build_id"], CONFIG["platform"], digest)
        with get(own) as response:
            require(response.status == 200, "Current vault authorization failed")
        try:
            with get(path):
                pass
        except urllib.error.HTTPError as error:
            with error:
                detail = json.load(error)
                denied = error.code == 401 and detail.get("type", "").endswith("/PolicyDeny")
            return {"cross_vault_denied": denied, "fresh_positive_appraisal": True}
    return {"cross_vault_denied": False}


def write_loop():
    with open("/vault/acceptance-journal-load", "wb", buffering=0) as stream:
        while True:
            stream.seek(0)
            stream.write(os.urandom(1024 * 1024))
            os.fsync(stream.fileno())


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def respond(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def do_GET(self):
        if self.path != "/state":
            self.send_error(404)
            return
        self.respond(state())

    def do_POST(self):
        self.phase = "dispatch"
        try:
            self.inject()
        except Exception as error:
            self.respond(
                {
                    "fault_error": type(error).__name__,
                    "phase": self.phase,
                    "diagnostic": str(error) if isinstance(error, BuildError) else "Test fixture exception",
                }
            )

    def inject(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/write":
            with open("/vault/acceptance-state", "wb") as stream:
                stream.write(b"CVM_PERSISTENT_WRITE\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.respond({"written": True})
        elif url.path == "/write-loop":
            threading.Thread(target=write_loop, daemon=True).start()
            self.respond({"writing": True})
        elif url.path == "/kill-monitor":
            pid = int(run(["systemctl", "show", "cvm_integrity.service", "--property=MainPID", "--value"]))
            require(pid > 1, "No live integrity monitor")
            self.respond({"injected_monitor_crash": True})
            os.kill(pid, signal.SIGKILL)
        elif url.path == "/crash-child":
            # A newly exec'd process resets PR_SET_DUMPABLE. Use a disposable
            # public sentinel, not any attestation or vault material.
            pid = os.posix_spawn(
                sys.executable,
                [sys.executable, "-c", "import os; marker=b'CVM_PUBLIC_DUMP_TEST'; os.abort()"],
                dict(os.environ),
            )
            _, status = os.waitpid(pid, 0)
            self.respond(
                {
                    "aborted": os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGABRT,
                    "core_dumped": os.WCOREDUMP(status),
                }
            )
        elif url.path == "/periodic":
            self.respond({"requested_periodic_check": True})
            run(["systemctl", "kill", "--kill-whom=main", "--signal=SIGUSR1", "cvm_bootstrap.service"])
        elif url.path == "/scan":
            self.respond({"requested_authenticated_scan": True})
            run(["sync"])
            Path("/proc/sys/vm/drop_caches").write_text("3\n")
            from cvm.common.luks import scan

            scan("/dev/mapper/vault")
        elif url.path == "/prepare-interrupted-load":
            # The test deliberately terminates a running container. Accept its
            # expected termination statuses only during fault preparation. All
            # integrity and attestation failure handlers remain active.
            override = Path("/run/systemd/system/cvm_app.service.d/acceptance.conf")
            self.phase = "prepare-unit-override"
            override.parent.mkdir(exist_ok=True)
            override.write_text("[Service]\nSuccessExitStatus=137 143\n")
            run(["systemctl", "daemon-reload"])
            self.phase = "stop-application"
            run(["systemctl", "stop", "cvm_app.service"], timeout=45)
            # Docker's attached CLI can report exit 137 after the stop job
            # completes. Keep this test override until the simulated power loss;
            # the next boot automatically restores the measured exit policy.
            self.phase = "remove-image"
            image = read_json("/vault/config/application.json")["image_id"]
            run(["docker", "image", "rm", "--force", image])
            self.phase = "reset-marker"
            Path("/vault/docker/image-loaded.json").unlink(missing_ok=True)
            run(["sync", "-f", "/vault"])
            self.phase = "start-loader"
            loader = subprocess.Popen(
                ["docker", "load"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            def feed_archive():
                try:
                    with open("/vault/docker/application.tar", "rb") as archive:
                        while block := archive.read(4096):
                            loader.stdin.write(block)
                            loader.stdin.flush()
                            time.sleep(0.05)
                    loader.stdin.close()
                    loader.wait()
                except (BrokenPipeError, OSError):
                    pass

            threading.Thread(target=feed_archive, daemon=True).start()
            require(loader.poll() is None, "Docker load did not start")
            self.respond({"load_in_progress": True, "marker": False})
        elif url.path == "/poweroff":
            self.respond({"requested_poweroff": True})
            run(["systemctl", "poweroff", "--no-block"])
        elif url.path == "/cross-vault":
            try:
                self.respond(cross_vault(urllib.parse.parse_qs(url.query)["resource"][0]))
            except Exception:
                self.respond({"test_failed": True})
        else:
            self.send_error(404)


http.server.HTTPServer(("0.0.0.0", 18081), Handler).serve_forever()
