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

"""Site launcher that provisions startup kits via the distributed provisioning CLI.

Replaces `nvflare provision -p project.yml` with the manual workflow:
  nvflare cert init  →  nvflare cert csr  →  nvflare cert sign  →  nvflare package

This exercises the full cert + package CLI end-to-end in an integration context.
"""

import os
import shlex
import shutil
import sys
import time

from .site_launcher import ServerProperties, SiteLauncher, SiteProperties, kill_process
from .utils import (
    cleanup_job_and_snapshot,
    run_command_in_subprocess,
    update_job_store_path_in_workspace,
    update_snapshot_path_in_workspace,
)

WORKSPACE = "ci_workspace_dist"


def _nvflare(args: str) -> int:
    """Run `nvflare <args>` as a subprocess; return exit code."""
    cmd = f"{sys.executable} -m nvflare {args}"
    proc = run_command_in_subprocess(cmd)
    proc.wait()
    return proc.returncode


def _provision_participant(work_dir: str, ca_dir: str, name: str, cert_type: str, endpoint: str, server_name: str):
    """Run cert csr → cert sign → package for a single participant.

    Produces a startup kit at <work_dir>/<name>/.
    """
    csr_dir = os.path.join(work_dir, name, "csr")
    signed_dir = os.path.join(work_dir, name, "signed")
    kit_dir = os.path.join(work_dir, name)
    os.makedirs(csr_dir, exist_ok=True)
    os.makedirs(signed_dir, exist_ok=True)

    # 1. Generate private key + CSR
    rc = _nvflare(f"cert csr -n {shlex.quote(name)} -t {cert_type} -o {shlex.quote(csr_dir)} --force")
    if rc != 0:
        raise RuntimeError(f"cert csr failed for {name} (exit {rc})")

    # 2. Sign with root CA
    csr_file = os.path.join(csr_dir, f"{name}.csr")
    rc = _nvflare(
        f"cert sign -r {shlex.quote(csr_file)} -c {shlex.quote(ca_dir)} "
        f"-o {shlex.quote(signed_dir)} -t {cert_type} --force"
    )
    if rc != 0:
        raise RuntimeError(f"cert sign failed for {name} (exit {rc})")

    # 3. Assemble startup kit
    cert_file = os.path.join(signed_dir, f"{cert_type}.crt")
    key_file = os.path.join(csr_dir, f"{name}.key")
    rootca_file = os.path.join(signed_dir, "rootCA.pem")

    pkg_args = (
        f"package -n {shlex.quote(name)} -t {cert_type} "
        f"-e {shlex.quote(endpoint)} "
        f"--cert {shlex.quote(cert_file)} "
        f"--key {shlex.quote(key_file)} "
        f"--rootca {shlex.quote(rootca_file)} "
        f"-o {shlex.quote(kit_dir)} --force"
    )
    if server_name:
        pkg_args += f" --server-name {shlex.quote(server_name)}"
    if cert_type == "server":
        # Disable require_signed_jobs so standard (unsigned) test jobs can run
        pkg_args += " --require-signed-jobs false"

    rc = _nvflare(pkg_args)
    if rc != 0:
        raise RuntimeError(f"package failed for {name} (exit {rc})")

    return kit_dir


def _start_site(site_properties: SiteProperties):
    start_sh = os.path.join(site_properties.root_dir, "startup", "start.sh")
    process = run_command_in_subprocess(f"bash {shlex.quote(start_sh)}")
    print(f"Starting {site_properties.name} ...")
    site_properties.process = process


def _stop_site(site_properties: SiteProperties):
    stop_sh = os.path.join(site_properties.root_dir, "startup", "stop_fl.sh")
    run_command_in_subprocess(f"bash {shlex.quote(stop_sh)}", stdin_data=b"y\n")
    print(f"Stopping {site_properties.name} ...")


class DistProvisionSiteLauncher(SiteLauncher):
    """Provision a federation using the distributed provisioning CLI.

    Args:
        server_name:  CN for the server cert (must resolve to localhost in tests).
        server_port:  gRPC port the server listens on.
        admin_port:   Admin console port (default: server_port + 1).
        client_names: List of client site names.
        admin_name:   Admin user name (cert CN).
        project_name: Project name used for cert init.
    """

    def __init__(
        self,
        server_name: str = "localhost",
        server_port: int = 8002,
        admin_port: int = None,
        client_names=("site-1", "site-2"),
        admin_name: str = "admin",
        project_name: str = "dist_prov_test",
    ):
        super().__init__()
        self.server_name = server_name
        self.server_port = server_port
        self.admin_port = admin_port if admin_port is not None else server_port + 1
        self.client_names = list(client_names)
        self.admin_name = admin_name
        self.admin_user_names = [admin_name]
        self.project_name = project_name
        self.work_dir = os.path.abspath(WORKSPACE)
        self.endpoint = f"grpc://{server_name}:{server_port}"

        # Pre-populate site property stubs (root_dir set after prepare_workspace)
        server_kit_dir = os.path.join(self.work_dir, server_name)
        self.server_properties[server_name] = ServerProperties(server_name, server_kit_dir, None, self.admin_port)
        for cn in client_names:
            client_kit_dir = os.path.join(self.work_dir, cn)
            self.client_properties[cn] = SiteProperties(cn, client_kit_dir, None)

    def prepare_workspace(self) -> str:
        os.makedirs(self.work_dir, exist_ok=True)

        # 1. Initialize root CA
        ca_dir = os.path.join(self.work_dir, "ca")
        rc = _nvflare(f"cert init -n {shlex.quote(self.project_name)} -o {shlex.quote(ca_dir)} --force")
        if rc != 0:
            raise RuntimeError(f"cert init failed (exit {rc})")

        # 2. Provision server
        _provision_participant(
            self.work_dir, ca_dir, self.server_name, "server", self.endpoint, server_name=None
        )

        # 3. Provision clients
        for cn in self.client_names:
            _provision_participant(
                self.work_dir, ca_dir, cn, "client", self.endpoint, server_name=self.server_name
            )

        # 4. Provision admin
        _provision_participant(
            self.work_dir, ca_dir, self.admin_name, "lead", self.endpoint, server_name=self.server_name
        )

        # 5. Update job/snapshot store paths so each test run gets a clean store
        server_kit = self.server_properties[self.server_name].root_dir
        update_job_store_path_in_workspace(self.work_dir, self.server_name)
        update_snapshot_path_in_workspace(self.work_dir, self.server_name)
        cleanup_job_and_snapshot(self.work_dir, self.server_name)

        return self.work_dir

    # SiteLauncher interface ---------------------------------------------------

    def start_overseer(self):
        pass  # No overseer in single-server mode

    def stop_overseer(self):
        pass

    def start_servers(self):
        self.start_server(self.server_name)
        time.sleep(3.0)

    def start_clients(self):
        for cn in self.client_names:
            self.start_client(cn)

    def start_server(self, server_id: str):
        _start_site(self.server_properties[server_id])

    def stop_server(self, server_id: str):
        _stop_site(self.server_properties[server_id])
        super().stop_server(server_id)

    def start_client(self, client_id: str):
        _start_site(self.client_properties[client_id])

    def stop_client(self, client_id: str):
        _stop_site(self.client_properties[client_id])
        super().stop_client(client_id)

    def cleanup(self):
        proc = run_command_in_subprocess(f"pkill -9 -f {shlex.quote(WORKSPACE)}")
        proc.wait()
        for sn in self.server_properties:
            try:
                cleanup_job_and_snapshot(self.work_dir, sn)
            except Exception:
                pass
        shutil.rmtree(self.work_dir, ignore_errors=True)
        super().cleanup()
