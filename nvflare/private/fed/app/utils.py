# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import signal
import threading
import time

import psutil

from nvflare.fuel.hci.security import hash_password
from nvflare.private.defs import SSLConstants
from nvflare.private.fed.runner import Runner
from nvflare.private.fed.server.admin import FedAdminServer
from nvflare.private.fed.server.fed_server import FederatedServer


def monitor_parent_process(runner: Runner, parent_pid, stop_event: threading.Event):
    while True:
        if stop_event.is_set() or not psutil.pid_exists(parent_pid):
            runner.stop()
            break
        time.sleep(1)


def check_parent_alive(parent_pid, stop_event: threading.Event):
    while True:
        if stop_event.is_set() or not psutil.pid_exists(parent_pid):
            pid = os.getpid()
            kill_child_processes(pid)
            os.killpg(os.getpgid(pid), 9)
            break
        time.sleep(1)


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)


def create_admin_server(fl_server: FederatedServer, server_conf=None, args=None, secure_train=False):
    """To create the admin server.

    Args:
        fl_server: fl_server
        server_conf: server config
        args: command args
        secure_train: True/False

    Returns:
        A FedAdminServer.
    """
    users = {}
    # Create a default user admin:admin for the POC insecure use case.
    if not secure_train:
        users = {"admin": hash_password("admin")}

    root_cert = server_conf[SSLConstants.ROOT_CERT] if secure_train else None
    server_cert = server_conf[SSLConstants.CERT] if secure_train else None
    server_key = server_conf[SSLConstants.PRIVATE_KEY] if secure_train else None
    admin_server = FedAdminServer(
        cell=fl_server.cell,
        fed_admin_interface=fl_server.engine,
        users=users,
        cmd_modules=fl_server.cmd_modules,
        file_upload_dir=os.path.join(args.workspace, server_conf.get("admin_storage", "tmp")),
        file_download_dir=os.path.join(args.workspace, server_conf.get("admin_storage", "tmp")),
        host=server_conf.get("admin_host", "localhost"),
        port=server_conf.get("admin_port", 5005),
        ca_cert_file_name=root_cert,
        server_cert_file_name=server_cert,
        server_key_file_name=server_key,
        accepted_client_cns=None,
        download_job_url=server_conf.get("download_job_url", "http://"),
    )
    return admin_server
