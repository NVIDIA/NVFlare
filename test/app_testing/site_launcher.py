# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import traceback


def process_logs(log_path, pid):
    try:
        with open(log_path, "ab") as file:
            for line in pid.stdout:  # b'\n'-separated lines
                sys.stdout.buffer.write(line)  # pass bytes as is
                file.write(line)
                file.flush()

    except BaseException as e:
        traceback.print_exc()
        print(f"Exception in process_logs for file {log_path}: {e.__str__()}")


class SiteLauncher(object):
    def __init__(
        self, poc_directory, server_dir_name="server", client_dir_name="client", admin_dir_name="admin", app_path=None
    ):
        """
        This class sets up the test environment for a test. It will launch and keep track of servers and clients.
        """
        super(SiteLauncher, self).__init__()

        self.original_poc_directory = poc_directory
        self.server_dir_name = server_dir_name
        self.client_dir_name = client_dir_name
        self.admin_dir_name = admin_dir_name
        self.app_path = app_path

        self.server_properties = {}
        self.client_properties = {}

        self.admin_api = None

        self.logger = logging.getLogger("SiteRunner")

        # Create temporary poc directory
        if not os.path.exists(self.original_poc_directory):
            raise RuntimeError("Please run POC command first and provide the POC path!")

        # TODO: What is log directory and should it be added here?
        root_dir = tempfile.mkdtemp()
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
        shutil.copytree(self.original_poc_directory, root_dir)
        self.poc_directory = root_dir
        print(f"Using root dir: {root_dir}")

    def start_server(self):
        server_dir = os.path.join(self.poc_directory, self.server_dir_name)
        log_path = os.path.join(server_dir, "log.txt")
        # log_file = open(log_path, 'w')
        new_env = os.environ.copy()

        # Create upload directory
        os.makedirs(os.path.join(server_dir, "transfer"), exist_ok=True)

        command = (
            f"{sys.executable} -m nvflare.private.fed.app.server.server_train "
            f"-m {server_dir} -s fed_server.json"
            f" --set secure_train=false config_folder=config"
        )
        process = subprocess.Popen(
            shlex.split(command, " "),
            preexec_fn=os.setsid,
            env=new_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # process = subprocess.Popen(shlex.split(command, " "), preexec_fn=os.setsid, env=new_env)
        print("Starting server ...")

        t = threading.Thread(target=process_logs, args=(log_path, process))
        t.start()

        self.server_properties["path"] = server_dir
        self.server_properties["process"] = process
        # self.server_properties["log_file"] = log_file
        self.server_properties["log_path"] = log_path

    def start_clients(self, n=2):
        # Make sure that previous clients are killed
        self.stop_all_clients()
        self.client_properties.clear()

        # For each client, copy the directory
        src_client_directory = os.path.join(self.poc_directory, self.client_dir_name)
        new_env = os.environ.copy()

        for i in range(n):
            client_id = i
            client_name = self.client_dir_name + f"_{client_id}"

            # Copy and create new directory
            client_dir_name = os.path.join(self.poc_directory, client_name)
            shutil.copytree(src_client_directory, client_dir_name)
            log_path = os.path.join(client_dir_name, "log.txt")

            self.client_properties[client_id] = {}
            self.client_properties[client_id]["path"] = client_dir_name
            self.client_properties[client_id]["name"] = client_name
            self.client_properties[client_id]["log_path"] = log_path

            # Launch the new client
            client_startup_dir = os.path.join(client_dir_name)
            command = (
                f"{sys.executable} -m nvflare.private.fed.app.client.client_train -m "
                f"{client_startup_dir} -s fed_client.json --set secure_train=false config_folder=config"
                f" uid=client_{client_id}"
            )

            process = subprocess.Popen(
                shlex.split(command, " "),
                preexec_fn=os.setsid,
                env=new_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            # process = subprocess.Popen(shlex.split(command, " "), preexec_fn=os.setsid, env=new_env)
            self.client_properties[client_id]["process"] = process

            print(f"Launched client {client_id} process.")

            t = threading.Thread(target=process_logs, args=(log_path, process))
            t.start()

    def get_server_data(self):
        if "log_file" in self.server_properties:
            self.server_properties["log_file"].flush()

        server_data = {
            "server_path": self.server_properties["path"],
            "server_process": self.server_properties["process"],
            "server_name": self.server_dir_name,
            "root_dir": self.poc_directory,
            "log_path": self.server_properties["log_path"],
        }

        return server_data

    def get_client_data(self):
        client_data = {
            "client_paths": [self.client_properties[x]["path"] for x in self.client_properties],
            "client_names": [self.client_properties[x]["name"] for x in self.client_properties],
            "client_processes": [self.client_properties[x]["process"] for x in self.client_properties],
        }

        return client_data

    def stop_server(self):
        # Kill all clients first
        try:
            self.stop_all_clients()

            # Kill the process
            if "process" in self.server_properties and self.server_properties["process"]:
                os.killpg(self.server_properties["process"].pid, signal.SIGTERM)

                subprocess.call(["kill", str(self.server_properties["process"].pid)])
                self.server_properties["process"].wait()
                print("Sent SIGTERM to server.")
            else:
                print("No server process.")
        except Exception as e:
            print(f"Exception in stopping server: {e.__str__()}")
        finally:
            self.server_properties.clear()

    def stop_client(self, client_id) -> bool:
        if client_id not in self.client_properties:
            print(f"Client {client_id} not present in client processes.")
            return False
        if not self.client_properties[client_id]["process"]:
            print(f"Client {client_id} process is None.")
            self.client_properties.pop(client_id)
            return False

        try:
            os.killpg(self.client_properties[client_id]["process"].pid, signal.SIGTERM)

            subprocess.call(["kill", str(self.client_properties[client_id]["process"].pid)])
            self.client_properties[client_id]["process"].wait()

            self.client_properties.pop(client_id)

            print(f"Sent SIGTERM to client {client_id}.")
        except Exception as e:
            print(f"Exception in stopping client {client_id}: {e.__str__()}")
            return False

        return True

    def stop_all_clients(self):
        client_ids = list(self.client_properties.keys())
        for client_id in client_ids:
            self.stop_client(client_id)

    def stop_all_sites(self):
        self.stop_server()

    def client_status(self, client_id):
        pass

    def finalize(self):
        print(f"Deleting temporary directory: {self.poc_directory}.")
        shutil.rmtree(self.poc_directory)
