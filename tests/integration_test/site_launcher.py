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
import time
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


class SiteLauncher:
    def __init__(
        self,
        poc_directory,
        ha=False,
        overseer_dir_name="overseer",
        server_dir_name="server",
        client_dir_name="client",
        admin_dir_name="admin",
    ):
        """
        This class sets up the test environment for a test. It will launch and keep track of servers and clients.
        """
        super().__init__()

        self.original_poc_directory = poc_directory
        self.overseer_dir_name = overseer_dir_name
        self.server_dir_name = server_dir_name
        self.client_dir_name = client_dir_name
        self.admin_dir_name = admin_dir_name

        self.overseer_properties = {}
        self.server_properties = {}
        self.client_properties = {}

        self.admin_api = None

        self.logger = logging.getLogger(self.__class__.__name__)

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

        if ha:
            for name in ("admin", "server", "client"):
                dir = os.path.join(self.poc_directory, f"{name}", "startup")
                os.rename(os.path.join(dir, f"fed_{name}.json"), os.path.join(dir, f"fed_{name}_old.json"))
                os.rename(os.path.join(dir, f"fed_{name}_HA.json"), os.path.join(dir, f"fed_{name}.json"))

    def start_overseer(self):
        overseer_dir = os.path.join(self.poc_directory, self.overseer_dir_name)
        log_path = os.path.join(overseer_dir, "log.txt")
        new_env = os.environ.copy()
        # new_env.update({"FLASK_RUN_PORT": "6000"})
        # new_env.get("PYTHONPATH", "") + os.sep + "nvflare.ha.overseer.overseer"})
        command = f"{sys.executable} -m nvflare.ha.overseer.overseer"

        process = subprocess.Popen(
            shlex.split(command, " "),
            preexec_fn=os.setsid,
            env=new_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print("Starting overseer ...")

        # t = threading.Thread(target=process_logs, args=(log_path, process))
        # t.start()

        self.overseer_properties["path"] = overseer_dir
        self.overseer_properties["process"] = process
        self.overseer_properties["log_path"] = log_path

    def start_servers(self, n=1):
        src_server_dir = os.path.join(self.poc_directory, self.server_dir_name)

        # Create upload directory
        os.makedirs(os.path.join(src_server_dir, "transfer"), exist_ok=True)

        if n == 1:
            server_name = self.server_dir_name

            # Copy and create new directory
            server_dir_name = os.path.join(self.poc_directory, server_name)

            # replace fed_server.json ports
            fed_server_path = os.path.join(server_dir_name, "startup", "fed_server.json")
            with open(fed_server_path, "r") as f:
                fed_server_json = f.read()

            admin_port = "8003"
            fed_server_json = fed_server_json.replace("8002", "8002").replace("8003", admin_port)

            with open(fed_server_path, "w") as f:
                f.write(fed_server_json)

            self.start_server()

        else:
            for i in range(n):
                server_id = i
                server_name = self.server_dir_name + f"_{server_id}"

                # Copy and create new directory
                server_dir_name = os.path.join(self.poc_directory, server_name)
                shutil.copytree(src_server_dir, server_dir_name)

                # replace fed_server.json ports
                fed_server_path = os.path.join(server_dir_name, "startup", "fed_server.json")
                with open(fed_server_path, "r") as f:
                    fed_server_json = f.read()

                admin_port = f"8{i}03"
                fed_server_json = fed_server_json.replace("8002", f"8{i}02").replace("8003", admin_port)

                with open(fed_server_path, "w") as f:
                    f.write(fed_server_json)

                self.start_server(server_id)
                time.sleep(5)

    def start_server(self, server_id=None):
        if server_id is None:
            server_name = self.server_dir_name
            server_id = 0
        else:
            server_name = self.server_dir_name + f"_{server_id}"
        server_dir_name = os.path.join(self.poc_directory, server_name)
        log_path = os.path.join(server_dir_name, "log.txt")

        self.server_properties[server_id] = {}
        self.server_properties[server_id]["path"] = server_dir_name
        self.server_properties[server_id]["name"] = server_name
        self.server_properties[server_id]["port"] = f"8003"
        self.server_properties[server_id]["log_path"] = log_path

        command = (
            f"{sys.executable} -m nvflare.private.fed.app.server.server_train "
            f"-m {server_dir_name} -s fed_server.json"
            f" --set secure_train=false config_folder=config"
        )
        new_env = os.environ.copy()
        process = subprocess.Popen(
            shlex.split(command, " "),
            preexec_fn=os.setsid,
            env=new_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(f"start server with {command}: {process.pid}")
        self.server_properties[server_id]["process"] = process

        print(f"Starting server {server_id}...")

        # t = threading.Thread(target=process_logs, args=(log_path, process))
        # t.start()

        # self.server_properties["path"] = server_dir_name
        # self.server_properties["process"] = process
        # self.server_properties["log_path"] = log_path

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

            self.start_client(client_id)

    def start_client(self, client_id):
        client_name = self.client_dir_name + f"_{client_id}"
        client_dir_name = os.path.join(self.poc_directory, client_name)
        log_path = os.path.join(client_dir_name, "log.txt")
        new_env = os.environ.copy()

        self.client_properties[client_id] = {}
        self.client_properties[client_id]["path"] = client_dir_name
        self.client_properties[client_id]["name"] = client_name
        self.client_properties[client_id]["log_path"] = log_path

        # Launch the new client
        client_startup_dir = os.path.join(client_dir_name)
        command = (
            f"{sys.executable} -m nvflare.private.fed.app.client.client_train -m "
            f"{client_startup_dir} -s fed_client.json --set secure_train=false config_folder=config"
            f" uid={client_name}"
        )

        process = subprocess.Popen(
            shlex.split(command, " "),
            preexec_fn=os.setsid,
            env=new_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.client_properties[client_id]["process"] = process

        print(f"Launched client {client_id} process.")

        # t = threading.Thread(target=process_logs, args=(log_path, process))
        # t.start()

    def get_active_server_id(self, port):
        active_server_id = -1
        for i in range(len(self.server_properties)):
            if (
                self.server_properties[i]
                and "port" in self.server_properties[i]
                and self.server_properties[i]["port"] == str(port)
            ):
                active_server_id = i
        return active_server_id

    def get_server_data(self, server_id=0):
        server_prop = self.server_properties[server_id]
        if "log_file" in server_prop:
            server_prop["log_file"].flush()

        server_data = {}
        if server_prop:
            server_data = {
                "server_path": server_prop["path"],
                "server_process": server_prop["process"],
                "server_name": server_prop["name"],
                "root_dir": self.poc_directory,
                "log_path": server_prop["log_path"],
            }

        return server_data

    def get_client_data(self):
        client_data = {
            "client_paths": [self.client_properties[x]["path"] for x in self.client_properties],
            "client_names": [self.client_properties[x]["name"] for x in self.client_properties],
            "client_processes": [self.client_properties[x]["process"] for x in self.client_properties],
        }

        return client_data

    def stop_overseer(self):
        try:
            # Kill the process
            if "process" in self.overseer_properties and self.overseer_properties["process"]:
                os.killpg(self.overseer_properties["process"].pid, signal.SIGTERM)

                subprocess.call(["kill", str(self.overseer_properties["process"].pid)])
                self.overseer_properties["process"].wait()
                print("Sent SIGTERM to overseer.")
            else:
                print("No overseer process.")
        except Exception as e:
            print(f"Exception in stopping overseer: {e.__str__()}")
        finally:
            self.overseer_properties.clear()

    def stop_server(self, server_id):
        server_prop = self.server_properties[server_id]
        # Kill all clients first
        try:
            # clients will be stopped by a separate call
            # self.stop_all_clients()

            # Kill the process
            if "process" in server_prop and server_prop["process"]:
                os.killpg(server_prop["process"].pid, signal.SIGTERM)

                subprocess.call(["kill", str(server_prop["process"].pid)])
                server_prop["process"].wait()
                print("Sent SIGTERM to server.")
            else:
                print("No server process.")
        except Exception as e:
            print(f"Exception in stopping server: {e.__str__()}")
        finally:
            server_prop.clear()

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

    def stop_all_servers(self):
        server_ids = list(self.server_properties.keys())
        for server_id in server_ids:
            self.stop_server(server_id)

    def stop_all_sites(self):
        self.stop_all_clients()
        self.stop_all_servers()
        self.stop_overseer()

    def cleanup(self):
        print(f"Deleting temporary directory: {self.poc_directory}.")
        shutil.rmtree(self.poc_directory)
