# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import os
import signal
import sys

import docker
from nvflare.apis.utils.format_check import name_check
from nvflare.lighter.utils import generate_password


def start(args):
    cwd = os.getcwd()
    if not args.folder:
        folder = cwd
    else:
        folder = os.path.join(cwd, args.folder)
    environment = dict()
    env_vars = args.env
    if env_vars:
        for e in env_vars:
            splitted = e.split("=")
            environment[splitted[0]] = splitted[1]
    passphrase = args.passphrase
    if passphrase:
        environment["NVFL_DASHBOARD_PP"] = passphrase
    if not os.path.exists(os.path.join(folder, "db.sqlite")):
        need_email = True
        while need_email:
            answer = input(
                "Please provide project admin email address.  This person will be the super user of the dashboard and this project.\n"
            )
            error, reason = name_check(answer, "email")
            if error:
                print(f"Expecting an email address, but got one in an invalid format.  Reason: {reason}")
            else:
                need_email = False
        print("generating random password")
        pwd = generate_password(8)
        print(f"Project admin credential is {answer} and the password is {pwd}")
        environment.update({"NVFL_CREDENTIAL": f"{answer}:{pwd}"})
    try:
        client = docker.from_env()
    except docker.errors.DockerException:
        print("Unable to communicate to docker daemon/socket.  Please make sure your docker is up and running.")
        exit(0)
    dashboard_image = "nvflare/nvflare"
    try:
        print(f"Pulling {dashboard_image}, may take some time to finish.")
        _ = client.images.pull(dashboard_image)
    except docker.errors.APIError:
        print(f"unable to pull {dashboard_image}")
        exit(1)
    print(f"Launching {dashboard_image}")
    print(f"Dashboard will listen to port {args.port}")
    print(f"{folder} on host mounted to /var/tmp/nvflare/dashboard in container")
    if environment:
        print(f"environment vars set to {environment}")
    else:
        print("No additional environment variables set to the launched container.")
    try:
        container_obj = client.containers.run(
            "nvflare/nvflare",
            entrypoint=["/usr/local/bin/python3", "nvflare/dashboard/wsgi.py"],
            detach=True,
            auto_remove=True,
            name="nvflare-dashboard",
            ports={8443: args.port},
            volumes={folder: {"bind": "/var/tmp/nvflare/dashboard", "model": "rw"}},
            environment=environment,
        )
    except docker.errors.APIError as e:
        print(f"Either {dashboard_image} image does not exist or another nvflare-dashboard instance is still running.")
        print("Please either provide an existing container image or stop the running container instance.")
        print(e)
        exit(1)
    if container_obj:
        print("Dashboard container started")
        print("Container name nvflare-dashboard")
        print(f"id is {container_obj.id}")
    else:
        print("Container failed to start")


def stop():
    try:
        client = docker.from_env()
    except docker.errors.DockerException:
        print("Unable to communicate to docker daemon/socket.  Please make sure your docker is up and running.")
        exit(0)
    try:
        container_obj = client.containers.get("nvflare-dashboard")
    except docker.errors.NotFound:
        print("No nvflare-dashboard container found")
        exit(0)
    container_obj.kill(signal=signal.SIGINT)
    print("nvflare-dashboard exited")


def has_no_arguments() -> bool:
    last_item = sys.argv[-1]
    return (
        last_item.endswith("dashboard.cli") or last_item.endswith("dashboard/cli.py") or last_item.endswith("dashboard")
    )


def main():
    parser = argparse.ArgumentParser()
    define_dashboard_parser(parser)
    args = parser.parse_args()
    handle_dashboard(args)


def define_dashboard_parser(parser):
    parser.add_argument("--start", action="store_true", help="start dashboard")
    parser.add_argument("--stop", action="store_true", help="stop dashboard")
    parser.add_argument("-p", "--port", type=str, default="443", help="port to listen")
    parser.add_argument(
        "-f", "--folder", type=str, help="folder containing necessary info (default: current working directory)"
    )
    parser.add_argument(
        "--passphrase", help="Passphrase to encrypt/decrypt root CA private key.  !!! Do not share it with others. !!!"
    )
    parser.add_argument("-e", "--env", action="append", help="additonal environment variables: var1=value1")


def handle_dashboard(args):
    if has_no_arguments():
        print("Add -h option to see usage")
        exit(0)
    if args.stop:
        stop()
    elif args.start:
        start(args)


if __name__ == "__main__":
    main()
