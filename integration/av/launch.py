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

import argparse
import os
import shlex
import subprocess

from nvflare.fuel.flare_api.flare_api import new_secure_session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_name", "-u", type=str, help="admin user name", required=True)
    parser.add_argument("--user_workspace", "-uw", type=str, help="admin user workspace folder", required=True)
    parser.add_argument("--site_name", "-s", type=str, help="flare site name", required=True)
    parser.add_argument("--site_workspace", "-sw", type=str, help="site workspace folder", required=True)
    parser.add_argument("--agent_id", "-a", type=str, help="agent id", required=True)
    parser.add_argument("--program", "-p", type=str, help="program to launch", required=False, default="av_trainer")

    args = parser.parse_args()
    print("logging in to FLARE Admin ...")
    sess = new_secure_session(
        username=args.user_name,
        startup_kit_location=args.user_workspace,
    )

    conn_info = sess.get_cell_conn_info(args.site_name)
    cp_url = conn_info["int_listener"]["url"]
    print(f"Got CP URL {cp_url}")

    program = args.program
    launch_cmd = f"python {program}.py -w {args.site_workspace} -s {args.site_name} -a {args.agent_id} -u {cp_url}"
    print(f"Launching: {launch_cmd}")

    new_env = os.environ.copy()
    process = subprocess.Popen(shlex.split(launch_cmd, True), preexec_fn=os.setsid, env=new_env)
    print(f"Launched trainer for {args.site_name}. PID={process.pid}")

    # print("Waiting for trainer to finish")
    # process.wait()
    #
    # print(f"Trainer finished with RC {process.returncode}")


if __name__ == "__main__":
    main()
