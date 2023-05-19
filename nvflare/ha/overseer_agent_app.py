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
from pprint import pprint

from nvflare.ha.overseer_agent import HttpOverseerAgent


def setup_basic_info():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", type=str, default="example_project", help="project name")
    parser.add_argument("-r", "--role", type=str, help="role (server, client or admin)")
    parser.add_argument("-n", "--name", type=str, help="globally unique name")
    parser.add_argument("-f", "--fl_port", type=str, help="fl port number")
    parser.add_argument("-a", "--admin_port", type=str, help="adm port number")
    parser.add_argument("-s", "--sleep", type=float, help="sleep (seconds) in heartbeat")
    parser.add_argument("-c", "--ca_path", type=str, help="root CA path")
    parser.add_argument("-o", "--overseer_url", type=str, help="Overseer URL")
    parser.add_argument("-t", "--cert_path", type=str, help="cert path")
    parser.add_argument("-v", "--prv_key_path", type=str, help="priviate key path")

    args = parser.parse_args()

    overseer_agent = HttpOverseerAgent(
        overseer_end_point=args.overseer_url,
        project=args.project,
        role=args.role,
        name=args.name,
        fl_port=args.fl_port,
        admin_port=args.admin_port,
        heartbeat_interval=args.sleep,
    )

    if args.ca_path:
        overseer_agent.set_secure_context(
            ca_path=args.ca_path, cert_path=args.cert_path, prv_key_path=args.prv_key_path
        )
    return overseer_agent


def main():
    overseer_agent = setup_basic_info()
    overseer_agent.start(simple_callback, conditional_cb=True)
    while True:
        answer = input("(p)ause/(r)esume/(s)witch/(d)ump/(e)nd? ")
        normalized_answer = answer.strip().upper()
        if normalized_answer == "P":
            overseer_agent.pause()
        elif normalized_answer == "R":
            overseer_agent.resume()
        elif normalized_answer == "E":
            overseer_agent.end()
            break
        elif normalized_answer == "D":
            pprint(overseer_agent.overseer_info)
        elif normalized_answer == "":
            continue
        elif normalized_answer[0] == "S":
            split = normalized_answer.split()
            if len(split) == 2:
                sp_index = int(split[1])
            else:
                print("expect sp index but got nothing.  Please provide the sp index to be promoted")
                continue
            try:
                sp = overseer_agent.overseer_info.get("sp_list")[sp_index]
            except IndexError:
                print("index out of range")
            else:
                resp = overseer_agent.promote_sp(sp.get("sp_end_point"))
                pprint(resp.json())


def simple_callback(overseer_agent):
    print(f"\nGot callback {overseer_agent.get_primary_sp()}")


if __name__ == "__main__":
    main()
