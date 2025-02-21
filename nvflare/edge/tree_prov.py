# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
This tool is for testing only. Do not use it for production purpose.
This tool automatically provisions a project with relay and client hierarchy based on user provided parameters.
"""

import argparse
import json
import os.path

from nvflare.lighter.entity import Participant, ParticipantType, Project
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.impl.signature import SignatureBuilder
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.provisioner import Provisioner


def _new_participant(name: str, ptype: str, props: dict) -> Participant:
    return Participant(type=ptype, name=name, org="nvidia", props=props)


def _make_client_name(relay_name: str) -> str:
    return relay_name.replace("R", "C")


class Stats:
    num_relays = 0
    num_leaf_relays = 0
    num_non_leaf_relays = 0

    num_clients = 0
    num_leaf_clients = 0
    num_non_leaf_clients = 0


class PortManager:
    last_port_number = 9000

    @classmethod
    def get_port(cls):
        cls.last_port_number += 1
        return cls.last_port_number


class _Node:
    def __init__(self):
        self.name = None
        self.client_name = None
        self.parent = None
        self.children = []
        self.port = PortManager.get_port()


def _build_tree(
    depth: int,
    width: int,
    max_depth: int,
    parent: _Node,
    num_clients: int,
    project: Project,
    lcp_map: dict,
):
    """Build relay hierarchy and client hierarchy, recursively.

    Relays are organized hierarchically. Attach a client to each relay. Such clients are non-leaf clients (a.k.a
    aggregation clients). In client hierarchy, the client attached to a relay is the child of the client attached to
    the relay's parent relay. If the relay doesn't have a parent relay, then the client won't have a parent client.

    Create num_clients leaf clients for each leaf relay.

    Stats are collected during the building process.

    Args:
        depth: current depth of the tree being built
        width: how many child nodes for each non-leaf node
        max_depth: how deep the relay tree is
        parent: the parent relay node
        num_clients: number of clients to create for each leaf node
        project: the project to add the sites to

    Returns: None

    """
    if depth == max_depth:
        # the parent is a leaf node - add leaf clients (LCPs)
        Stats.num_leaf_relays += 1
        for i in range(num_clients):
            name = _make_client_name(parent.name) + str(i + 1)
            client = _new_participant(
                name, ParticipantType.CLIENT, props={"parent": parent.client_name, "connect_to": {"name": parent.name}}
            )
            project.add_participant(client)
            Stats.num_clients += 1
            Stats.num_leaf_clients += 1

            lcp_map[name] = {"host": "localhost", "port": PortManager.get_port()}
        return

    if depth > 0:
        # ignore level 0, which is the root that is not treated as a site.
        Stats.num_non_leaf_relays += 1

    for i in range(width):
        child = _Node()
        child.name = parent.name + str(i + 1)
        child.client_name = _make_client_name(child.name)
        child.parent = parent
        parent.children.append(child)

        relay = _new_participant(
            child.name,
            ParticipantType.RELAY,
            props={
                "listening_host": {
                    "default_host": "localhost",
                    "port": child.port,
                },
            },
        )
        project.add_participant(relay)
        Stats.num_relays += 1

        # attach a client to the replay and make it a child of the parent relay's attached client
        client = _new_participant(child.client_name, ParticipantType.CLIENT, props={"connect_to": {"name": child.name}})
        if depth >= 1:
            client.set_prop("parent", parent.client_name)

        project.add_participant(client)
        Stats.num_clients += 1
        Stats.num_non_leaf_clients += 1

        # depth-first recursion
        _build_tree(depth + 1, width, max_depth, child, num_clients, project, lcp_map)


def main():
    parser = argparse.ArgumentParser()

    # analyze only and do not do provision
    parser.add_argument("--analyze", "-a", action="store_true", help="only analyze but does not generate files")

    # where the result will be stored
    parser.add_argument("--root_dir", "-r", type=str, help="project root dir", required=True)

    parser.add_argument("--project_name", "-p", type=str, help="project name", required=True)
    parser.add_argument("--depth", "-d", type=int, help="depth of the relay tree", required=True)
    parser.add_argument("--width", "-w", type=int, help="width of each tree", required=False, default=2)

    # number of sites will go up exponentially when depth goes up.
    # do not do provision if the number of sites exceeds max_sites
    parser.add_argument("--max_sites", "-m", type=int, help="max number sites", required=False, default=100)

    parser.add_argument("--clients", "-c", type=int, help="number of clients per leaf node", required=False, default=2)
    args = parser.parse_args()

    if args.depth < 1 or args.depth > 5:
        print(f"bad depth {args.depth}: must be [1..5]")
        return

    if args.width <= 1 or args.width > 9:
        print(f"bad width {args.depth}: must be [2..9]")
        return

    if args.clients <= 1 or args.clients > 9:
        print(f"bad clients-per-leaf-node {args.clients}: must be [2..9]")
        return

    overseer_agent = {
        "path": "nvflare.ha.dummy_overseer_agent.DummyOverseerAgent",
        "overseer_exists": False,
        "args": {"sp_end_point": "server:8002:8003"},
    }

    builders = [
        WorkspaceBuilder(["master_template.yml"]),
        StaticFileBuilder(
            config_folder="config",
            scheme="grpc",
            overseer_agent=overseer_agent,
        ),
        CertBuilder(),
        SignatureBuilder(),
    ]

    provisioner = Provisioner(args.root_dir, builders)
    project = Project(
        name=args.project_name,
        description="this is a test",
        props={
            "api_version": 3,
            "connection_security": "clear",
        },
    )

    # add server
    server = _new_participant(
        "server",
        ParticipantType.SERVER,
        props={
            "fed_learn_port": 8002,
            "admin_port": 8003,
            "host_names": ["localhost", "127.0.0.1"],
            "default_host": "localhost",
        },
    )
    project.add_participant(server)

    # add relays and clients
    root_relay = _Node()
    root_relay.name = "R"
    lcp_map = {}
    _build_tree(0, args.width, args.depth, root_relay, args.clients, project, lcp_map)

    total_sites = Stats.num_clients + Stats.num_relays + 1

    print(f"Relays:  leaf={Stats.num_leaf_relays}; non-leaf={Stats.num_non_leaf_relays}; total={Stats.num_relays}")
    print(f"Clients: leaf={Stats.num_leaf_clients}; non-leaf={Stats.num_non_leaf_clients}; total={Stats.num_clients}")
    print(f"Total Sites: {total_sites}")

    if args.analyze:
        return

    if total_sites > args.max_sites:
        print(f"Too many sites: {total_sites} > {args.max_sites}")
        return

    # add admins
    admin = _new_participant(
        "admin@nvidia.com", ParticipantType.ADMIN, props={"role": "project_admin", "connect_to": "localhost"}
    )
    project.add_participant(admin)
    ctx = provisioner.provision(project)
    location = ctx.get_result_location()
    lcp_map_file_name = os.path.join(location, "lcp_map.json")
    with open(lcp_map_file_name, "wt") as f:
        json.dump(lcp_map, f, indent=4)
    print(f"Generated LCP Map: {lcp_map_file_name}")


if __name__ == "__main__":
    main()
