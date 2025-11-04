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
import shutil

import nvflare.lighter.utils as utils
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Participant, ParticipantType, Project
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.impl.edge import EdgeBuilder
from nvflare.lighter.impl.signature import SignatureBuilder
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.spec import Builder, Packager

PROV_KEY_ANALYZE = "analyze"
PROV_KEY_LCP_ONLY = "lcp_only"
PROV_KEY_ROOT_DIR = "root_dir"
PROV_KEY_PROJ_NAME = "project_name"
PROV_KEY_DEPTH = "depth"
PROV_KEY_WIDTH = "width"
PROV_KEY_MAX_SITES = "max_sites"
PROV_KEY_CLIENTS = "clients"
PROV_KEY_RP_PORT = "rp"


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


LCP_MAP_BASENAME = "lcp_map.json"
LOCAL_HOST = "localhost"
CA_CERT_NAME = "rootCA.pem"
SIMULATION_CONFIG = "simulation_config.json"
RUN_SIMULATOR = "python -m nvflare.edge.simulation.run_device_simulator"


class _Packager(Packager):

    def __init__(self, lcp_map, rp_port):
        self.lcp_map = lcp_map
        self.rp_port = rp_port

    def package(self, project: Project, ctx: ProvisionContext):
        location = ctx.get_result_location()
        script_dir = os.path.join(location, "scripts")
        os.mkdir(script_dir)
        lcp_map_file_name = os.path.join(script_dir, LCP_MAP_BASENAME)

        with open(lcp_map_file_name, "wt") as f:
            json.dump(self.lcp_map, f, indent=4)

        # copy CA cert to demo dir
        ca_cert_path = os.path.join(location, "server", "startup", CA_CERT_NAME)
        shutil.copy(ca_cert_path, os.path.join(script_dir, CA_CERT_NAME))

        utils.write(
            file_full_path=os.path.join(script_dir, "start_rp.sh"),
            content=f"python -m nvflare.edge.web.routing_proxy {self.rp_port} {LCP_MAP_BASENAME} {CA_CERT_NAME}",
            mode="t",
            exe=True,
        )

        utils.write(
            file_full_path=os.path.join(script_dir, "simulate_lcp.sh"),
            content=f"{RUN_SIMULATOR} {SIMULATION_CONFIG} -m {LCP_MAP_BASENAME} -c {CA_CERT_NAME}",
            mode="t",
            exe=True,
        )

        utils.write(
            file_full_path=os.path.join(script_dir, "simulate_rp.sh"),
            content=f"{RUN_SIMULATOR} {SIMULATION_CONFIG}",
            mode="t",
            exe=True,
        )

        sample_sim_config = {
            "job_name": "edge_job",
            "endpoint": f"http://localhost:{self.rp_port}",
            "num_devices": 10000,
            "num_workers": 30,
            "processor": {
                "path": "nvflare.edge.simulation.devices.num.NumProcessor",
                "args": {"min_train_time": 0.2, "max_train_time": 1.0},
            },
        }

        with open(os.path.join(script_dir, SIMULATION_CONFIG), "wt") as f:
            json.dump(sample_sim_config, f, indent=4)


def _build_tree(
    lcp_only: bool,
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
        lcp_only: only generate leaf CPs
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
            edge_service_port = PortManager.get_port()
            props = {
                "connect_to": {"name": parent.name},
                "listening_host": LOCAL_HOST,  # create server cert for the Edge API Service
                "edge_service_port": edge_service_port,
            }
            if not lcp_only:
                props["parent"] = parent.client_name
            client = _new_participant(name, ParticipantType.CLIENT, props=props)
            project.add_participant(client)
            Stats.num_clients += 1
            Stats.num_leaf_clients += 1
            lcp_map[name] = {"host": LOCAL_HOST, "port": edge_service_port}
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

        props = {
            "listening_host": {
                "default_host": LOCAL_HOST,
                "port": child.port,
            },
        }
        if depth > 0:
            props["connect_to"] = {"name": parent.name}

        relay = _new_participant(
            child.name,
            ParticipantType.RELAY,
            props=props,
        )
        project.add_participant(relay)
        Stats.num_relays += 1

        # attach a client to the replay and make it a child of the parent relay's attached client
        if not lcp_only:
            client = _new_participant(
                child.client_name, ParticipantType.CLIENT, props={"connect_to": {"name": child.name}}
            )
            if depth > 0:
                client.set_prop("parent", parent.client_name)

            project.add_participant(client)
            Stats.num_clients += 1
            Stats.num_non_leaf_clients += 1

        # depth-first recursion
        _build_tree(lcp_only, depth + 1, width, max_depth, child, num_clients, project, lcp_map)


def hierachical_provision(params: dict, project: Project, builders: list[Builder], admins):
    if not isinstance(project, Project):
        raise ValueError("Expected 'project' to be a Project instance")
    if not isinstance(builders, list) or not all(isinstance(b, Builder) for b in builders):
        raise ValueError("Expected 'builders' to be a list of Builder instances")
    if not isinstance(admins, list):
        raise ValueError("Expected 'admins' to be a list")
    if params is None:
        raise ValueError("'params' must not be None")

    required_params = [PROV_KEY_DEPTH, PROV_KEY_WIDTH, PROV_KEY_CLIENTS]
    for key in required_params:
        if key not in params:
            raise ValueError(f"Missing required parameter '{key}' in 'params'")

    depth = params.get(PROV_KEY_DEPTH)
    if depth < 1 or depth > 5:
        print(f"bad depth {depth}: must be [1..5]")
        return
    width = params.get(PROV_KEY_WIDTH)
    if width <= 1 or width > 9:
        print(f"bad width {depth}: must be [2..9]")
        return
    clients = params.get(PROV_KEY_CLIENTS)
    if clients <= 1 or clients > 9:
        print(f"bad clients-per-leaf-node {clients}: must be [2..9]")
        return

    # add server
    server = _new_participant(
        "server",
        ParticipantType.SERVER,
        props={
            "fed_learn_port": 8002,
            "admin_port": 8003,
            "host_names": [LOCAL_HOST, "127.0.0.1"],
            "default_host": LOCAL_HOST,
        },
    )
    project.add_participant(server)

    # add relays and clients
    root_relay = _Node()
    root_relay.name = "R"
    lcp_map = {}
    lcp_only = params.get(PROV_KEY_LCP_ONLY)
    _build_tree(lcp_only, 0, width, depth, root_relay, clients, project, lcp_map)

    total_sites = Stats.num_clients + Stats.num_relays + 1

    print(f"Relays:  leaf={Stats.num_leaf_relays}; non-leaf={Stats.num_non_leaf_relays}; total={Stats.num_relays}")
    print(f"Clients: leaf={Stats.num_leaf_clients}; non-leaf={Stats.num_non_leaf_clients}; total={Stats.num_clients}")
    print(f"Total Sites: {total_sites}")

    analyze = params.get(PROV_KEY_ANALYZE)
    if analyze:
        return

    max_sites = params.get(PROV_KEY_MAX_SITES)

    if max_sites and total_sites > max_sites:
        print(f"Too many sites: {total_sites} > {max_sites}")
        return

    for admin in admins:
        project.add_participant(admin)

    root_dir = params.get(PROV_KEY_ROOT_DIR)
    rp = params.get(PROV_KEY_RP_PORT)
    provisioner = Provisioner(root_dir, builders, _Packager(lcp_map, rp))
    provisioner.provision(project)


def main():
    parser = argparse.ArgumentParser()

    # analyze only and do not do provision
    parser.add_argument("--analyze", "-a", action="store_true", help="only analyze but does not generate files")

    # LCP only
    parser.add_argument("--lcp_only", "-l", action="store_true", help="only generate leaf CPs")

    # where the result will be stored
    parser.add_argument("--root_dir", "-r", type=str, help="project root dir", required=True)

    parser.add_argument("--project_name", "-p", type=str, help="project name", required=True)
    parser.add_argument("--depth", "-d", type=int, help="depth of the relay tree", required=True)
    parser.add_argument(
        "--width",
        "-w",
        type=int,
        help="width of the relay tree, the number of children for each relay node",
        required=True,
        default=2,
    )
    parser.add_argument(
        "--clients", "-c", type=int, help="number of clients per leaf relay node", required=True, default=2
    )

    # number of sites will go up exponentially when depth goes up.
    # do not do provision if the number of sites exceeds max_sites
    parser.add_argument("--max_sites", "-m", type=int, help="max number sites", required=False, default=100)

    parser.add_argument("--rp", "-rp", type=int, help="routing proxy port", required=False, default=4321)

    args = parser.parse_args()
    params = {
        PROV_KEY_ANALYZE: args.analyze,
        PROV_KEY_LCP_ONLY: args.lcp_only,
        PROV_KEY_ROOT_DIR: args.root_dir,
        PROV_KEY_PROJ_NAME: args.project_name,
        PROV_KEY_DEPTH: args.depth,
        PROV_KEY_WIDTH: args.width,
        PROV_KEY_MAX_SITES: args.max_sites,
        PROV_KEY_CLIENTS: args.clients,
        PROV_KEY_RP_PORT: args.rp,
    }

    builders = [
        WorkspaceBuilder(["master_template.yml"]),
        StaticFileBuilder(
            config_folder="config",
            scheme="grpc",
        ),
        CertBuilder(),
        SignatureBuilder(),
        EdgeBuilder(),
    ]

    project_name = params.get(PROV_KEY_PROJ_NAME)
    project = Project(
        name=project_name,
        description="Edge test project",
        props={
            "api_version": 3,
            "connection_security": "clear",
        },
    )

    admins = [
        _new_participant(
            "admin@nvidia.com", ParticipantType.ADMIN, props={"role": "project_admin", "connect_to": LOCAL_HOST}
        )
    ]

    hierachical_provision(params, project, builders, admins)


if __name__ == "__main__":
    main()
