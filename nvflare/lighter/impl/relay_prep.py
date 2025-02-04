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
from nvflare.lighter.constants import PropKey
from nvflare.lighter.entity import Participant, parse_connect_to
from nvflare.lighter.spec import Builder, Project, ProvisionContext


def check_parent(c: Participant, path: list):
    if c.name in path:
        return f"circular parent ref {c.name}"

    path.insert(0, c.name)
    parent, _, _ = c.get_prop(PropKey.PARENT)
    if not parent:
        return ""
    return check_parent(parent, path)


class ReplayPrep(Builder):
    def initialize(self, project: Project, ctx: ProvisionContext):
        # name => relay
        name_to_relay = {}

        relays = project.get_relays()
        if not relays:
            # nothing to prepare
            return

        for r in relays:
            assert isinstance(r, Participant)
            name_to_relay[r.name] = r

        # determine parents
        for r in relays:
            assert isinstance(r, Participant)
            parent_def = r.get_prop(PropKey.CONNECT_TO)
            parent_name, parent_addr, port = parse_connect_to(parent_def)
            if not parent_name:
                parent = None
            else:
                parent = name_to_relay.get(parent_name)
                if not parent:
                    raise ValueError(f"undefined parent {parent_name} in relay {r.name}")
            r.add_prop(PropKey.PARENT, (parent, parent_addr, port))

        # determine FQCNs
        for r in relays:
            fqcn_path = []
            err = check_parent(r, fqcn_path)
            if err:
                raise ValueError(f"bad relay definitions: {err}")
            fqcn = ".".join(fqcn_path)
            r.add_prop(PropKey.FQCN, fqcn)
