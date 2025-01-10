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

import pytest

from nvflare.lighter.entity import Participant, Project


def create_participants(type, number, org, name, props=None):
    p_list = list()
    for i in range(number):
        name = f"{name[:2]}{i}{name[2:]}"
        p_list.append(Participant(name=name, org=org, type=type, props=props))
    return p_list


class TestProject:
    def test_single_server(self):
        p1 = Participant(name="server1", org="org", type="server")
        p2 = Participant(name="server2", org="org", type="server")
        with pytest.raises(ValueError, match=r".* already has a server defined"):
            _ = Project("name", "description", [p1, p2])

    def test_single_overseer(self):
        p1 = Participant(name="name1", org="org", type="overseer")
        p2 = Participant(name="name2", org="org", type="overseer")
        with pytest.raises(ValueError, match=r".* already has an overseer defined"):
            _ = Project("name", "description", [p1, p2])

    def test_get_clients(self):
        p = create_participants(type="client", number=3, org="org", name="name")
        prj = Project("name", "description", p)
        c = prj.get_clients()
        assert len(c) == len(p)
        assert all(c[i].name == p[i].name and c[i].org == p[i].org for i in range(len(p)))

    def test_get_admins(self):
        p = create_participants(
            type="admin", number=3, org="org", name="admin@nvidia.com", props={"role": "project_admin"}
        )
        prj = Project("name", "description", p)
        c = prj.get_admins()
        assert len(c) == len(p)
        assert all(c[i].name == p[i].name and c[i].org == p[i].org for i in range(len(p)))

    def test_admin_role_required(self):
        p = create_participants(type="admin", number=3, org="org", name="admin@nvidia.com")
        with pytest.raises(ValueError, match=r"missing role *."):
            _ = Project("name", "description", p)

    def test_bad_admin_role(self):
        with pytest.raises(ValueError, match=r"bad value for role *."):
            _ = create_participants(
                type="admin", number=3, org="org", name="admin@nvidia.com", props={"role": "invalid"}
            )

    @pytest.mark.parametrize(
        "type1,type2",
        [
            ("client", "client"),
            ("server", "client"),
            ("admin", "admin"),
        ],
    )
    def test_dup_names(self, type1, type2):
        if type1 == "admin":
            name = "name@xyz.com"
            props = {"role": "project_admin"}
        else:
            name = "name"
            props = None

        p1 = Participant(name=name, org="org", type=type1, props=props)
        p2 = Participant(name=name, org="org", type=type2, props=props)
        with pytest.raises(ValueError, match=r".* already has a participant with the name *."):
            _ = Project("name", "description", [p1, p2])
