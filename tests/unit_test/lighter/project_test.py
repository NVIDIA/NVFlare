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

from nvflare.lighter.spec import Participant, Project


def create_participants(type, number, org, name):
    p_list = list()
    for i in range(number):
        name = f"{name[:2]}{i}{name[2:]}"
        p_list.append(Participant(name=name, org=org, type=type))
    return p_list


class TestProject:
    def test_invalid_project(self):
        p1 = create_participants("server", 3, "org", "server")
        p2 = create_participants("server", 3, "org", "server")
        p = p1 + p2
        with pytest.raises(ValueError, match=r".* se0rver .*"):
            _ = Project("name", "description", p)

    @pytest.mark.parametrize(
        "p_type,name",
        [("server", "server"), ("client", "client"), ("admin", "admin@abc.com"), ("overseer", "overseer")],
    )
    def test_get_participants_by_type(self, p_type, name):
        p = create_participants(type=p_type, number=3, org="org", name=name)
        prj = Project("name", "description", p)
        assert prj.get_participants_by_type(p_type) == p[0]
        assert prj.get_participants_by_type(p_type, first_only=False) == p
