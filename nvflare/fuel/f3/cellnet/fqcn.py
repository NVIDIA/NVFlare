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
from typing import Optional, Tuple

from nvflare.fuel.common.fqn import FQN


class FQCN(FQN):
    pass


# CellPipe cells connected through another cell (e.g. a relay) use an alias
# leaf segment of the form <owner>_<runtime_id>_<mode>, built by
# make_cell_pipe_alias and parsed by parse_cell_pipe_alias. The alias maps the
# cell to the owning site for mTLS identity resolution and stream message
# authentication. Both directions of the grammar live here so they cannot
# drift apart.
CELL_PIPE_ALIAS_MODES = ("active", "passive")


def make_cell_pipe_alias(owner: str, runtime_id: str, mode: str) -> str:
    return f"{owner}_{runtime_id}_{mode}"


def parse_cell_pipe_alias(segment: str) -> Optional[Tuple[str, str, str]]:
    """Parse a CellPipe alias leaf segment into (owner, runtime_id, mode).

    Only the constrained form <owner>_<runtime_id>_(active|passive) with a
    non-empty runtime_id that contains no "." or "_" is a valid alias: parsing
    from the right makes the interpretation unambiguous, so
    "site-a_x_<uuid>_active" can only belong to "site-a_x", never to "site-a"
    with a runtime id of "x_<uuid>".

    Returns None if the segment is not a valid alias.
    """
    head, sep, mode = segment.rpartition("_")
    if not sep or mode not in CELL_PIPE_ALIAS_MODES:
        return None

    # rpartition splits on the last "_", so runtime_id can never contain "_";
    # only the "." constraint needs an explicit check.
    owner, sep, runtime_id = head.rpartition("_")
    if not sep or not owner or not runtime_id or "." in runtime_id:
        return None

    return owner, runtime_id, mode


class FqcnInfo:
    def __init__(self, fqcn: str):
        self.fqcn = fqcn
        self.path = FQCN.split(fqcn)
        self.gen = len(self.path)
        self.is_root = self.gen == 1
        self.root = self.path[0]
        self.is_on_server = self.root == FQCN.ROOT_SERVER


def same_family(info1: FqcnInfo, info2: FqcnInfo):
    return info1.root == info2.root
