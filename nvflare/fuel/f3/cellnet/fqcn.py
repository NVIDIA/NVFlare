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


# CellPipe cells use explicitly marked leaf segments so pipe names can never
# be confused with other cell names or with each other:
#   - plain leaf "cellpipe-<token>_<mode>" for pipes connected to the server
#     root or to the site's own CP;
#   - alias leaf "cellpipe-alias-<owner>_<token>_<mode>" for pipes connected
#     through another cell (e.g. a relay). The alias maps the cell to the
#     owning site for mTLS identity resolution and stream message
#     authentication. Both directions of the alias grammar live here so they
#     cannot drift apart.
#
# CellPipe cell-name schemes, in historical order:
#   1. flat (pre-2.7): the whole FQCN is "<site>_<token>_<mode>", a root-level
#      sibling of the site cell.
#   2. hierarchical (#4801, never released): "<site>.<token>.<mode>". Replaced
#      because the extra segments created unconnected FQCN parents that broke
#      routing (NVBug 6371056).
#   3. topology (current): a single prefixed leaf segment under the FQCN of
#      the cell the pipe actually connects to,
#      "<parent>.cellpipe-<token>_<mode>", or
#      "<relay_fqcn>.cellpipe-alias-<owner>_<token>_<mode>" when connected
#      through another cell.
# Mixed-version notes: scheme-1 aliases are still accepted by identity
# resolution and stream auth (as whole-FQCN aliases via the bare grammar),
# but the two ends of one pipe pair must run the same scheme - each end
# derives the peer's name from its own code, so a CJ and a training
# subprocess on different schemes fail with "peer FQCN mismatch".
CELL_PIPE_LEAF_PREFIX = "cellpipe-"
CELL_PIPE_ALIAS_PREFIX = "cellpipe-alias-"
CELL_PIPE_ALIAS_MODES = ("active", "passive")


def make_cell_pipe_alias(owner: str, runtime_id: str, mode: str) -> str:
    return f"{CELL_PIPE_ALIAS_PREFIX}{owner}_{runtime_id}_{mode}"


def parse_cell_pipe_alias(segment: str) -> Optional[Tuple[str, str, str]]:
    """Parse a CellPipe alias leaf segment into (owner, runtime_id, mode).

    Two shapes are accepted:
      - the current explicit form "cellpipe-alias-<owner>_<runtime_id>_<mode>";
      - the bare legacy form "<owner>_<runtime_id>_<mode>" used by pre-2.8
        flat CellPipe names, where the whole FQCN is the alias. Callers decide
        where the bare form is acceptable; it is normally restricted to
        single-segment FQCNs so an unmarked "<token>_<mode>" leaf inside a
        longer FQCN is never misread as an alias.

    In both shapes the runtime_id must be non-empty and contain no "." or
    "_": parsing from the right makes the interpretation unambiguous, so
    "site-a_x_<uuid>_active" can only belong to "site-a_x", never to "site-a"
    with a runtime id of "x_<uuid>".

    Returns None if the segment is not a valid alias.
    """
    if segment.startswith(CELL_PIPE_ALIAS_PREFIX):
        segment = segment[len(CELL_PIPE_ALIAS_PREFIX) :]

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
