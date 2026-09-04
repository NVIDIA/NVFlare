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
from nvflare.fuel.common.fqn import FQN


class FQCN(FQN):
    VALID_PATTERN = "^[A-Za-z0-9_.~-]*$"

    # A job's cells hang directly off the owning site: <owner>.<job_id> (the job cell) and its
    # descendants, or an auxiliary cell <owner>.<name>_<job_id> (e.g. the workspace-transfer
    # bootstrap cell). Nothing deeper counts: a cell under another job's cell is that job's.
    JOB_AUX_SEPARATOR = "_"

    @staticmethod
    def job_aux_name(name: str, job_id: str) -> str:
        return f"{name}{FQCN.JOB_AUX_SEPARATOR}{job_id}"

    @staticmethod
    def belongs_to_job(fqcn: str, job_id: str, owner_segments: int = 1) -> bool:
        """True if the segment right after the owner's prefix is the job id or a <name>_<job id> auxiliary name."""
        if not job_id:
            return False
        parts = FQCN.split(FQCN.normalize(fqcn))
        if len(parts) <= owner_segments:
            return False
        job_segment = parts[owner_segments]
        return job_segment == job_id or job_segment.endswith(FQCN.JOB_AUX_SEPARATOR + job_id)


# A network Attach trainer connects beneath the stable site CP and authenticates
# with that physical parent's provisioned identity.
CLIENT_API_ATTACH_LEAF_PREFIX = "-client_api_"


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
