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

    # A job's cells are the job cell (<site>.<job_id>) and its descendants, plus auxiliary cells
    # named <name>_<job_id> directly under the site (e.g. the workspace-transfer bootstrap cell).
    JOB_AUX_SEPARATOR = "_"

    @staticmethod
    def job_aux_name(name: str, job_id: str) -> str:
        return f"{name}{FQCN.JOB_AUX_SEPARATOR}{job_id}"

    @staticmethod
    def belongs_to_job(fqcn: str, job_id: str) -> bool:
        if not job_id:
            return False
        aux_suffix = FQCN.JOB_AUX_SEPARATOR + job_id
        return any(seg == job_id or seg.endswith(aux_suffix) for seg in FQCN.split(FQCN.normalize(fqcn)))


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
