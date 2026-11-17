# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import os

import yaml

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.job_def import JobMetaKey

CONTRACT = """\
apiVersion: v3.1.0
kind: DataContract
id: omop
version: 1.0.0
status: active
contractCreatedTs: 2024-01-01  # a plain YAML date, kept as the steward wrote it
schema:
- name: person
  properties:
  - {name: person_id, physicalType: bigint}
  - {name: year_of_birth, physicalType: integer}
"""


class FakeWorkspace:
    def __init__(self, root_dir):
        self.root_dir = str(root_dir)

    def get_site_config_dir(self):
        return os.path.join(self.root_dir, "local")

    def get_run_dir(self, job_id):
        return os.path.join(self.root_dir, job_id)


class FakeFLContext:
    def __init__(self, root_dir, site="site-1", study="oncology", simulate=False, job_id="job-1"):
        self.workspace = FakeWorkspace(root_dir)
        self.site = site
        self.job_id = job_id
        self.props = {FLContextKey.JOB_META: {JobMetaKey.STUDY.value: study}, FLContextKey.SIMULATE_MODE: simulate}

    def get_identity_name(self):
        return self.site

    def get_job_id(self):
        return self.job_id

    def get_workspace(self):
        return self.workspace

    def get_prop(self, key, default=None):
        return self.props.get(key, default)


def write_site_config(config_dir, datasets, study="oncology", contract=CONTRACT):
    """Write catalog.yaml declaring ``datasets`` for ``study``, and the contract file the provided ones point at."""
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "omop.odcs.yaml"), "w") as f:
        f.write(contract)
    with open(os.path.join(config_dir, "catalog.yaml"), "w") as f:
        yaml.safe_dump({"studies": {study: {"datasets": datasets}}}, f)
    return str(config_dir)
