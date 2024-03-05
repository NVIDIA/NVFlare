# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.spec import Participant, Project


def read_file(file_name: str):
    with open(file_name, "rb") as f:
        return f.read()


def generate_all_keys(key_path: str, server_name: str = "localhost", client_name: str = "xgboost_client"):
    parties = [
        Participant(type="server", name=server_name, org="self"),
        Participant(type="client", name=client_name, org="self"),
    ]
    for party in parties:
        os.makedirs(os.path.join(key_path, party.name, "startup"), exist_ok=True)
    p = Project(name="xgboost", description="xgboost native integration", participants=parties)
    ctx = {"wip_dir": key_path}
    builder = CertBuilder()
    builder.build(p, ctx)
