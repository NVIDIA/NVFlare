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
from typing import List

from nvflare.apis.client import Client
from nvflare.fuel.utils.tree_utils import build_forest


def build_client_hierarchy(clients: List[Client]):
    """Build client hierarchy from a list of clients based on their FQSNs.

    Args:
        clients: clients of the hierarchy

    Returns: a Forest object that represents the client hierarchy.

    Note that the client hierarchy is a Forest (not a tree) since it could have multiple root clients.

    """
    return build_forest(
        objs=clients,
        get_fqn_f=lambda c: c.get_fqsn(),
        get_name_f=lambda c: c.name,
    )
