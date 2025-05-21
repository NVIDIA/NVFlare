# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.lighter.constants import PropKey, ProvFileName, TemplateSectionKey
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Participant, Project
from nvflare.lighter.spec import Builder


class EdgeBuilder(Builder):

    def __init__(self):
        Builder.__init__(self)

    def initialize(self, project: Project, ctx: ProvisionContext):
        ctx.load_templates("edge_template.yml")

    def build(self, project: Project, ctx: ProvisionContext):
        for client in project.get_clients():
            self._build_client(client, ctx)

    def _build_client(self, client: Participant, ctx: ProvisionContext):
        is_leaf = client.get_prop(PropKey.IS_LEAF, True)
        if not is_leaf:
            return

        service_port = client.get_prop(PropKey.EDGE_SERVICE_PORT)
        if not service_port:
            ctx.error(f"missing {PropKey.EDGE_SERVICE_PORT} in client {client.name}")
            return

        lh = client.get_listening_host()
        if not lh:
            ctx.error(f"missing {PropKey.LISTENING_HOST} in client {client.name}")
            return

        replacement = {"host": "0.0.0.0", "port": service_port}

        dest_dir = ctx.get_local_dir(client)
        ctx.build_from_template(
            dest_dir=dest_dir,
            file_name=ProvFileName.EDGE_RESOURCES_JSON,
            temp_section=TemplateSectionKey.EDGE_LCP_RESOURCES,
            replacement=replacement,
        )
