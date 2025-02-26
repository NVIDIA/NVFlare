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
from nvflare.lighter.constants import ProvFileName, TemplateSectionKey
from nvflare.lighter.spec import Builder, Project, ProvisionContext


class AWSBuilder(Builder):
    def __init__(self):
        Builder.__init__(self)

    def initialize(self, project: Project, ctx: ProvisionContext):
        ctx.load_templates(["master_template.yml", "aws_template.yml"])

    def build(self, project: Project, ctx: ProvisionContext):
        # build server
        server = project.get_server()
        dest_dir = ctx.get_kit_dir(server)
        replacement = {
            "type": "server",
            "inbound_rule": "aws ec2 authorize-security-group-ingress --region ${REGION} --group-id $sg_id --protocol tcp --port 8002-8003 --cidr 0.0.0.0/0 >> ${LOGFILE}.sec_grp.log",
            "cln_uid": "",
            "server_name": server.name,
            "ORG": server.org,
        }
        ctx.build_from_template(
            dest_dir=dest_dir,
            file_name=ProvFileName.AWS_START_SH,
            temp_section=[
                TemplateSectionKey.CLOUD_SCRIPT_HEADER,
                TemplateSectionKey.AWS_START_SH,
            ],
            replacement=replacement,
            exe=True,
        )

        for client in project.get_clients():
            dest_dir = ctx.get_kit_dir(client)
            replacement = {
                "type": "client",
                "inbound_rule": "",
                "cln_uid": f"uid={client.name}",
                "ORG": client.org,
            }

            ctx.build_from_template(
                dest_dir=dest_dir,
                file_name=ProvFileName.AWS_START_SH,
                temp_section=[
                    TemplateSectionKey.CLOUD_SCRIPT_HEADER,
                    TemplateSectionKey.AWS_START_SH,
                ],
                replacement=replacement,
                exe=True,
            )
