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


class AzureBuilder(Builder):
    def __init__(self):
        Builder.__init__(self)

    def initialize(self, project: Project, ctx: ProvisionContext):
        ctx.load_templates(["master_template.yml", "azure_template.yml"])

    def build(self, project: Project, ctx: ProvisionContext):
        # build server
        server = project.get_server()
        dest_dir = ctx.get_kit_dir(server)
        ctx.build_from_template(
            dest_dir=dest_dir,
            file_name=ProvFileName.AZURE_START_SH,
            temp_section=[
                TemplateSectionKey.CLOUD_SCRIPT_HEADER,
                TemplateSectionKey.AZURE_START_SVR_HEADER_SH,
                TemplateSectionKey.AZURE_START_COMMON_SH,
            ],
            exe=True,
        )

        for participant in project.get_clients():
            dest_dir = ctx.get_kit_dir(participant)
            ctx.build_from_template(
                dest_dir=dest_dir,
                file_name=ProvFileName.AZURE_START_SH,
                temp_section=[
                    TemplateSectionKey.CLOUD_SCRIPT_HEADER,
                    TemplateSectionKey.AZURE_START_CLN_HEADER_SH,
                    TemplateSectionKey.AZURE_START_COMMON_SH,
                ],
                exe=True,
            )
