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

from .template_engine import TemplateEngine


class Template:
    def __init__(self, template=None):
        self._engine = TemplateEngine()
        self.supported_csps = ("azure", "aws")

    def _render(self, sections, context=None):
        """Render one or more template sections with the given context."""
        if isinstance(sections, str):
            sections = [sections]
        result = ""
        for section in sections:
            result += self._engine.render(f"{section}.j2", context or {})
        return result

    def get_cloud_script_header(self):
        return self._render("cloud_script_header")

    def get_azure_server_start_sh(self, entity):
        sections = [
            "cloud_script_header",
            "azure_start_svr_header_sh",
            "azure_start_common_sh",
        ]
        return self._render(
            sections,
            {
                "type": "server",
                "docker_network": "--network host",
                "cln_uid": "",
                "server_name": entity.name,
                "ORG": "",
            },
        )

    def get_aws_server_start_sh(self, entity):
        sections = ["cloud_script_header", "aws_start_sh"]
        return self._render(
            sections,
            {
                "type": "server",
                "inbound_rule": "aws ec2 authorize-security-group-ingress --region ${REGION} --group-id $sg_id --protocol tcp --port 8002-8003 --cidr 0.0.0.0/0 >> ${LOGFILE}.sec_grp.log",
                "cln_uid": "",
                "server_name": entity.name,
                "ORG": "",
            },
        )

    def get_azure_client_start_sh(self, entity):
        sections = [
            "cloud_script_header",
            "azure_start_cln_header_sh",
            "azure_start_common_sh",
        ]
        return self._render(
            sections,
            {
                "type": "client",
                "docker_network": "",
                "cln_uid": f"uid={entity.name}",
                "ORG": entity.org,
            },
        )

    def get_aws_client_start_sh(self, entity):
        sections = ["cloud_script_header", "aws_start_sh"]
        return self._render(
            sections,
            {
                "type": "client",
                "inbound_rule": "",
                "cln_uid": f"uid={entity.name}",
                "ORG": entity.org,
            },
        )

    def get_azure_start_svr_header_sh(self):
        return self._render("azure_start_svr_header_sh")

    def get_azure_start_cln_header_sh(self):
        return self._render("azure_start_cln_header_sh")

    def get_azure_start_common_sh(self):
        return self._render("azure_start_common_sh")

    def get_sub_start_sh(self):
        return self._render("sub_start_sh")

    def get_azure_svr_sh(self):
        sections = [
            "cloud_script_header",
            "azure_start_svr_header_sh",
            "azure_start_common_sh",
        ]
        return self._render(sections)

    def get_azure_cln_sh(self):
        sections = [
            "cloud_script_header",
            "azure_start_cln_header_sh",
            "azure_start_common_sh",
        ]
        return self._render(sections)

    def get_start_sh(self, csp, type, entity):
        try:
            func = getattr(self, f"get_{csp}_{type}_start_sh")
            return func(entity)
        except AttributeError:
            return ""
