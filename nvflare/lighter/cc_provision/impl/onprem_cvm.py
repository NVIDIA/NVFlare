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

import json
import os
from pathlib import Path

from nvflare.lighter import utils
from nvflare.lighter.constants import PropKey
from nvflare.lighter.ctx import ProvisionContext
from nvflare.lighter.entity import Entity, Project
from nvflare.lighter.spec import Builder

from ..cc_constants import CC_AUTHORIZERS_KEY, CCAuthConfig

AUTHORIZERS_YAML_PATH = Path(__file__).parent.parent / "resources" / "onprem" / "cc_authorizers.yml"


class OnPremCVMBuilder(Builder):
    """Builder for On-premises Confidential Computing VMs."""

    def __init__(
        self,
        token_expiration=3600,
        authorizers_yaml_path: str = AUTHORIZERS_YAML_PATH,
    ):
        """Initialize the builder.

        Args:
            token_expiration: Token expiration time in seconds
            authorizers_yaml_path: Path to the authorizers yaml file
        """
        self.token_expiration = token_expiration
        self.authorizers_yaml_path = authorizers_yaml_path
        self.authorizers = []

    def initialize(self, project: Project, ctx: ProvisionContext):
        """Initialize the configurator.

        Args:
            project: The project to configure
            ctx: The provisioning context
        """
        print(f"Loading authorizers from {self.authorizers_yaml_path}")
        self.authorizers = utils.load_yaml(self.authorizers_yaml_path).get(CC_AUTHORIZERS_KEY, [])
        #  Add authorizers to ctx for cc manager
        authorizers = ctx.get(CC_AUTHORIZERS_KEY, [])
        for authorizer in self.authorizers:
            authorizers.append(
                {
                    CCAuthConfig.AUTHORIZER_ID: authorizer.get("id"),
                    CCAuthConfig.TOKEN_EXPIRATION: self.token_expiration,
                }
            )
        ctx[CC_AUTHORIZERS_KEY] = authorizers

    def build(self, project: Project, ctx: ProvisionContext):
        """Build CVM configuration for all participants."""
        server = project.get_server()
        if server and server.get_prop(PropKey.CC_ENABLED, False):
            self._build_resources(server, ctx)

        for client in project.get_clients():
            if client.get_prop(PropKey.CC_ENABLED, False):
                self._build_resources(client, ctx)

    def _build_resources(self, entity: Entity, ctx: ProvisionContext):
        """Build resources for the entity."""
        # Write authorizers to local resources
        dest_dir = ctx.get_local_dir(entity)
        for authorizer in self.authorizers:
            utils.write(
                os.path.join(dest_dir, f"{authorizer['id']}__p_resources.json"),
                json.dumps({"components": [authorizer]}, indent=2),
                "t",
            )
