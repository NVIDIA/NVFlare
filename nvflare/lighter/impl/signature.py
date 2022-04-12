# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.lighter.spec import Builder, Project
from nvflare.lighter.utils import sign_all


class SignatureBuilder(Builder):
    """Sign files with rootCA's private key.

    Creates signatures for all the files signed with the root CA for the startup kits so that they
    can be cryptographically verified to ensure any tampering is detected. This builder writes the signature.pkl file.
    """

    def _do_sign(self, root_pri_key, dest_dir):
        signatures = sign_all(dest_dir, root_pri_key)
        json.dump(signatures, open(os.path.join(dest_dir, "signature.json"), "wt"))

    def build(self, project: Project, ctx: dict):
        root_pri_key = ctx.get("root_pri_key")

        overseer = project.get_participants_by_type("overseer", first_only=True)
        dest_dir = self.get_kit_dir(overseer, ctx)
        self._do_sign(root_pri_key, dest_dir)

        servers = project.get_participants_by_type("server", first_only=False)
        for server in servers:
            dest_dir = self.get_kit_dir(server, ctx)
            self._do_sign(root_pri_key, dest_dir)

        for p in project.get_participants_by_type("client", first_only=False):
            dest_dir = self.get_kit_dir(p, ctx)
            self._do_sign(root_pri_key, dest_dir)
