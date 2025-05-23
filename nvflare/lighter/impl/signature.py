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

import json
import os

from nvflare.lighter.constants import CtxKey, ProvFileName
from nvflare.lighter.spec import Builder, Project, ProvisionContext
from nvflare.lighter.utils import sign_all


class SignatureBuilder(Builder):
    """Sign files with rootCA's private key.

    Creates signatures for all the files signed with the root CA for the startup kits so that they
    can be cryptographically verified to ensure any tampering is detected. This builder writes the signature.json file.
    """

    @staticmethod
    def _do_sign(root_pri_key, dest_dir):
        signatures = sign_all(dest_dir, root_pri_key)
        with open(os.path.join(dest_dir, ProvFileName.SIGNATURE_JSON), "wt") as f:
            json.dump(signatures, f)

    def build(self, project: Project, ctx: ProvisionContext):
        root_pri_key = ctx.get(CtxKey.ROOT_PRI_KEY)
        if not root_pri_key:
            raise RuntimeError(f"missing {CtxKey.ROOT_PRI_KEY} in ProvisionContext")

        for p in project.get_all_participants():
            dest_dir = ctx.get_kit_dir(p)
            self._do_sign(root_pri_key, dest_dir)
