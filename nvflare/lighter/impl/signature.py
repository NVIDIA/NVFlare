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

import os
import pickle

from nvflare.lighter.spec import Builder, Study
from nvflare.lighter.utils import sign_all


class SignatureBuilder(Builder):
    """Sign files with rootCA's private key.

    Creates signatures for all the files signed with the root CA for the startup kits so that they
    can be cryptographically verified to ensure any tampering is detected. This builder writes the signature.pkl file.
    """

    def build(self, study: Study, ctx: dict):
        server = study.get_participants_by_type("server")
        dest_dir = self.get_kit_dir(server, ctx)
        root_pri_key = ctx.get("root_pri_key")
        signatures = sign_all(dest_dir, root_pri_key)
        pickle.dump(signatures, open(os.path.join(dest_dir, "signature.pkl"), "wb"))
        for p in study.get_participants_by_type("client", first_only=False):
            dest_dir = self.get_kit_dir(p, ctx)
            root_pri_key = ctx.get("root_pri_key")
            signatures = sign_all(dest_dir, root_pri_key)
            pickle.dump(signatures, open(os.path.join(dest_dir, "signature.pkl"), "wb"))
