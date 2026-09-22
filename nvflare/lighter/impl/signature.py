# Copyright (c) 2021-2026, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.lighter.constants import CtxKey, PropKey, ProvFileName
from nvflare.lighter.spec import Builder, Project, ProvisionContext
from nvflare.lighter.utils import sign_folders


class SignatureBuilder(Builder):
    """Sign files with rootCA's private key.

    Creates signatures for all the files signed with the root CA for the startup kits so that they
    can be cryptographically verified to ensure any tampering is detected. This builder writes the
    signature.json file.

    signature.json is generated for HE (Homomorphic Encryption) kits: startup + local dirs are
    signed to protect the shared TenSEAL context. CVM vault workspaces are signed separately by
    :class:`VaultSignatureBuilder` after provisioning finalization.

    Plain non-HE kits do not receive signature.json. mTLS is the trust anchor for those deployments.
    Absence of signature.json is the correct and expected state for centrally provisioned standard
    kits and for kits assembled via the Manual Workflow (nvflare package).
    """

    def build(self, project: Project, ctx: ProvisionContext):
        root_pri_key = ctx.get(CtxKey.ROOT_PRI_KEY)
        if not root_pri_key:
            raise RuntimeError(f"missing {CtxKey.ROOT_PRI_KEY} in ProvisionContext")

        for p in project.get_all_participants():
            kit_dir = ctx.get_kit_dir(p)
            he_present = os.path.exists(os.path.join(kit_dir, ProvFileName.SERVER_CONTEXT_TENSEAL)) or os.path.exists(
                os.path.join(kit_dir, ProvFileName.CLIENT_CONTEXT_TENSEAL)
            )
            if he_present:
                # HE mode: sign startup and local to protect the shared TenSEAL context.
                # load_tenseal_context_from_workspace requires LoadResult.OK in secure mode.
                sign_folders(kit_dir, root_pri_key, signature_file=ProvFileName.SIGNATURE_JSON)
                sign_folders(ctx.get_local_dir(p), root_pri_key, signature_file=ProvFileName.SIGNATURE_JSON)
            # Plain non-HE kits are not signed here. CVM vault kits are signed after finalization.


class VaultSignatureBuilder(Builder):
    """Sign selected vault workspaces after config finalization, before relocation.

    Insert immediately after WorkspaceBuilder: reverse finalization then signs
    files such as comm_config.json that other builders create in finalize().
    """

    def finalize(self, project: Project, ctx: ProvisionContext):
        root_pri_key = ctx.get(CtxKey.ROOT_PRI_KEY)
        if not root_pri_key:
            raise RuntimeError(f"missing {CtxKey.ROOT_PRI_KEY} in ProvisionContext")
        for participant in project.get_all_participants():
            if participant.get_prop(PropKey.CVM_VAULT):
                sign_folders(ctx.get_ws_dir(participant), root_pri_key, signature_file=ProvFileName.SIGNATURE_JSON)
