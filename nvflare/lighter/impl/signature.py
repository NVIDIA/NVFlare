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

    signature.json is generated for:
    - Azure Confidential Computing kits: the full workspace is signed for startup integrity.
    - HE (Homomorphic Encryption) kits: startup + local dirs are signed to protect the shared
      TenSEAL context.

    Signing runs in ``finalize()``, not ``build()``. Other builders create files while finalizing,
    notably ``local/comm_config.json`` from StaticFileBuilder, and verification rejects any file
    that has no signature entry. Signing during ``build()`` therefore left a freshly provisioned
    kit unable to pass its own startup integrity check.

    Builders finalize in reverse order, so :func:`order_builders_for_signing` places this builder
    immediately after WorkspaceBuilder: late enough to follow every other builder's ``finalize()``,
    early enough to precede the workspace relocation WorkspaceBuilder performs.

    CVM vault workspaces are signed separately by :class:`VaultSignatureBuilder`, which already ran
    after finalization. Plain non-CC, non-HE kits do not receive signature.json. mTLS is the trust
    anchor for those deployments.
    """

    def finalize(self, project: Project, ctx: ProvisionContext):
        root_pri_key = ctx.get(CtxKey.ROOT_PRI_KEY)
        if not root_pri_key:
            raise RuntimeError(f"missing {CtxKey.ROOT_PRI_KEY} in ProvisionContext")

        for p in project.get_all_participants():
            if p.get_prop(PropKey.CC_ENABLED):
                sign_folders(ctx.get_ws_dir(p), root_pri_key, signature_file=ProvFileName.SIGNATURE_JSON)
            else:
                kit_dir = ctx.get_kit_dir(p)
                he_present = os.path.exists(
                    os.path.join(kit_dir, ProvFileName.SERVER_CONTEXT_TENSEAL)
                ) or os.path.exists(os.path.join(kit_dir, ProvFileName.CLIENT_CONTEXT_TENSEAL))
                if he_present:
                    # HE mode: sign startup and local to protect the shared TenSEAL context.
                    # load_tenseal_context_from_workspace requires LoadResult.OK in secure mode.
                    sign_folders(kit_dir, root_pri_key, signature_file=ProvFileName.SIGNATURE_JSON)
                    sign_folders(ctx.get_local_dir(p), root_pri_key, signature_file=ProvFileName.SIGNATURE_JSON)


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


def order_builders_for_signing(builders):
    """Return the builder list with signature builders positioned to finalize last.

    Finalization runs in reverse builder order and WorkspaceBuilder.finalize() relocates the
    workspace out of the work-in-progress directory. A signature builder must therefore sit
    immediately after WorkspaceBuilder so it signs once every other builder has finalized and
    while the workspace is still in place. Lists that do not start with WorkspaceBuilder are
    returned unchanged, because there is no safe position to move to.
    """
    from nvflare.lighter.impl.workspace import WorkspaceBuilder

    if not builders or not isinstance(builders[0], WorkspaceBuilder):
        return builders
    signers = [b for b in builders if isinstance(b, SignatureBuilder)]
    if not signers:
        return builders
    others = [b for b in builders if not isinstance(b, SignatureBuilder)]
    return others[:1] + signers + others[1:]
