# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Acceptance-only verifier configuration, installed before kit signing."""

import json
import os
import re
import tempfile
from pathlib import Path

from nvflare.app_common.default_component_policy import DEFAULT_CLASS_ALLOW_LIST
from nvflare.lighter.cc_provision.impl.coco import AUTHOR_PATH, MANAGER_PATH, CoCoBuilder
from nvflare.lighter.constants import ProvFileName
from nvflare.lighter.spec import Builder

APPLICATION_CLASSES = ("tdx_acceptance.AcceptanceController", "tdx_acceptance.AcceptanceExecutor")


class ObserverBuilder(Builder):
    @staticmethod
    def _allow_baked_application(resources_file):
        resources = json.loads(resources_file.read_text())
        allowed = resources.get("class_allow_list", list(DEFAULT_CLASS_ALLOW_LIST))
        if not isinstance(allowed, list) or any(
            not isinstance(value, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", value)
            for value in allowed
        ):
            raise ValueError("Acceptance requires an explicit component allow-list")
        if resources.get("class_list_enforcement_mode", "enforce") != "enforce":
            raise ValueError("Acceptance requires enforced component authorization")
        resources["class_allow_list"] = allowed + [name for name in APPLICATION_CLASSES if name not in allowed]
        descriptor, temporary = tempfile.mkstemp(dir=resources_file.parent, prefix=".acceptance-resources-")
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(resources, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, resources_file)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def build(self, project, ctx):
        observer = next(c for c in project.get_clients() if c.name == "site-observer")
        server = project.get_server()
        # The server always has a verifier, including topology A. Copy only its
        # public verifier settings, never issuer credentials or startup files.
        source = Path(ctx.get_local_dir(server))
        # Topology A's ordinary server has no protected CC configuration to
        # extend its component policy. Authorize only the reviewed baked classes
        # during provisioning, before SignatureBuilder runs for protected kits.
        self._allow_baked_application(source / ProvFileName.RESOURCES_JSON_DEFAULT)
        authorizer = json.loads((source / "coco_authorizer__p_resources.json").read_text())["components"][0]["args"]
        manager = json.loads((source / "cc_manager__p_resources.json").read_text())["components"][0]["args"]
        authorizer = {
            k: v for k, v in authorizer.items() if k not in {"site_name", "token_url"} and not k.startswith("retry_")
        }
        manager["cc_issuers_conf"] = []
        CoCoBuilder._write(ctx, observer, "coco_authorizer", AUTHOR_PATH, authorizer)
        CoCoBuilder._write(ctx, observer, "cc_manager", MANAGER_PATH, manager)
