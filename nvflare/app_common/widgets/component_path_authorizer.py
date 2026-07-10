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

import json
import os
import threading
from typing import Optional

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, SystemConfigs
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.fuel.sec.audit import AuditService
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.widgets.widget import Widget

CLASS_ALLOW_LIST = "class_allow_list"
CLASS_LIST_ENFORCEMENT_MODE = "class_list_enforcement_mode"
ENFORCEMENT_MODE_ENFORCE = "enforce"
ENFORCEMENT_MODE_WARN = "warn"
ALLOW_ALL = "*"

_VALID_ENFORCEMENT_MODES = (ENFORCEMENT_MODE_ENFORCE, ENFORCEMENT_MODE_WARN)
_ALLOW_ALL_AUDIT_ACTION = "component_authorization.class_allow_list_disabled"


class ComponentPathAuthorizer(Widget):
    def __init__(self):
        """Allows component builds by path prefixes configured in site resources."""
        super().__init__()
        self._allow_list_cache = {}
        self._allow_list_cache_lock = threading.Lock()
        self._allow_all_audit_keys = set()
        self._allow_all_audit_lock = threading.Lock()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type != EventType.BEFORE_BUILD_COMPONENT:
            return

        if self._job_has_byoc(fl_ctx):
            return

        component_config = fl_ctx.get_prop(FLContextKey.COMPONENT_CONFIG)
        node = fl_ctx.get_prop(FLContextKey.COMPONENT_NODE)
        self.authorize_component_config(component_config, node, fl_ctx=fl_ctx)

    def authorize_component_config(
        self, component_config, node=None, fl_ctx: Optional[FLContext] = None, workspace=None
    ):
        if self._job_has_byoc(fl_ctx):
            return

        component_path = self._get_component_path(component_config)
        if component_path is None:
            return

        allow_list, enforcement_mode = self._get_policy(fl_ctx=fl_ctx, workspace=workspace)
        if ALLOW_ALL in allow_list:
            self._audit_allow_all(fl_ctx=fl_ctx, workspace=workspace)
            return

        if not any(self._path_matches_prefix(component_path, prefix) for prefix in allow_list):
            node_path = node.path() if node else ""
            message = f"Component '{component_path}' at config path '{node_path}' is not in allow_list"
            if enforcement_mode == ENFORCEMENT_MODE_WARN:
                self.logger.warning(
                    f"{message}; allowing it because {CLASS_LIST_ENFORCEMENT_MODE} is '{ENFORCEMENT_MODE_WARN}'"
                )
                return
            raise UnsafeComponentError(message)

    def _audit_allow_all(self, fl_ctx: Optional[FLContext] = None, workspace=None):
        resources_file = self._get_resources_file_path(fl_ctx=fl_ctx, workspace=workspace)
        source = os.path.abspath(resources_file) if resources_file else SystemConfigs.RESOURCES_CONF
        job_id = fl_ctx.get_job_id() if fl_ctx else ""
        audit_key = (job_id, source)
        message = (
            f"{CLASS_ALLOW_LIST} contains '{ALLOW_ALL}'; all component classes are allowed and "
            "the remaining allow-list entries are ignored"
        )

        with self._allow_all_audit_lock:
            if audit_key in self._allow_all_audit_keys:
                return
            event_id = AuditService.add_event(
                user=fl_ctx.get_identity_name() if fl_ctx and fl_ctx.get_identity_name() else "system",
                action=_ALLOW_ALL_AUDIT_ACTION,
                ref=job_id or "",
                msg=message,
            )
            if event_id:
                self._allow_all_audit_keys.add(audit_key)

        self.logger.warning(message)

    def _get_allow_list(self, fl_ctx: Optional[FLContext] = None, workspace=None):
        allow_list, _ = self._get_policy(fl_ctx=fl_ctx, workspace=workspace)
        return allow_list

    def _get_policy(self, fl_ctx: Optional[FLContext] = None, workspace=None):
        resources_file = self._get_resources_file_path(fl_ctx=fl_ctx, workspace=workspace)
        if resources_file:
            return self._get_policy_from_file(resources_file)

        resources = ConfigService.get_section(SystemConfigs.RESOURCES_CONF)
        return self._get_policy_from_resources(resources)

    def _get_allow_list_from_file(self, resources_file):
        allow_list, _ = self._get_policy_from_file(resources_file)
        return allow_list

    def _get_policy_from_file(self, resources_file):
        cache_key = os.path.abspath(resources_file)

        with self._allow_list_cache_lock:
            stat_result = os.stat(cache_key)
            cache_signature = self._make_file_signature(stat_result)
            cached = self._allow_list_cache.get(cache_key)
            if cached and cached[0] == cache_signature:
                return cached[1], cached[2]

            with open(cache_key, "rt") as f:
                resources = json.load(f)
                cache_signature = self._make_file_signature(os.fstat(f.fileno()))
            allow_list, enforcement_mode = self._get_policy_from_resources(resources)

            self._allow_list_cache[cache_key] = (cache_signature, allow_list, enforcement_mode)

        return allow_list, enforcement_mode

    @staticmethod
    def _make_file_signature(stat_result):
        return (stat_result.st_mtime_ns, stat_result.st_size, stat_result.st_ino, stat_result.st_dev)

    @classmethod
    def _get_allow_list_from_resources(cls, resources):
        allow_list, _ = cls._get_policy_from_resources(resources)
        return allow_list

    @classmethod
    def _get_policy_from_resources(cls, resources):
        if not isinstance(resources, dict) or CLASS_ALLOW_LIST not in resources:
            raise UnsafeComponentError(
                f"{CLASS_ALLOW_LIST} is not configured in resources.json or resources.json.default. "
                f"Non-BYOC jobs require a top-level {CLASS_ALLOW_LIST}; add allowed class path prefixes "
                'such as "nvflare." to site resources, or enable BYOC for jobs that load custom code.'
            )

        allow_list = resources.get(CLASS_ALLOW_LIST)
        if not isinstance(allow_list, list):
            raise UnsafeComponentError(f"{CLASS_ALLOW_LIST} must be list but got {type(allow_list)}")

        enforcement_mode = resources.get(CLASS_LIST_ENFORCEMENT_MODE, ENFORCEMENT_MODE_ENFORCE)
        if not isinstance(enforcement_mode, str):
            raise UnsafeComponentError(f"{CLASS_LIST_ENFORCEMENT_MODE} must be str but got {type(enforcement_mode)}")
        if enforcement_mode not in _VALID_ENFORCEMENT_MODES:
            raise UnsafeComponentError(
                f"{CLASS_LIST_ENFORCEMENT_MODE} must be one of {_VALID_ENFORCEMENT_MODES} but got '{enforcement_mode}'"
            )
        try:
            return cls._normalize_allow_list(allow_list), enforcement_mode
        except (TypeError, ValueError) as ex:
            raise UnsafeComponentError(str(ex))

    @staticmethod
    def _get_resources_file_path(fl_ctx: Optional[FLContext] = None, workspace=None):
        if fl_ctx and workspace is None:
            workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        if workspace and hasattr(workspace, "get_resources_file_path"):
            resources_file = workspace.get_resources_file_path()
            if resources_file and os.path.exists(resources_file):
                return resources_file

        return None

    @staticmethod
    def _job_has_byoc(fl_ctx: Optional[FLContext]):
        if not fl_ctx:
            return False

        job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        if not isinstance(job_meta, dict):
            return False
        return bool(job_meta.get(AppValidationKey.BYOC))

    @classmethod
    def _normalize_allow_list(cls, allow_list):
        if ALLOW_ALL in allow_list:
            return [ALLOW_ALL]

        result = []
        for prefix in allow_list:
            prefix = cls._validate_prefix(prefix)
            if prefix not in result:
                result.append(prefix)
        return result

    @staticmethod
    def _validate_prefix(prefix: str):
        if not isinstance(prefix, str):
            raise TypeError(f"allow_list entries must be str but got {type(prefix)}")
        if not prefix:
            raise ValueError("allow_list entries must not be empty")
        if prefix != prefix.strip():
            raise ValueError(f"allow_list entry '{prefix}' must not contain leading or trailing whitespace")

        if prefix.endswith("."):
            prefix_body = prefix[:-1]
            if not prefix_body:
                raise ValueError("allow_list entries must not be empty")
        else:
            prefix_body = prefix
            if "." not in prefix_body:
                raise ValueError(
                    f"allow_list entry '{prefix}' must end with '.' for package prefixes "
                    "or be a fully qualified dotted path"
                )

        if any(not part for part in prefix_body.split(".")):
            raise ValueError(f"allow_list entry '{prefix}' is not a valid dotted path prefix")

        return prefix

    @staticmethod
    def _path_matches_prefix(component_path: str, prefix: str):
        if component_path == prefix:
            return True
        if prefix.endswith("."):
            return component_path.startswith(prefix)
        return component_path.startswith(f"{prefix}.")

    @staticmethod
    def _get_component_path(component_config):
        if not isinstance(component_config, dict):
            raise UnsafeComponentError(f"Component config must be dict but got {type(component_config)}")

        if "name" in component_config:
            raise UnsafeComponentError("Component config must use path or class_path; name is not allowed")

        if "path" in component_config:
            component_path = component_config["path"]
            key = "path"
        elif "class_path" in component_config:
            component_path = component_config["class_path"]
            key = "class_path"
        else:
            raise UnsafeComponentError("Component config must specify path or class_path")

        if not isinstance(component_path, str):
            raise UnsafeComponentError(f"Component {key} must be str but got {type(component_path)}")
        if not component_path:
            raise UnsafeComponentError(f"Component {key} must not be empty")

        return component_path
