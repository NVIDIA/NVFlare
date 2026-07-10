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
import re
import threading
from enum import Enum
from typing import Optional

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, SystemConfigs
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.app_common.default_component_policy import DEFAULT_CLASS_ALLOW_LIST
from nvflare.fuel.sec.audit import AuditService
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.widgets.widget import Widget

CLASS_ALLOW_LIST = "class_allow_list"
CLASS_LIST_ENFORCEMENT_MODE = "class_list_enforcement_mode"
ALLOW_ALL = "*"

_ALLOW_ALL_AUDIT_ACTION = "component_authorization.class_allow_list_disabled"
_WARN_MODE_AUDIT_ACTION = "component_authorization.unlisted_class_allowed"
_COMPONENT_PATH_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")


class ClassListEnforcementMode(str, Enum):
    ENFORCE = "enforce"
    WARN = "warn"


class ComponentPathAuthorizer(Widget):
    def __init__(self):
        """Allows component builds by path prefixes configured in site resources."""
        super().__init__()
        self._allow_list_cache = {}
        self._allow_list_cache_lock = threading.Lock()
        self._successful_audit_keys = set()
        self._warning_keys = set()
        self._audit_state_lock = threading.Lock()
        self._warned_default_allow_list = False

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

        allow_list, enforcement_mode, policy_source = self._get_policy(fl_ctx=fl_ctx, workspace=workspace)
        if ALLOW_ALL in allow_list:
            self._audit_allow_all(fl_ctx=fl_ctx, workspace=workspace, policy_source=policy_source)
            return

        if not any(self._path_matches_prefix(component_path, prefix) for prefix in allow_list):
            node_path = node.path() if node else ""
            message = f"Component {component_path!r} at config path {node_path!r} is not in allow_list"
            if enforcement_mode == ClassListEnforcementMode.WARN:
                message = (
                    f"{message}; allowing it because {CLASS_LIST_ENFORCEMENT_MODE} is "
                    f"'{ClassListEnforcementMode.WARN.value}'"
                )
                self._audit_warn_mode_allow(
                    component_path=component_path,
                    message=message,
                    policy_source=policy_source,
                    fl_ctx=fl_ctx,
                    workspace=workspace,
                )
                self._log_warning(fl_ctx, message)
                return
            raise UnsafeComponentError(message)

    def _audit_allow_all(self, policy_source: str, fl_ctx: Optional[FLContext] = None, workspace=None):
        audit_scope = self._get_audit_scope(fl_ctx=fl_ctx, workspace=workspace)
        audit_key = (_ALLOW_ALL_AUDIT_ACTION, audit_scope, policy_source)
        message = (
            f"{CLASS_ALLOW_LIST} contains '{ALLOW_ALL}'; all component classes are allowed and "
            f"the remaining allow-list entries are ignored; policy source: {policy_source}"
        )
        self._add_audit_event_once(audit_key=audit_key, action=_ALLOW_ALL_AUDIT_ACTION, message=message, fl_ctx=fl_ctx)
        self._log_warning_once(audit_key, fl_ctx, message)

    def _audit_warn_mode_allow(
        self,
        component_path: str,
        message: str,
        policy_source: str,
        fl_ctx: Optional[FLContext] = None,
        workspace=None,
    ):
        audit_scope = self._get_audit_scope(fl_ctx=fl_ctx, workspace=workspace)
        audit_key = (_WARN_MODE_AUDIT_ACTION, audit_scope, policy_source, component_path)
        audit_message = f"{message}; policy source: {policy_source}"
        self._add_audit_event_once(
            audit_key=audit_key,
            action=_WARN_MODE_AUDIT_ACTION,
            message=audit_message,
            fl_ctx=fl_ctx,
        )

    def _add_audit_event_once(self, audit_key, action: str, message: str, fl_ctx: Optional[FLContext] = None):
        with self._audit_state_lock:
            if audit_key in self._successful_audit_keys:
                return

        job_id = self._get_job_id(fl_ctx)
        event_id = AuditService.add_event(
            user=fl_ctx.get_identity_name() if fl_ctx and fl_ctx.get_identity_name() else "system",
            action=action,
            ref=str(job_id) if job_id else "",
            msg=message,
        )
        if event_id:
            with self._audit_state_lock:
                self._successful_audit_keys.add(audit_key)

    def _log_warning_once(self, warning_key, fl_ctx: Optional[FLContext], message: str):
        with self._audit_state_lock:
            if warning_key in self._warning_keys:
                return
            self._warning_keys.add(warning_key)
        self._log_warning(fl_ctx, message)

    def _log_warning(self, fl_ctx: Optional[FLContext], message: str):
        if fl_ctx:
            self.log_warning(fl_ctx=fl_ctx, msg=message, fire_event=False)
        else:
            self.logger.warning(message)

    def _get_policy(self, fl_ctx: Optional[FLContext] = None, workspace=None):
        resources_file = self._get_resources_file_path(fl_ctx=fl_ctx, workspace=workspace)
        if resources_file:
            allow_list, enforcement_mode = self._get_policy_from_file(resources_file, fl_ctx=fl_ctx)
            return allow_list, enforcement_mode, os.path.abspath(resources_file)

        resources = ConfigService.get_section(SystemConfigs.RESOURCES_CONF)
        allow_list, enforcement_mode = self._get_policy_from_resources(resources, fl_ctx=fl_ctx)
        return allow_list, enforcement_mode, SystemConfigs.RESOURCES_CONF

    def _get_policy_from_file(self, resources_file, fl_ctx: Optional[FLContext] = None):
        cache_key = os.path.abspath(resources_file)

        # cache entries are (file_signature, allow_list, enforcement_mode)
        with self._allow_list_cache_lock:
            stat_result = os.stat(cache_key)
            cache_signature = self._make_file_signature(stat_result)
            cached = self._allow_list_cache.get(cache_key)
            if cached and cached[0] == cache_signature:
                return cached[1], cached[2]

            with open(cache_key, "rt") as f:
                resources = json.load(f)
                cache_signature = self._make_file_signature(os.fstat(f.fileno()))
            allow_list, enforcement_mode = self._get_policy_from_resources(resources, fl_ctx=fl_ctx)

            self._allow_list_cache[cache_key] = (cache_signature, allow_list, enforcement_mode)

        return allow_list, enforcement_mode

    @staticmethod
    def _make_file_signature(stat_result):
        return (stat_result.st_mtime_ns, stat_result.st_size, stat_result.st_ino, stat_result.st_dev)

    def _get_policy_from_resources(self, resources, fl_ctx: Optional[FLContext] = None):
        if resources is None:
            resources = {}
        elif not isinstance(resources, dict):
            raise UnsafeComponentError(f"resources must be dict but got {type(resources)}")

        if CLASS_ALLOW_LIST in resources:
            allow_list = resources.get(CLASS_ALLOW_LIST)
        else:
            allow_list = list(DEFAULT_CLASS_ALLOW_LIST)
            self._warn_default_allow_list_once(fl_ctx)
        if not isinstance(allow_list, list):
            raise UnsafeComponentError(f"{CLASS_ALLOW_LIST} must be list but got {type(allow_list)}")

        enforcement_mode = resources.get(CLASS_LIST_ENFORCEMENT_MODE, ClassListEnforcementMode.ENFORCE.value)
        if not isinstance(enforcement_mode, str):
            raise UnsafeComponentError(f"{CLASS_LIST_ENFORCEMENT_MODE} must be str but got {type(enforcement_mode)}")
        try:
            enforcement_mode = ClassListEnforcementMode(enforcement_mode)
        except ValueError:
            valid_modes = tuple(mode.value for mode in ClassListEnforcementMode)
            raise UnsafeComponentError(
                f"{CLASS_LIST_ENFORCEMENT_MODE} must be one of {valid_modes} but got '{enforcement_mode}'"
            )
        try:
            return self._normalize_allow_list(allow_list), enforcement_mode
        except (TypeError, ValueError) as ex:
            raise UnsafeComponentError(str(ex))

    def _warn_default_allow_list_once(self, fl_ctx: Optional[FLContext] = None):
        # plain check-then-set: a rare duplicate log is harmless, and this must
        # not take _allow_list_cache_lock (already held on the file-cache path)
        if self._warned_default_allow_list:
            return
        self._warned_default_allow_list = True
        self._log_warning(
            fl_ctx,
            f"{CLASS_ALLOW_LIST} is not configured in resources.json or resources.json.default; "
            f"using the built-in default list of {len(DEFAULT_CLASS_ALLOW_LIST)} NVFLARE component classes. "
            f"Configure a top-level {CLASS_ALLOW_LIST} in site resources to replace the default.",
        )

    def _get_audit_scope(self, fl_ctx: Optional[FLContext] = None, workspace=None):
        job_id = self._get_job_id(fl_ctx)
        if job_id:
            return ("job", str(job_id))

        if fl_ctx:
            engine = fl_ctx.get_engine()
            if engine:
                return ("engine", id(engine))
            if workspace is None:
                workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        if workspace:
            return ("workspace", id(workspace))
        if fl_ctx:
            return ("context", id(fl_ctx))
        return ("authorizer", id(self))

    @staticmethod
    def _get_job_id(fl_ctx: Optional[FLContext]):
        if not fl_ctx:
            return ""
        return (
            fl_ctx.get_job_id()
            or fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID)
            or fl_ctx.get_prop(FLContextKey.JOB_RUN_NUMBER)
            or ""
        )

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
        if not _COMPONENT_PATH_PATTERN.fullmatch(component_path):
            raise UnsafeComponentError(
                f"Component {key} must be a fully qualified dotted Python path but got {component_path!r}"
            )

        return component_path
