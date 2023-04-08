# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Union

from nvflare.apis.filter import Filter, FilterChainType, FilterContextKey, FilterSource


class Scope(object):
    def __init__(self):
        self.name = ""
        self.props = {}
        self.task_data_filters = []
        self.task_result_filters = []

    def set_name(self, name: str):
        if not isinstance(name, str):
            raise TypeError(f"scope name must be str but got {type(name)}")
        self.name = name

    def set_props(self, props: dict):
        if not isinstance(props, dict):
            raise TypeError(f"scope properties must be dict but got {type(props)}")
        self.props = props

    def add_task_data_filter(self, f: Filter):
        if not isinstance(f, Filter):
            raise TypeError(f"task data filter must be Filter but got {type(f)}")
        f.set_prop(FilterContextKey.CHAIN_TYPE, FilterChainType.TASK_DATA_CHAIN)
        f.set_prop(FilterContextKey.SOURCE, FilterSource.SITE)
        self.task_data_filters.append(f)

    def add_task_result_filter(self, f: Filter):
        if not isinstance(f, Filter):
            raise TypeError(f"task result filter must be Filter but got {type(f)}")
        f.set_prop(FilterContextKey.CHAIN_TYPE, FilterChainType.TASK_RESULT_CHAIN)
        f.set_prop(FilterContextKey.SOURCE, FilterSource.SITE)
        self.task_result_filters.append(f)


class PrivacyManager(object):
    def __init__(
        self, scopes: Union[None, List[Scope]], default_scope_name: Union[None, str], components: Union[None, dict]
    ):
        self.name_to_scopes = {}
        self.default_scope = None
        self.components = components

        if scopes:
            for s in scopes:
                if s.name in self.name_to_scopes:
                    raise ValueError(f"duplicate scopes defined for name '{s.name}'")
                self.name_to_scopes[s.name] = s
            if default_scope_name:
                self.default_scope = self.name_to_scopes.get(default_scope_name)
                if not self.default_scope:
                    raise ValueError(f"specified default scope '{default_scope_name}' does not exist")
            self.policy_defined = True
        else:
            self.policy_defined = False

    def get_scope(self, name: Union[None, str]):
        if not name:
            return self.default_scope

        return self.name_to_scopes.get(name)

    def is_policy_defined(self):
        return self.policy_defined


class PrivacyService(object):
    manager = None

    @staticmethod
    def initialize(manager: PrivacyManager):
        if manager and not isinstance(manager, PrivacyManager):
            raise TypeError(f"manager must be an instance of PrivacyManager, but get {type(manager)}.")
        PrivacyService.manager = manager

    @staticmethod
    def get_scope(name: Union[None, str]):
        if not PrivacyService.manager:
            return None
        else:
            return PrivacyService.manager.get_scope(name)

    @staticmethod
    def is_policy_defined():
        if not PrivacyService.manager:
            return False
        else:
            return PrivacyService.manager.is_policy_defined()

    @staticmethod
    def is_scope_allowed(scope_name: str):
        """Check whether the specified scope is allowed

        Args:
            scope_name: scope to be checked

        Returns:

        """
        if not PrivacyService.is_policy_defined():
            return True

        scope = PrivacyService.get_scope(scope_name)
        return scope is not None

    @staticmethod
    def get_manager():
        return PrivacyService.manager
