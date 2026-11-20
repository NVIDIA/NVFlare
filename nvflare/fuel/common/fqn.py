# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import re
from typing import List


class FQN:

    SEPARATOR = "."
    ROOT_SERVER = "server"
    VALID_PATTERN = "^[A-Za-z0-9_.-]*$"

    @staticmethod
    def normalize(fqn: str) -> str:
        # Pass a non-str through unchanged so validate reports its type, rather than
        # raising AttributeError here. Callers normalize before they validate.
        return fqn.strip() if isinstance(fqn, str) else fqn

    @staticmethod
    def split(fqn: str) -> List[str]:
        return fqn.split(FQN.SEPARATOR)

    @staticmethod
    def join(path: List[str]) -> str:
        return FQN.SEPARATOR.join(path)

    @classmethod
    def validate(cls, fqn) -> str:
        if not isinstance(fqn, str):
            return f"must be str but got {type(fqn)}"
        if not fqn:
            return "empty"
        # Check the name the caller holds, not a stripped copy of it: the caller keeps
        # using its own string. fullmatch is needed because "$" also matches just before
        # a trailing newline, which would let "server\n" through on its own.
        valid = bool(re.fullmatch(cls.VALID_PATTERN, fqn))
        if not valid:
            return "invalid char"
        parts = cls.split(fqn)
        info = {}
        for p in parts:
            if not p:
                return "empty part"
            if info.get(p):
                return f"dup '{p}'"
            info[p] = True
        return ""

    @staticmethod
    def get_root(fqn: str) -> str:
        parts = FQN.split(fqn)
        return parts[0]

    @staticmethod
    def get_parent(fqn: str) -> str:
        parts = FQN.split(fqn)
        if len(parts) == 1:
            return ""
        return FQN.join(parts[0:-1])

    @staticmethod
    def is_parent(fqn1: str, fqn2: str) -> bool:
        return fqn1 == FQN.get_parent(fqn2)

    @staticmethod
    def is_ancestor(fqn1: str, fqn2: str) -> bool:
        return fqn2.startswith(fqn1 + FQN.SEPARATOR)
