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


class FQCN:

    SEPARATOR = "."
    ROOT_SERVER = "server"

    @staticmethod
    def normalize(fqcn: str) -> str:
        return fqcn.strip()

    @staticmethod
    def split(fqcn: str) -> List[str]:
        return fqcn.split(FQCN.SEPARATOR)

    @staticmethod
    def join(path: List[str]) -> str:
        return FQCN.SEPARATOR.join(path)

    @staticmethod
    def validate(fqcn) -> str:
        if not isinstance(fqcn, str):
            return f"must be str but got {type(fqcn)}"
        fqcn = FQCN.normalize(fqcn)
        if not fqcn:
            return "empty"
        pattern = "^[A-Za-z0-9_.-]*$"
        valid = bool(re.match(pattern, fqcn))
        if not valid:
            return "invalid char"
        parts = FQCN.split(fqcn)
        info = {}
        for p in parts:
            if not p:
                return "empty part"
            if info.get(p):
                return f"dup '{p}'"
            info[p] = True
        return ""

    @staticmethod
    def get_root(fqcn: str) -> str:
        parts = FQCN.split(fqcn)
        return parts[0]

    @staticmethod
    def get_parent(fqcn: str) -> str:
        parts = FQCN.split(fqcn)
        if len(parts) == 1:
            return ""
        return FQCN.join(parts[0:-1])

    @staticmethod
    def is_parent(fqcn1: str, fqcn2: str) -> bool:
        return fqcn1 == FQCN.get_parent(fqcn2)

    @staticmethod
    def is_ancestor(fqcn1: str, fqcn2: str) -> bool:
        return fqcn2.startswith(fqcn1 + FQCN.SEPARATOR)


class FqcnInfo:
    def __init__(self, fqcn: str):
        self.fqcn = fqcn
        self.path = FQCN.split(fqcn)
        self.gen = len(self.path)
        self.is_root = self.gen == 1
        self.root = self.path[0]
        self.is_on_server = self.root == FQCN.ROOT_SERVER


def same_family(info1: FqcnInfo, info2: FqcnInfo):
    return info1.root == info2.root
