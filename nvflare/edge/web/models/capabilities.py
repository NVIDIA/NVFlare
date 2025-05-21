# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.edge.web.models.base_model import BaseModel, EdgeProtoKey


class Capabilities(BaseModel):
    def __init__(self, methods: list):
        super().__init__()
        self.methods = methods

    @staticmethod
    def extract_from_dict(d: dict):
        cap_dict = d.pop(EdgeProtoKey.CAPABILITIES, None)
        if not cap_dict:
            return "missing capabilities", None

        methods = cap_dict.pop(EdgeProtoKey.METHODS, None)
        if not methods:
            return "missing methods", None

        if not isinstance(methods, list):
            return f"invalid methods type {type(methods)}", None

        return "", Capabilities(methods)
