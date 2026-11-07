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


class Function:
    def __init__(self, name, arguments, type_name):
        self.name = name
        self.arguments = arguments
        if type_name != "number":
            raise SystemExit("Error: object fluents not supported\n" + "(function %s has type %s)" % (name, type_name))
        self.type_name = type_name

    def __str__(self):
        result = "%s(%s)" % (self.name, ", ".join(map(str, self.arguments)))
        if self.type_name:
            result += ": %s" % self.type_name
        return result
