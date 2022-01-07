# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

type_pattern_mapping = {
    "server": r"^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9])$",
    "client": r"^[A-Za-z0-9-_]+$",
    "admin": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$",
    "email": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$",
    "org": r"^[A-Za-z0-9_]+$",
}


def name_check(name: str, entity_type: str):
    regex_pattern = type_pattern_mapping.get(entity_type)
    if regex_pattern is None:
        return True, "entity_type={} not defined, unable to check name={}.".format(entity_type, name)
    if re.match(regex_pattern, name):
        return False, "name={} passed on regex_pattern={} check".format(name, regex_pattern)
    else:
        return True, "name={} is ill-formatted based on regex_pattern={}".format(name, regex_pattern)
