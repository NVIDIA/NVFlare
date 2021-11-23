# Copyright (c) 2021, NVIDIA CORPORATION.
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

from distutils.util import strtobool


def parse_var(s):
    """Returns (key, value) tuple from string with equals sign with the portion before the first equals sign as the key
    and the rest as the value.

    Args:
        s: string to parse

    Returns: Tuple of key and value
    """
    items = s.split("=")
    key = items[0].strip()  # we remove blanks around keys, as is logical
    value = ""
    if len(items) > 1:
        # rejoin the rest:
        value = "=".join(items[1:])
    return key, value


def parse_vars(items):
    """Converts a list of key value pairs into a dictionary.

    Args:
        items: list like ['a=1', 'b=2', 'c=3']

    Returns: dictionary like {'a': '1', 'b': '2', 'c': '3'}

    """
    d = {}
    if items:
        for item in items:
            key, value = parse_var(item)

            # d[key] = value
            try:
                d[key] = int(value)
            except ValueError:
                pass
                try:
                    d[key] = float(value)
                except ValueError:
                    pass
                    try:
                        d[key] = bool(strtobool(str(value)))
                    except ValueError:
                        pass
                        d[key] = value
    return d
