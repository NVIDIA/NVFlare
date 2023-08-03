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

#  NOTE: This code is adapted from pyhocon.converter.HOCONCOnvert.to_hocon() method
#  https://github.com/chimpler/pyhocon/blob/master/pyhocon/converter.py
#  The code is modified to allow comment to the output string.
from pyhocon import HOCONConverter
from pyhocon.config_tree import ConfigQuotedString, ConfigSubstitution, ConfigTree, ConfigValues, NoneValue
from pyhocon.period_serializer import is_timedelta_like, timedelta_to_hocon

try:
    basestring
except NameError:
    basestring = str
    unicode = str

try:
    from dateutil.relativedelta import relativedelta
except Exception:
    relativedelta = None


class ConfigTreeEx(ConfigTree):
    def __init__(self, config: ConfigTree, *args, **kwds):
        super().__init__(*args, **kwds)
        ConfigTree.merge_configs(self, config, copy_trees=True)
        self.comment = ""

    def put_comment(self, comment: str):
        self.comment = comment

    def to_hocon(self):
        return ConfigTreeEx._to_hocon(self)

    @classmethod
    def _to_hocon(cls, config, compact=False, indent=2, level=0):
        """
        return string honcon representation
        Args:
            config:
            compact:
            indent:
            level:
        Returns: string hocon format
        """

        lines = ""
        if isinstance(config, ConfigTreeEx):
            if config.comment:
                lines += f"\n#{config.comment}\n"

        if isinstance(config, ConfigTree):
            if len(config) == 0:
                lines += "{}"
            else:
                if level > 0:  # don't display { at root level
                    lines += "{\n"
                bet_lines = []

                for key, item in config.items():
                    if compact:
                        full_key = key
                        while isinstance(item, ConfigTree) and len(item) == 1:
                            key, item = next(iter(item.items()))
                            full_key += "." + key
                    else:
                        full_key = key

                    bet_lines.append(
                        "{indent}{key}{assign_sign} {value}".format(
                            indent="".rjust(level * indent, " "),
                            key=full_key,
                            assign_sign="" if isinstance(item, dict) else " =",
                            value=cls._to_hocon(item, compact, indent, level + 1),
                        )
                    )
                lines += "\n".join(bet_lines)

                if level > 0:  # don't display { at root level
                    lines += "\n{indent}}}".format(indent="".rjust((level - 1) * indent, " "))
        elif isinstance(config, list):
            if len(config) == 0:
                lines += "[]"
            else:
                lines += "[\n"
                bet_lines = []
                for item in config:
                    bet_lines.append(
                        "{indent}{value}".format(
                            indent="".rjust(level * indent, " "), value=cls._to_hocon(item, compact, indent, level + 1)
                        )
                    )
                lines += "\n".join(bet_lines)
                lines += "\n{indent}]".format(indent="".rjust((level - 1) * indent, " "))
        elif isinstance(config, basestring):
            if "\n" in config and len(config) > 1:
                lines = '"""{value}"""'.format(value=config)  # multilines
            else:
                lines = '"{value}"'.format(value=HOCONConverter._escape_string(config))
        elif isinstance(config, ConfigValues):
            lines = "".join(cls._to_hocon(o, compact, indent, level) for o in config.tokens)
        elif isinstance(config, ConfigSubstitution):
            lines = "${"
            if config.optional:
                lines += "?"
            lines += config.variable + "}" + config.ws
        elif isinstance(config, ConfigQuotedString):
            if "\n" in config.value and len(config.value) > 1:
                lines = '"""{value}"""'.format(value=config.value)  # multilines
            else:
                lines = '"{value}"'.format(value=HOCONConverter._escape_string(config.value))
        elif is_timedelta_like(config):
            lines += timedelta_to_hocon(config)
        elif config is None or isinstance(config, NoneValue):
            lines = "null"
        elif config is True:
            lines = "true"
        elif config is False:
            lines = "false"
        else:
            lines = str(config)
        return lines
