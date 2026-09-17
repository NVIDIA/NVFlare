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

"""Restricted, platform-neutral optional application service definitions."""

import configparser
import re

from .common import require


def validate_service(name, text):
    require(
        re.fullmatch(r"app_[a-z0-9][a-z0-9_]*\.service", name),
        "Application service names must use app_<name>.service with lowercase letters, digits, and underscores",
    )
    require(len(text) < 65536 and "\x00" not in text and "\\\n" not in text, "Invalid application service")
    require(
        not any(value in text for value in ("/dev/sev", "/dev/tdx", "snpguest", "TEE_PLATFORM=", "TEE_DEVICE=")),
        "Application services must use /run/cvm/platform.env",
    )
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    parser.read_string(text)
    require(not parser.defaults() and set(parser.sections()) <= {"Unit", "Service"}, "Unsupported service section")
    require("Service" in parser, "Missing service configuration")
    require(
        set(parser["Unit"] if "Unit" in parser else {}) <= {"Description"}, "Builder owns application service ordering"
    )
    require(
        set(parser["Service"]) <= {"Type", "ExecStart", "WorkingDirectory", "User", "Group", "Environment"},
        "Unsupported service override",
    )
    require(parser["Service"].get("Type", "simple") in ("simple", "exec"), "Application service must be supervised")
    require(
        parser["Service"].get("ExecStart", "").startswith("/vault/application/"),
        "Service executable must be in authenticated application payload",
    )
    return text
