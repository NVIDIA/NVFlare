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

from argparse import Namespace
from collections import namedtuple

from nvflare.tool.cli_arg_utils import get_arg_value


def test_get_arg_value_returns_default_for_none():
    assert get_arg_value(None, "missing", "default") == "default"


def test_get_arg_value_reads_namespace_dict():
    assert get_arg_value(Namespace(study="default"), "study") == "default"
    assert get_arg_value(Namespace(), "study", "fallback") == "fallback"


def test_get_arg_value_falls_back_to_attribute_lookup_for_slots_like_args():
    Args = namedtuple("Args", ["kit_id"])

    assert get_arg_value(Args("lead"), "kit_id") == "lead"
    assert get_arg_value(Args("lead"), "startup_kit", "fallback") == "fallback"
