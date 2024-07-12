# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import operator
from typing import Callable, Optional, Tuple

operator_mapping = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "=": operator.eq,
}


def parse_compare_criteria(compare_expr: Optional[str] = None) -> Tuple[str, float, Callable]:
    """
        Parse the compare expression into individual component
        compare expression is in the format of string literal : "<key> <op> <value"
        such as
            accuracy >= 0.5
            loss > 2.4
    Args:
        compare_expr: string literal in the format of  "<key> <op> <value>"
    Returns: Tuple key, value, operator
    """
    tokens = compare_expr.split(" ")
    if len(tokens) != 3:
        raise ValueError(
            f"Invalid early_stop_condition, expecting form of '<metric> <op> value' but got '{compare_expr}'"
        )

    key = tokens[0]
    op = tokens[1]
    target = tokens[2]
    op_fn = operator_mapping.get(op, None)
    if op_fn is None:
        raise ValueError("Invalid operator symbol: expecting one of <=, =, >=, <, > ")
    if not target:
        raise ValueError("Invalid empty or None target value")
    try:
        target_value = float(target)
    except Exception as e:
        raise ValueError(f"expect a number, but get '{target}' in '{compare_expr}'")

    return key, target_value, op_fn
