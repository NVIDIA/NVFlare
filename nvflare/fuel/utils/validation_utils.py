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

SYMBOL_ALL = "@all"
SYMBOL_NONE = "@none"


class DefaultPolicy:

    DISALLOW = "disallow"
    ANY = "any"
    EMPTY = "empty"
    ALL = "all"


def check_positive_int(name, value):
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, but got {type(value)}.")
    if value <= 0:
        raise ValueError(f"{name} must > 0, but got {value}")


def check_non_negative_int(name, value):
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, but got {type(value)}.")
    if value < 0:
        raise ValueError(f"{name} must >= 0, but got {value}")


def check_positive_number(name, value):
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, but got {type(value)}.")
    if value <= 0:
        raise ValueError(f"{name} must > 0, but got {value}")


def check_non_negative_number(name, value):
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, but got {type(value)}.")
    if value < 0:
        raise ValueError(f"{name} must >= 0, but got {value}")


def check_str(name, value):
    check_object_type(name, value, str)


def check_object_type(name, value, obj_type):
    if not isinstance(value, obj_type):
        raise TypeError(f"{name} must be {obj_type}, but got {type(value)}.")


def check_callable(name, value):
    if not callable(value):
        raise ValueError(f"{name} must be callable, but got {type(value)}.")


def _validate_candidates(var_name: str, candidates, base: list):
    if candidates is None:
        return []  # empty

    if isinstance(candidates, str):
        c = candidates.lower().strip()
        if not c:
            return None

        if c == SYMBOL_ALL:
            return candidates
        elif c == SYMBOL_NONE:
            return None
        elif c in candidates:
            return [c]
        else:
            raise ValueError(f"value of '{var_name}' ({candidates}) is invalid")

    if not isinstance(candidates, list):
        raise ValueError(f"invalid '{var_name}': expect str or list of str but got {type(candidates)}")

    validated = []
    for c in candidates:
        if not isinstance(c, str):
            raise ValueError(f"invalid value in '{var_name}': must be str but got {type(c)}")
        n = c.strip()
        if n not in base:
            raise ValueError(f"invalid value '{n}' in '{var_name}'")
        if n not in validated:
            validated.append(n)

    return validated


def validate_candidates(var_name: str, candidates, base: list, default_policy: str, allow_none: bool):
    c = _validate_candidates(var_name, candidates, base)

    if c is None:
        if not allow_none:
            raise ValueError(f"{var_name} must not be none")
        else:
            return []  # empty

    if not c:
        # empty
        if default_policy == DefaultPolicy.EMPTY:
            return []
        elif default_policy == DefaultPolicy.ALL:
            return base
        elif default_policy == DefaultPolicy.DISALLOW:
            raise ValueError(f"invalid value '{candidates}' in '{var_name}': it must be subset of {base}")
        else:
            # any
            return [candidates[0]]
    return c


def _validate_candidate(var_name: str, candidate, base: list):
    if candidate is None:
        return None

    if not isinstance(candidate, str):
        raise ValueError(f"invalid '{var_name}': must be str but got {type(candidate)}")
    n = candidate.strip()
    if n in base:
        return n

    c = n.lower()
    if c == SYMBOL_NONE:
        return None
    elif not c:
        return ""
    else:
        raise ValueError(f"invalid value '{candidate}' in '{var_name}'")


def validate_candidate(var_name: str, candidate, base: list, default_policy: str, allow_none: bool):
    c = _validate_candidate(var_name, candidate, base)
    if c is None:
        if not allow_none:
            raise ValueError(f"{var_name} must be specified")
        else:
            return ""

    if not c:
        if default_policy == DefaultPolicy.EMPTY:
            return ""
        elif default_policy == DefaultPolicy.ANY:
            return base[0]
        else:
            raise ValueError(f"invalid value '{candidate}' in '{var_name}': it must be one of {base}")
    else:
        return c
