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

import random

SYMBOL_ALL = "@all"
SYMBOL_NONE = "@none"


class DefaultValuePolicy:

    """
    Defines policy for how to determine default value
    """

    DISALLOW = "disallow"
    ANY = "any"
    RANDOM = "random"
    EMPTY = "empty"
    ALL = "all"

    @classmethod
    def valid_policy(cls, p: str):
        return p in [cls.DISALLOW, cls.ANY, cls.RANDOM, cls.EMPTY, cls.ALL]


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


def check_number_range(name, value, min_value=None, max_value=None):
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, but got {type(value)}.")

    if min_value is not None:
        if not isinstance(min_value, (int, float)):
            raise TypeError(f"{name}: min_value must be a number but got {type(min_value)}.")
        if value < min_value:
            raise ValueError(f"{name} must be >= {min_value} but got {value}")

    if max_value is not None:
        if not isinstance(max_value, (int, float)):
            raise TypeError(f"{name}: max_value must be a number but got {type(max_value)}.")
        if value > max_value:
            raise ValueError(f"{name} must be <= {max_value} but got {value}")


def check_non_negative_number(name, value):
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, but got {type(value)}.")
    if value < 0:
        raise ValueError(f"{name} must >= 0, but got {value}")


def check_str(name, value):
    check_object_type(name, value, str)


def check_non_empty_str(name, value):
    check_object_type(name, value, str)
    v = value.strip()
    if not v:
        raise ValueError(f"{name} must not be empty")


def check_object_type(name, value, obj_type):
    if not isinstance(value, obj_type):
        raise TypeError(f"{name} must be {obj_type}, but got {type(value)}.")


def check_callable(name, value):
    if not callable(value):
        raise ValueError(f"{name} must be callable, but got {type(value)}.")


def _determine_candidates_value(var_name: str, candidates, base: list):
    if not isinstance(base, list):
        raise TypeError(f"base must be list but got {type(base)}")

    if candidates is None:
        return None  # empty

    if isinstance(candidates, str):
        nc = candidates.strip()
        if not nc:
            return []

        c = nc.lower()
        if c == SYMBOL_ALL:
            return base
        elif c == SYMBOL_NONE:
            return None
        elif nc in base:
            return [nc]
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
    """Validate specified candidates against the items in the "base" list, based on specified policy
    and returns determined value for the candidates.

    The value of candidates could have the following cases:
    1. Not explicitly specified (Python object None or empty list [])
    In this case, the default_policy decides the final result:
    - ANY: returns a list that contains a single item from the base
    - RANDOM: returns a list that contains a random item from the base
    - EMPTY: returns an empty list
    - ALL: returns the base list
    - DISALLOW: raise exception - candidates must be explicitly specified

    2. A list of string items
    In this case, each item in the candidates list must be in the "base". Duplicates are removed.

    3. A string with special value "@all" to mean "all items from the base"
    Returns the base list.

    4. A string with special value "@none" to mean "no items"
    If allow_none is True, then returns an empty list; otherwise raise exception.

    5. A string that is not a special value
    If it is in the "base", return a list that contains this item; otherwise raise exception.

    Args:
        var_name: the name of the "candidates" var from the caller
        candidates: the candidates to be validated
        base: the base list that contains valid items
        default_policy: policy for how to handle default value when "candidates" is not explicitly specified.
        allow_none: whether "none" is allowed for candidates.

    Returns:

    """
    if not DefaultValuePolicy.valid_policy(default_policy):
        raise ValueError(f"invalid default policy {default_policy}")

    c = _determine_candidates_value(var_name, candidates, base)

    if c is None:
        if not allow_none:
            raise ValueError(f"{var_name} must not be none")
        else:
            return []  # empty

    if not c:
        # empty
        if default_policy == DefaultValuePolicy.EMPTY:
            return []
        elif default_policy == DefaultValuePolicy.ALL:
            return base
        elif default_policy == DefaultValuePolicy.DISALLOW:
            raise ValueError(f"invalid value '{candidates}' in '{var_name}': it must be subset of {base}")
        elif default_policy == DefaultValuePolicy.RANDOM:
            return [random.choice(base)]
        else:
            # any
            return [base[0]]
    return c


def _determine_candidate_value(var_name: str, candidate, base: list):
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
    """Validate specified candidate against the items in the "base" list, based on specified policy
    and returns determined value for the candidate.

    The value of candidate could have the following cases:
    1. Not explicitly specified (Python object None or empty string)
    In this case, the default_policy decides the final result:
    - ANY: returns the first item from the base
    - RANDOM: returns a random item from the base
    - EMPTY: returns an empty str
    - ALL or DISALLOW: raise exception - candidate must be explicitly specified

    2. A string with special value "@none" to mean "nothing"
    If allow_none is True, then returns an empty str; otherwise raise exception.

    3. A string that is not a special value
    If it is in the "base", return it; otherwise raise exception.

    All other cases, raise exception.

    NOTE: the final value is normalized (leading and trailing white spaces are removed).

    Args:
        var_name: the name of the "candidate" var from the caller
        candidate: the candidate to be validated
        base: the base list that contains valid items
        default_policy: policy for how to handle default value when "candidates" is not explicitly specified.
        allow_none: whether "none" is allowed for candidates.

    Returns:

    """
    if not DefaultValuePolicy.valid_policy(default_policy):
        raise ValueError(f"invalid default policy {default_policy}")

    if default_policy == DefaultValuePolicy.ALL:
        raise ValueError(f"the policy '{default_policy}' is not applicable to validate_candidate")

    c = _determine_candidate_value(var_name, candidate, base)
    if c is None:
        if not allow_none:
            raise ValueError(f"{var_name} must be specified")
        else:
            return ""

    if not c:
        if default_policy == DefaultValuePolicy.EMPTY:
            return ""
        elif default_policy == DefaultValuePolicy.ANY:
            return base[0]
        elif default_policy == DefaultValuePolicy.RANDOM:
            return random.choice(base)
        else:
            raise ValueError(f"invalid value '{candidate}' in '{var_name}': it must be one of {base}")
    else:
        return c


def normalize_config_arg(value):
    if value is False:
        return None  # specified to be "empty"
    if isinstance(value, str):
        if value.strip().lower() == SYMBOL_NONE:
            return None
    if not value:
        return ""  # meaning to take default
    return value
