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

import functools
import inspect
import warnings


def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    def decorator(func):
        fmt = "Call to deprecated {kind} {name}{reason}."

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                fmt.format(
                    kind="class" if inspect.isclass(func) else "function",
                    name=func.__name__,
                    reason=f" ({reason})" if reason else "",
                ),
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter("default", DeprecationWarning)
            return func(*args, **kwargs)

        return new_func

    if inspect.isclass(reason) or inspect.isfunction(reason):
        # The @deprecated is used without any 'reason'.
        return decorator(reason)
    elif isinstance(reason, str):
        # The @deprecated is used with a 'reason'.
        return decorator
    else:
        raise TypeError(repr(type(reason)))
