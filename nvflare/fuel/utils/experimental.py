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


def experimental(reason):
    """
    This is a decorator which can be used to mark classes
    as experimental. It will result in a warning being emitted
    when the class is used.
    """

    def decorator(func):
        fmt = "Use of experimental class {name}{reason}."

        class DeprecatedClass(cls):
            __name__ = cls.__name__
            def __new__(cls, *args, **kwargs):
                warnings.simplefilter("always", DeprecationWarning)
                warnings.warn(
                    fmt.format(
                        name=cls.__name__,
                        reason=f" ({reason})" if reason else "",
                    ),
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                warnings.simplefilter("default", DeprecationWarning)
                return super(DeprecatedClass, cls).__new__(cls)

        return DeprecatedClass

    if inspect.isclass(reason) or inspect.isfunction(reason):
        # The @experimental is used without any 'reason'.
        return decorator(reason)
    elif isinstance(reason, str):
        # The @experimental is used with a 'reason'.
        return decorator
    else:
        raise TypeError(repr(type(reason)))
