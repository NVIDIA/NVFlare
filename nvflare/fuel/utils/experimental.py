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
    This is a decorator which can be used to mark classes and functions
    as experimental. It will result in a warning being emitted
    when the class is used.

    # Example of usage:

    # Example 1: Use with class inheritance
        @experimental("Because it's experimental")
        class ExperimentalBaseClass:
            def __init__(self):
                print("Creating an instance of ExperimentalBaseClass")

        @experimental("Derived from experimental class")
        class ExperimentalClass(ExperimentalBaseClass):
            def __init__(self):
                ExperimentalBaseClass.__init__(self)
                print("Creating an instance of ExperimentalClass")

        # Testing the experimental class
        experimental_instance = ExperimentalClass()  # This should emit two experimental warnings for base and derived class.

    # Example 2: Use with functions
        @experimental("Experimental function")
        def test_f(a, b):
            print(f"hello {a} and {b}")

        # Testing the experimental function
        test_f("Adam", "Eve")  # This should emit an experimental warning for use of the function.
    """

    def decorator(obj):
        if inspect.isclass(obj):
            fmt = "Use of experimental class {name}{reason}."
            orig_cls_name = obj.__name__

            class ExperimentalClass(obj):
                def __new__(obj, *args, **kwargs):
                    warnings.simplefilter("always", Warning)
                    warnings.warn(
                        fmt.format(
                            name=orig_cls_name,
                            reason=f" ({reason})" if reason else "",
                        ),
                        category=Warning,
                        stacklevel=2,
                    )
                    warnings.simplefilter("default", Warning)
                    return super(ExperimentalClass, obj).__new__(obj)

            return ExperimentalClass
        else:  # function
            fmt = "Call to experimental function {name}{reason}."

            @functools.wraps(obj)
            def new_func(*args, **kwargs):
                warnings.simplefilter("always", Warning)
                warnings.warn(
                    fmt.format(
                        name=obj.__name__,
                        reason=f" ({reason})" if reason else "",
                    ),
                    category=Warning,
                    stacklevel=2,
                )
                warnings.simplefilter("default", Warning)
                return obj(*args, **kwargs)

            return new_func

    if inspect.isclass(reason) or inspect.isfunction(reason):
        # The @experimental is used without any 'reason'.
        return decorator(reason)
    elif isinstance(reason, str):
        # The @experimental is used with a 'reason'.
        return decorator
    else:
        raise TypeError(f"@experimental decorator `reason` expected to be string but got {type(reason)}")
