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


def check_str(name, value):
    check_object_type(name, value, str)


def check_object_type(name, value, obj_type):
    if not isinstance(value, obj_type):
        raise TypeError(f"{name} must be {obj_type}, but got {type(value)}.")


def check_callable(name, value):
    if not callable(value):
        raise ValueError(f"{name} must be callable, but got {type(value)}.")
