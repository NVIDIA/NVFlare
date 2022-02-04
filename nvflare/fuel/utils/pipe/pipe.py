# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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


class EndPoint(object):
    def __init__(self, get_func, put_func):
        """Object with put and get functions.

        Args:
            get_func: get function
            put_func: put function
        """
        self.get_func = get_func
        self.put_func = put_func

    def put(self, topic: str, data_bytes):
        self.put_func(topic=topic, data_bytes=data_bytes)

    def get(self, timeout=None):
        return self.get_func(timeout=timeout)


class Pipe(object):
    def __init__(self, name):
        """Base class for communication.

        Args:
            name: name of pipe
        """
        self.name = name
        self.x = EndPoint(self.x_get, self.x_put)
        self.y = EndPoint(self.y_get, self.y_put)

    def clear(self):
        pass

    def x_put(self, topic: str, data_bytes):
        pass

    def x_get(self, timeout=None):
        pass

    def y_put(self, topic: str, data_bytes):
        pass

    def y_get(self, timeout=None):
        pass
