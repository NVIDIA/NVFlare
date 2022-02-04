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

import os
import time
import uuid

from .pipe import Pipe


class FilePipe(Pipe):
    def __init__(self, root_path: str, name: str):
        """Implementation of communication through the file system.

        Args:
            root_path: root path
            name: name of pipe
        """
        assert os.path.exists(root_path), 'root path "{}" does not exist'.format(root_path)
        pipe_path = os.path.join(root_path, name)

        if not os.path.exists(pipe_path):
            os.mkdir(pipe_path)

        x_path = os.path.join(pipe_path, "x")
        if not os.path.exists(x_path):
            os.mkdir(x_path)
            print("created {}".format(x_path))

        y_path = os.path.join(pipe_path, "y")
        if not os.path.exists(y_path):
            os.mkdir(y_path)
            print("created {}".format(y_path))

        t_path = os.path.join(pipe_path, "t")
        if not os.path.exists(t_path):
            os.mkdir(t_path)
            print("created {}".format(t_path))

        Pipe.__init__(self, name)
        self.pipe_path = pipe_path
        self.x_path = x_path
        self.y_path = y_path
        self.t_path = t_path

    def _clear_dir(self, p: str):
        file_list = os.listdir(p)
        if file_list:
            for f in file_list:
                os.remove(os.path.join(p, f))

    def _topic_to_file_name(self, topic: str):
        return "{}.{}".format(topic, uuid.uuid4())

    def _file_name_to_topic(self, file_name: str):
        parts = file_name.split(".")
        return ".".join(parts[0 : len(parts) - 1])

    def _create_file(self, dir: str, topic: str, data_bytes):
        file_name = self._topic_to_file_name(topic)
        file_path = os.path.join(dir, file_name)

        tmp_name = file_name + ".tmp"
        tmp_path = os.path.join(self.t_path, tmp_name)
        with open(tmp_path, "wb") as f:
            f.write(data_bytes)
        os.rename(tmp_path, file_path)

    def clear(self):
        self._clear_dir(self.x_path)
        self._clear_dir(self.y_path)
        self._clear_dir(self.t_path)

    def x_put(self, topic: str, data_bytes):
        # put it in Y's queue
        self._create_file(self.y_path, topic, data_bytes)

    def _get_next(self, from_dir: str):
        # print('get from dir: {}'.format(from_dir))
        files = os.listdir(from_dir)
        if files:
            files = [os.path.join(from_dir, f) for f in files]
            files.sort(key=os.path.getmtime, reverse=False)
            next = files[0]
            # print('got file {}'.format(next))

            topic = self._file_name_to_topic(os.path.basename(next))
            with open(next, mode="rb") as file:  # b is important -> binary
                data = file.read()
            os.remove(next)  # remove this file
            return topic, data
        else:
            return None, None

    def _get_from_dir(self, from_dir: str, timeout=None):
        if not timeout or timeout <= 0:
            return self._get_next(from_dir)

        start = time.time()
        while True:
            topic, data = self._get_next(from_dir)
            if topic is not None:
                return topic, data

            if time.time() - start >= timeout:
                break
            time.sleep(2)

        return None, None

    def x_get(self, timeout=None) -> (str, bytes):
        # read from X's queue
        return self._get_from_dir(self.x_path, timeout)

    def y_put(self, topic: str, data_bytes):
        # put it in X's queue
        self._create_file(self.x_path, topic, data_bytes)

    def y_get(self, timeout=None):
        # read from Y's queue
        return self._get_from_dir(self.y_path, timeout)
