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

import os
import shutil
import time
import uuid

from nvflare.fuel.utils.validation_utils import check_positive_number, check_str

from .pipe import Pipe


class FilePipe(Pipe):
    def __init__(self, root_path: str, file_check_interval=0.1):
        """Implementation of communication through the file system.
        Args:
            root_path: root path
        """
        check_str("root_path", root_path)
        check_positive_number("file_check_interval", file_check_interval)

        if not os.path.exists(root_path):
            # create the root path
            os.makedirs(root_path)

        self.root_path = root_path
        self.file_check_interval = file_check_interval
        self.pipe_path = None
        self.x_path = None
        self.y_path = None
        self.t_path = None
        self.get_f = None
        self.put_f = None

    def open(self, name: str, me: str):
        if me == "x":
            self.get_f = self.x_get
            self.put_f = self.x_put
        elif me == "y":
            self.get_f = self.y_get
            self.put_f = self.y_put
        else:
            raise ValueError(f"me must be 'x' or 'y' but got {me}")

        pipe_path = os.path.join(self.root_path, name)

        if not os.path.exists(pipe_path):
            os.mkdir(pipe_path)

        x_path = os.path.join(pipe_path, "x")
        if not os.path.exists(x_path):
            os.mkdir(x_path)
            print(f"created {x_path}")

        y_path = os.path.join(pipe_path, "y")
        if not os.path.exists(y_path):
            os.mkdir(y_path)
            print(f"created {y_path}")

        t_path = os.path.join(pipe_path, "t")
        if not os.path.exists(t_path):
            os.mkdir(t_path)
            print(f"created {t_path}")

        Pipe.__init__(self)
        self.pipe_path = pipe_path
        self.x_path = x_path
        self.y_path = y_path
        self.t_path = t_path

    @staticmethod
    def _clear_dir(p: str):
        file_list = os.listdir(p)
        if file_list:
            for f in file_list:
                os.remove(os.path.join(p, f))

    @staticmethod
    def _topic_to_file_name(topic: str):
        return f"{topic}.{uuid.uuid4()}"

    @staticmethod
    def _file_name_to_topic(file_name: str):
        parts = file_name.split(".")
        return ".".join(parts[0 : len(parts) - 1])

    def _create_file(self, to_dir: str, topic: str, data_bytes) -> str:
        file_name = self._topic_to_file_name(topic)
        file_path = os.path.join(to_dir, file_name)

        tmp_path = os.path.join(self.t_path, file_name)
        if not self.pipe_path:
            raise BrokenPipeError("pipe broken")
        try:
            with open(tmp_path, "wb") as f:
                f.write(data_bytes)
            os.rename(tmp_path, file_path)
        except FileNotFoundError:
            raise BrokenPipeError("pipe closed")
        return file_path

    def clear(self):
        self._clear_dir(self.x_path)
        self._clear_dir(self.y_path)
        self._clear_dir(self.t_path)

    def _monitor_file(self, file_path: str, timeout) -> bool:
        """
        Monitor the file until it's read-and-removed by peer, or timed out.
        If timeout, remove the file.

        Args:
            file_path: the path to be monitored
            timeout: how long to wait for timeout

        Returns: whether the file has been read and removed

        """
        if not timeout:
            return False
        start = time.time()
        while True:
            if not self.pipe_path:
                raise BrokenPipeError("pipe broken")

            if not os.path.exists(file_path):
                return True
            if time.time() - start > timeout:
                # timed out - try to delete the file
                try:
                    os.remove(file_path)
                except FileNotFoundError:
                    # the file is read by the peer!
                    return True
                return False
            time.sleep(self.file_check_interval)

    def x_put(self, topic: str, data_bytes, timeout) -> bool:
        """

        Args:
            topic:
            data_bytes:
            timeout:

        Returns: tuple of (file_path, whether file is read by the peer)

        """
        # put it in Y's queue
        file_path = self._create_file(self.y_path, topic, data_bytes)
        return self._monitor_file(file_path, timeout)

    def _read_file(self, file_path: str):
        # since reading file may take time and another process may try to delete the file
        # we move the file to a temp name before reading it
        file_name = os.path.basename(file_path)
        topic = self._file_name_to_topic(file_name)
        tmp_path = os.path.join(self.t_path, file_name)
        try:
            os.rename(file_path, tmp_path)
            with open(tmp_path, mode="rb") as file:  # b is important -> binary
                data = file.read()
            os.remove(tmp_path)  # remove this file
            return topic, data
        except FileNotFoundError:
            raise BrokenPipeError("pipe closed")

    def _get_next(self, from_dir: str):
        # print('get from dir: {}'.format(from_dir))
        try:
            files = os.listdir(from_dir)
        except:
            raise BrokenPipeError(f"error reading from {from_dir}")

        if files:
            files = [os.path.join(from_dir, f) for f in files]
            files.sort(key=os.path.getmtime, reverse=False)
            file_path = files[0]
            return self._read_file(file_path)
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
            time.sleep(self.file_check_interval)

        return None, None

    def x_get(self, timeout=None) -> (str, bytes):
        # read from X's queue
        return self._get_from_dir(self.x_path, timeout)

    def y_put(self, topic: str, data_bytes, timeout) -> bool:
        # put it in X's queue
        file_path = self._create_file(self.x_path, topic, data_bytes)
        return self._monitor_file(file_path, timeout)

    def y_get(self, timeout=None):
        # read from Y's queue
        return self._get_from_dir(self.y_path, timeout)

    def send(self, topic: str, data: bytes, timeout=None) -> bool:
        """

        Args:
            topic:
            data:
            timeout:

        Returns: whether the message is read by peer (if timeout is specified)

        """
        if not self.pipe_path:
            raise BrokenPipeError("pipe is not open")
        return self.put_f(topic, data, timeout)

    def receive(self, timeout=None):
        if not self.pipe_path:
            raise BrokenPipeError("pipe is not open")
        return self.get_f(timeout)

    def close(self):
        pipe_path = self.pipe_path
        self.pipe_path = None
        if pipe_path and os.path.exists(pipe_path):
            try:
                shutil.rmtree(pipe_path)
            except:
                pass
