# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Tuple

from nvflare.fuel.utils.attributes_exportable import ExportMode
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.file_accessor import FileAccessor
from nvflare.fuel.utils.pipe.file_name_utils import file_name_to_message, message_to_file_name
from nvflare.fuel.utils.pipe.fobs_file_accessor import FobsFileAccessor
from nvflare.fuel.utils.pipe.pipe import Message, Pipe, Topic
from nvflare.fuel.utils.validation_utils import check_object_type, check_positive_number, check_str


class FilePipe(Pipe):
    def __init__(self, mode: Mode, root_path: str, file_check_interval=0.1):
        """Implementation of communication through the file system.

        Args:
            mode (Mode): Mode of the endpoint. A pipe has two endpoints.
                An endpoint can be either the one that initiates communication or the one listening.
            root_path (str): root path for this file pipe, folders and files will be created under this root_path
                for communication.
            file_check_interval (float): how often should to check the file exists.
        """
        super().__init__(mode=mode)
        check_positive_number("file_check_interval", file_check_interval)
        check_str("root_path", root_path)

        self._remove_root = False
        if not os.path.exists(root_path):
            try:
                # create the root path
                os.makedirs(root_path)
                self._remove_root = True
            except Exception:
                pass

        self.root_path = root_path
        self.file_check_interval = file_check_interval
        self.pipe_path = None
        self.x_path = None
        self.y_path = None
        self.t_path = None

        if self.mode == Mode.ACTIVE:
            self.get_f = self.x_get
            self.put_f = self.x_put
        elif self.mode == Mode.PASSIVE:
            self.get_f = self.y_get
            self.put_f = self.y_put

        self.accessor = FobsFileAccessor()  # default

    def set_file_accessor(self, accessor: FileAccessor):
        """Sets the file accessor to be used by the pipe.
        The default file accessor is FobsFileAccessor.

        Args:
            accessor: the accessor to be used.
        """
        check_object_type("accessor", accessor, FileAccessor)
        self.accessor = accessor

    @staticmethod
    def _make_dir(path):
        try:
            os.mkdir(path)
        except FileExistsError:
            # this is okay
            pass

    def open(self, name: str):
        if not self.accessor:
            raise RuntimeError("File accessor is not set. Make sure to set a FileAccessor before opening the pipe")

        pipe_path = os.path.join(self.root_path, name)

        if not os.path.exists(pipe_path):
            self._make_dir(pipe_path)

        x_path = os.path.join(pipe_path, "x")
        if not os.path.exists(x_path):
            self._make_dir(x_path)

        y_path = os.path.join(pipe_path, "y")
        if not os.path.exists(y_path):
            self._make_dir(y_path)

        t_path = os.path.join(pipe_path, "t")
        if not os.path.exists(t_path):
            self._make_dir(t_path)

        self.pipe_path = pipe_path
        self.x_path = x_path
        self.y_path = y_path
        self.t_path = t_path

    @staticmethod
    def _clear_dir(p: str):
        file_list = os.listdir(p)
        if file_list:
            for f in file_list:
                try:
                    os.remove(os.path.join(p, f))
                except FileNotFoundError:
                    pass

    def _create_file(self, to_dir: str, msg: Message) -> str:
        file_name = message_to_file_name(msg)
        file_path = os.path.join(to_dir, file_name)

        tmp_path = os.path.join(self.t_path, file_name)
        if not self.pipe_path:
            raise BrokenPipeError("pipe broken")
        try:
            self.accessor.write(msg.data, tmp_path)
            os.rename(tmp_path, file_path)
        except FileNotFoundError:
            raise BrokenPipeError("pipe closed")
        return file_path

    def clear(self):
        self._clear_dir(self.x_path)
        self._clear_dir(self.y_path)
        self._clear_dir(self.t_path)

    def _monitor_file(self, file_path: str, timeout=None) -> bool:
        """Monitors the file until it's read-and-removed by peer, or timed out.

        Args:
            file_path: the path to be monitored
            timeout: how long to wait for timeout

        Returns:
            whether the file has been read and removed
        """
        start = time.time()
        while True:
            if not self.pipe_path:
                raise BrokenPipeError("pipe broken")

            if not os.path.exists(file_path):
                return True
            if timeout and time.time() - start > timeout:
                # timed out - try to delete the file
                try:
                    os.remove(file_path)
                except FileNotFoundError:
                    # the file is read by the peer!
                    return True
                return False
            time.sleep(self.file_check_interval)

    def x_put(self, msg: Message, timeout) -> bool:
        """

        Args:
            msg:
            timeout:

        Returns: whether file is read by the peer

        """
        # put it in Y's queue
        file_path = self._create_file(self.y_path, msg)
        return self._monitor_file(file_path, timeout)

    def _read_file(self, file_path: str):
        # since reading file may take time and another process may try to delete the file
        # we move the file to a temp name before reading it
        file_name = os.path.basename(file_path)
        msg = file_name_to_message(file_name)
        tmp_path = os.path.join(self.t_path, file_name)
        try:
            create_time = os.path.getctime(file_path)
            os.rename(file_path, tmp_path)
            data = self.accessor.read(tmp_path)
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)  # remove this file
            elif os.path.isdir(tmp_path):
                shutil.rmtree(tmp_path)
            else:
                raise RuntimeError(f"cannot remove unsupported path: '{tmp_path}'")
            msg.data = data
            msg.sent_time = create_time
            msg.received_time = time.time()
            return msg
        except FileNotFoundError:
            raise BrokenPipeError("pipe closed")

    def _get_next(self, from_dir: str):
        try:
            files = os.listdir(from_dir)
        except Exception:
            raise BrokenPipeError(f"error reading from {from_dir}")

        if files:
            files = [os.path.join(from_dir, f) for f in files]
            files.sort(key=os.path.getmtime, reverse=False)
            file_path = files[0]
            return self._read_file(file_path)
        else:
            return None

    def _get_from_dir(self, from_dir: str, timeout=None):
        if not timeout or timeout <= 0:
            return self._get_next(from_dir)

        start = time.time()
        while True:
            msg = self._get_next(from_dir)
            if msg:
                return msg

            if time.time() - start >= timeout:
                break
            time.sleep(self.file_check_interval)

        return None

    def x_get(self, timeout=None):
        # read from X's queue
        return self._get_from_dir(self.x_path, timeout)

    def y_put(self, msg: Message, timeout) -> bool:
        # put it in X's queue
        file_path = self._create_file(self.x_path, msg)
        return self._monitor_file(file_path, timeout)

    def y_get(self, timeout=None):
        # read from Y's queue
        return self._get_from_dir(self.y_path, timeout)

    def send(self, msg: Message, timeout=None) -> bool:
        """Sends the specified message to the peer.

        Args:
            msg: the message to be sent
            timeout: if specified, number of secs to wait for the peer to read the message.
                If not specified, wait indefinitely.

        Returns:
            Whether the message is read by the peer.

        """
        if not self.pipe_path:
            raise BrokenPipeError("pipe is not open")

        if not timeout and msg.topic in [Topic.END, Topic.ABORT, Topic.HEARTBEAT]:
            timeout = 5.0

        return self.put_f(msg, timeout)

    def receive(self, timeout=None):
        if not self.pipe_path:
            raise BrokenPipeError("pipe is not open")
        return self.get_f(timeout)

    def close(self):
        pipe_path = self.pipe_path
        self.pipe_path = None
        if self.mode == Mode.PASSIVE:
            if pipe_path and os.path.exists(pipe_path):
                shutil.rmtree(pipe_path, ignore_errors=True)
        if self._remove_root and os.path.exists(self.root_path):
            shutil.rmtree(self.root_path, ignore_errors=True)

    def can_resend(self) -> bool:
        return False

    def export(self, export_mode: str) -> Tuple[str, dict]:
        if export_mode == ExportMode.SELF:
            mode = self.mode
        else:
            mode = Mode.ACTIVE if self.mode == Mode.PASSIVE else Mode.PASSIVE

        export_args = {"mode": mode, "root_path": self.root_path}
        return f"{self.__module__}.{self.__class__.__name__}", export_args
