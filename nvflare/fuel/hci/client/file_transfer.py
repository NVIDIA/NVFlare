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
import tempfile
import time
import uuid
from pathlib import Path

import nvflare.fuel.hci.file_transfer_defs as ftd
from nvflare.app_common.streamers.file_streamer import FileStreamer
from nvflare.fuel.hci.client.api import FileWaiter
from nvflare.fuel.hci.client.event import EventType
from nvflare.fuel.hci.cmd_arg_utils import join_args
from nvflare.fuel.hci.proto import MetaKey, ProtoKey
from nvflare.fuel.hci.reg import CommandEntry, CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.utils.zip_utils import split_path, unzip_all_from_file, zip_directory_to_file
from nvflare.lighter.utils import load_private_key_file, sign_folders

from .api_spec import CommandContext, ReceiveBytesFromServer, SendBytesToServer
from .api_status import APIStatus


class _SendFileToServer(SendBytesToServer):
    def __init__(self, file_name: str):
        self.file_name = file_name

    def send(self, sock, meta: str):
        result = sock.stream_file(self.file_name, meta)
        os.remove(self.file_name)
        return result


class _ReceiveFileFromServer(ReceiveBytesFromServer):
    def __init__(self, api, file_name: str, waiter: FileWaiter, progress_timeout=5.0):
        self.api = api
        self.file_name = file_name
        self.waiter = waiter
        self.progress_timeout = progress_timeout
        self.num_bytes_received = 0

    def receive(self, sock):
        result_received = False
        while True:
            if self.waiter.wait(timeout=0.5):
                # wait ended normally
                result_received = True
                break

            # is there any progress?
            if time.time() - self.waiter.last_progress_time > self.progress_timeout:
                # no progress for too long
                break

        self.api.pop_download_waiter(self.waiter.tx_id)
        if not result_received:
            print(f"failed to receive file {self.file_name}: no progress for {self.progress_timeout} seconds")
            return False

        stream_ctx = self.waiter.stream_ctx
        tmp_file_name = FileStreamer.get_file_location(stream_ctx)
        file_stats = os.stat(tmp_file_name)
        self.num_bytes_received = file_stats.st_size
        Path(os.path.dirname(self.file_name)).mkdir(parents=True, exist_ok=True)
        shutil.move(tmp_file_name, self.file_name)
        return True


class FileTransferModule(CommandModule):
    """Command module with commands relevant to file transfer."""

    def __init__(self, upload_dir: str, download_dir: str):
        if not os.path.isdir(upload_dir):
            raise ValueError("upload_dir {} is not a valid dir".format(upload_dir))

        if not os.path.isdir(download_dir):
            raise ValueError("download_dir {} is not a valid dir".format(download_dir))

        self.upload_dir = upload_dir
        self.download_dir = download_dir

        self.cmd_handlers = {
            ftd.PUSH_FOLDER_FQN: self.push_folder,
            ftd.PULL_BINARY_FQN: self.pull_binary_file,
            ftd.PULL_FOLDER_FQN: self.pull_folder,
        }

    def get_spec(self):
        return CommandModuleSpec(
            name="file_transfer",
            cmd_specs=[
                CommandSpec(
                    name="pull_binary",
                    description="download one binary files in the download_dir",
                    usage="pull_binary control_id file_name",
                    handler_func=self.pull_binary_file,
                    visible=False,
                ),
                CommandSpec(
                    name="push_folder",
                    description="Submit application to the server",
                    usage="submit_job job_folder",
                    handler_func=self.push_folder,
                    visible=False,
                ),
                CommandSpec(
                    name="info",
                    description="show folder setup info",
                    usage="info",
                    handler_func=self.info,
                ),
            ],
        )

    def generate_module_spec(self, server_cmd_spec: CommandSpec):
        """
        Generate a new module spec based on a server command

        Args:
            server_cmd_spec:

        Returns:

        """
        # print('generating cmd module for {}'.format(server_cmd_spec.client_cmd))
        if not server_cmd_spec.client_cmd:
            return None

        handler = self.cmd_handlers.get(server_cmd_spec.client_cmd)
        if handler is None:
            print("no cmd handler found for {}".format(server_cmd_spec.client_cmd))
            return None

        return CommandModuleSpec(
            name=server_cmd_spec.scope_name,
            cmd_specs=[
                CommandSpec(
                    name=server_cmd_spec.name,
                    description=server_cmd_spec.description,
                    usage=server_cmd_spec.usage,
                    handler_func=handler,
                    visible=server_cmd_spec.visible,
                )
            ],
        )

    def _tx_path(self, tx_id: str, folder_name: str):
        return os.path.join(self.download_dir, f"{folder_name}__{tx_id}")

    def pull_binary_file(self, args, ctx: CommandContext):
        """
        Args: cmd_name, ctl_id, folder_name, file_name, [end]
        """
        cmd_entry = ctx.get_command_entry()
        if len(args) < 4 or len(args) > 5:
            return {ProtoKey.STATUS: APIStatus.ERROR_SYNTAX, ProtoKey.DETAILS: "usage: {}".format(cmd_entry.usage)}
        tx_id = args[1]
        folder_name = args[2]
        file_name = args[3]
        tx_path = self._tx_path(tx_id, folder_name)
        file_path = os.path.join(tx_path, file_name)
        api = ctx.get_api()
        waiter = api.set_download_waiter(tx_id)
        receiver = _ReceiveFileFromServer(api, file_path, waiter, api.file_download_progress_timeout)
        api.fire_session_event(EventType.BEFORE_DOWNLOAD_FILE, f"downloading {file_name} ...")
        ctx.set_bytes_receiver(receiver)
        download_start = time.time()
        result = api.server_execute(ctx.get_command(), cmd_ctx=ctx)
        if result.get(ProtoKey.STATUS) != APIStatus.SUCCESS:
            return result
        download_end = time.time()
        api.fire_session_event(
            EventType.AFTER_DOWNLOAD_FILE,
            f"downloaded {file_name} ({receiver.num_bytes_received} bytes) in {download_end - download_start} seconds",
        )
        dir_name, ext = os.path.splitext(file_path)
        if ext == ".zip":
            # unzip the file
            api.debug(f"unzipping file {file_path} to {dir_name}")
            os.makedirs(dir_name, exist_ok=True)
            unzip_all_from_file(file_path, dir_name)

            # remove the zip file
            os.remove(file_path)
        return result

    def pull_folder(self, args, ctx: CommandContext):
        cmd_entry = ctx.get_command_entry()
        if len(args) < 2:
            return {ProtoKey.STATUS: APIStatus.ERROR_SYNTAX, ProtoKey.DETAILS: "usage: {}".format(cmd_entry.usage)}
        folder_name = args[1]
        destination_name = folder_name
        if len(args) > 2:
            destination_name = args[2]

        parts = [cmd_entry.full_command_name(), folder_name]
        command = join_args(parts)
        api = ctx.get_api()
        result = api.server_execute(command)
        if result.get(ProtoKey.STATUS) != APIStatus.SUCCESS:
            return result

        meta = result.get(ProtoKey.META)
        if not meta:
            return result

        file_names = meta.get(MetaKey.FILES)
        tx_id = meta.get(MetaKey.TX_ID)
        api.debug(f"received tx_id {tx_id}, file names: {file_names}")
        if not file_names:
            return result

        cmd_name = meta.get(MetaKey.CMD_NAME)

        error = None
        for i, file_name in enumerate(file_names):
            parts = [cmd_name, tx_id, folder_name, file_name]
            if i == len(file_names) - 1:
                # this is the last file
                parts.append("end")

            command = join_args(parts)
            reply = api.do_command(command)
            if reply.get(ProtoKey.STATUS) != APIStatus.SUCCESS:
                error = reply
                break

        if not error:
            tx_path = self._tx_path(tx_id, folder_name)
            destination_path = os.path.join(self.download_dir, destination_name)
            location = self._rename_folder(tx_path, destination_path)
            reply = {
                ProtoKey.STATUS: APIStatus.SUCCESS,
                ProtoKey.DETAILS: f"content downloaded to {location}",
                ProtoKey.META: {MetaKey.LOCATION: location},
            }
        else:
            reply = error
        return reply

    @staticmethod
    def _rename_folder(src: str, destination: str):
        max_tries = 1000
        for i in range(max_tries):
            if i == 0:
                d = destination
            else:
                d = f"{destination}__{i}"
            try:
                os.rename(src, d)
                return d
            except:
                # try next
                pass

        # all rename attempts have failed - keep the original destination name
        return destination

    def info(self, args, ctx: CommandContext):
        msg = f"Local Upload Source: {self.upload_dir}\n"
        msg += f"Local Download Destination: {self.download_dir}\n"
        return {"status": "ok", "details": msg}

    def push_folder(self, args, ctx: CommandContext):
        # upload with binary protocol
        cmd_entry = ctx.get_command_entry()
        assert isinstance(cmd_entry, CommandEntry)
        if len(args) != 2:
            return {"status": APIStatus.ERROR_SYNTAX, "details": "usage: {}".format(cmd_entry.usage)}

        folder_name = args[1]
        if folder_name.endswith("/"):
            folder_name = folder_name.rstrip("/")

        full_path = os.path.join(self.upload_dir, folder_name)
        if not os.path.isdir(full_path):
            return {"status": APIStatus.ERROR_RUNTIME, "details": f"'{full_path}' is not a valid folder."}

        # sign folders and files
        api = ctx.get_api()
        client_key_file_path = api.client_key
        private_key = load_private_key_file(client_key_file_path)
        sign_folders(full_path, private_key, api.client_cert)

        # zip the data
        out_file = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        zip_directory_to_file(self.upload_dir, folder_name, out_file)

        folder_name = split_path(full_path)[1]
        parts = [cmd_entry.full_command_name(), folder_name]
        command = join_args(parts)
        sender = _SendFileToServer(out_file)
        ctx.set_bytes_sender(sender)
        return api.server_execute(command, cmd_ctx=ctx)
