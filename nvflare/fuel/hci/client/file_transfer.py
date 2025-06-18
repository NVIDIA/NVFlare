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
import tempfile
import time
import uuid

import nvflare.fuel.hci.file_transfer_defs as ftd
from nvflare.fuel.hci.client.event import EventType
from nvflare.fuel.hci.cmd_arg_utils import join_args
from nvflare.fuel.hci.proto import MetaKey, ProtoKey
from nvflare.fuel.hci.reg import CommandEntry, CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.utils.zip_utils import split_path, unzip_all_from_file, zip_directory_to_file
from nvflare.lighter.utils import load_private_key_file, sign_folders

from .api_spec import CommandContext, HCIRequester
from .api_status import APIStatus


class _FileSender(HCIRequester):
    def __init__(self, file_name: str):
        self.file_name = file_name

    def send_request(self, api, conn, cmd_ctx):
        result = api.upload_file(self.file_name, conn)
        os.remove(self.file_name)
        return result


class _FileReceiver(HCIRequester):
    def __init__(self, source_fqcn: str, ref_id, file_name: str):
        self.source_fqcn = source_fqcn
        self.ref_id = ref_id
        self.file_name = file_name
        self.num_bytes_received = 0

    def send_request(self, api, conn, cmd_ctx):
        self.num_bytes_received = api.download_file(self.source_fqcn, self.ref_id, self.file_name)
        if self.num_bytes_received is not None:
            cmd_ctx.set_command_result({ProtoKey.STATUS: APIStatus.SUCCESS, ProtoKey.DETAILS: "OK"})
        else:
            cmd_ctx.set_command_result(
                {ProtoKey.STATUS: APIStatus.ERROR_RUNTIME, ProtoKey.DETAILS: "error receiving file"}
            )
        return None


class FileTransferModule(CommandModule):
    """Command module with commands relevant to file transfer."""

    PULL_BINARY_FILE_CMD = "pull_binary_file"

    def __init__(self, upload_dir: str, download_dir: str):
        if not os.path.isdir(upload_dir):
            raise ValueError("upload_dir {} is not a valid dir".format(upload_dir))

        if not os.path.isdir(download_dir):
            raise ValueError("download_dir {} is not a valid dir".format(download_dir))

        self.upload_dir = upload_dir
        self.download_dir = download_dir

        self.cmd_handlers = {
            ftd.PUSH_FOLDER_FQN: self.push_folder,
            ftd.PULL_FOLDER_FQN: self.pull_folder,
        }

    def get_spec(self):
        return CommandModuleSpec(
            name="file_transfer",
            cmd_specs=[
                CommandSpec(
                    name=self.PULL_BINARY_FILE_CMD,
                    description="download one binary files in the download_dir",
                    usage="pull_binary source_fqcn tx_id ref_id folder_name file_name",
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
        Args: cmd_name, source_fqcn, tx_id, ref_id, folder_name, file_name, [end]
        """
        cmd_entry = ctx.get_command_entry()
        if len(args) < 6 or len(args) > 7:
            return {ProtoKey.STATUS: APIStatus.ERROR_SYNTAX, ProtoKey.DETAILS: "usage: {}".format(cmd_entry.usage)}
        source_fqcn = args[1]
        tx_id = args[2]
        ref_id = args[3]
        folder_name = args[4]
        file_name = args[5]
        tx_path = self._tx_path(tx_id, folder_name)
        file_path = os.path.join(tx_path, file_name)
        api = ctx.get_api()
        receiver = _FileReceiver(source_fqcn, ref_id, file_path)
        api.fire_session_event(EventType.BEFORE_DOWNLOAD_FILE, f"downloading {file_name} ...")
        ctx.set_requester(receiver)
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

        files = meta.get(MetaKey.FILES)
        tx_id = meta.get(MetaKey.TX_ID)
        source_fqcn = meta.get(MetaKey.SOURCE_FQCN)
        api.debug(f"received tx_id {tx_id}, file names: {files}")
        if not files:
            return result

        cmd_name = self.PULL_BINARY_FILE_CMD
        error = None
        for i, f in enumerate(files):
            file_name = f[0]
            ref_id = f[1]
            parts = [cmd_name, source_fqcn, tx_id, ref_id, folder_name, file_name]
            if i == len(files) - 1:
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
        sender = _FileSender(out_file)
        ctx.set_requester(sender)
        return api.server_execute(command, cmd_ctx=ctx)
