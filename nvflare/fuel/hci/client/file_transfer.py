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
from nvflare.fuel.hci.base64_utils import (
    b64str_to_binary_file,
    b64str_to_bytes,
    b64str_to_text_file,
    binary_file_to_b64str,
    text_file_to_b64str,
)
from nvflare.fuel.hci.binary_proto import CT_BINARY, receive_all, send_binary_file
from nvflare.fuel.hci.client.event import EventType
from nvflare.fuel.hci.cmd_arg_utils import join_args
from nvflare.fuel.hci.proto import MetaKey, ProtoKey
from nvflare.fuel.hci.reg import CommandEntry, CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.table import Table
from nvflare.fuel.utils.zip_utils import split_path, unzip_all_from_bytes, unzip_all_from_file, zip_directory_to_file
from nvflare.lighter.utils import load_private_key_file, sign_folders
from nvflare.security.logging import secure_format_exception, secure_log_traceback

from .api_spec import CommandContext, ReceiveBytesFromServer, ReplyProcessor, SendBytesToServer
from .api_status import APIStatus


def _server_cmd_name(name: str):
    return ftd.SERVER_MODULE_NAME + "." + name


class _SendFileToServer(SendBytesToServer):
    def __init__(self, file_name: str):
        self.file_name = file_name

    def send(self, sock, meta: str):
        send_binary_file(sock, self.file_name, meta)
        os.remove(self.file_name)


class _ReceiveFileFromServer(ReceiveBytesFromServer):
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.num_bytes_received = 0

    def receive(self, sock):
        ct, _, tmp_file_name = receive_all(sock)
        if ct != CT_BINARY:
            raise RuntimeError(f"expecting BINARY type {CT_BINARY} but got {ct}")
        if not tmp_file_name:
            raise RuntimeError("nothing received from the server")
        file_stats = os.stat(tmp_file_name)
        self.num_bytes_received = file_stats.st_size
        Path(os.path.dirname(self.file_name)).mkdir(parents=True, exist_ok=True)
        shutil.move(tmp_file_name, self.file_name)


class _DownloadProcessor(ReplyProcessor):
    """Reply processor to handle downloads."""

    def __init__(self, download_dir: str, str_to_file_func):
        self.download_dir = download_dir
        self.str_to_file_func = str_to_file_func
        self.data_received = False
        self.table = None

    def reply_start(self, ctx: CommandContext, reply_json):
        self.data_received = False
        self.table = Table(["file", "size"])

    def reply_done(self, ctx: CommandContext):
        if not self.data_received:
            ctx.set_command_result({"status": APIStatus.ERROR_PROTOCOL, "details": "protocol error - no data received"})
        else:
            command_result = ctx.get_command_result()
            if command_result is None:
                command_result = {}
            command_result["status"] = APIStatus.SUCCESS
            command_result["details"] = self.table
            ctx.set_command_result(command_result)

    def process_table(self, ctx: CommandContext, table: Table):
        try:
            rows = table.rows
            if len(rows) < 1:
                # no data
                ctx.set_command_result({"status": APIStatus.ERROR_PROTOCOL, "details": "protocol error - no file data"})
                return

            for i in range(len(rows)):
                if i == 0:
                    # this is header
                    continue

                row = rows[i]
                if len(row) < 1:
                    ctx.set_command_result(
                        {
                            "status": APIStatus.ERROR_PROTOCOL,
                            "details": "protocol error - missing file name",
                        }
                    )
                    return

                if len(row) < 2:
                    ctx.set_command_result(
                        {
                            "status": APIStatus.ERROR_PROTOCOL,
                            "details": "protocol error - missing file data",
                        }
                    )
                    return

                file_name = row[0]
                encoded_str = row[1]
                full_path = os.path.join(self.download_dir, file_name)
                num_bytes = self.str_to_file_func(encoded_str, full_path)
                self.table.add_row([file_name, str(num_bytes)])
                self.data_received = True
        except Exception as e:
            secure_log_traceback()
            ctx.set_command_result(
                {
                    "status": APIStatus.ERROR_RUNTIME,
                    "details": f"exception processing file: {secure_format_exception(e)}",
                }
            )


class _DownloadFolderProcessor(ReplyProcessor):
    """Reply processor for handling downloading directories."""

    def __init__(self, download_dir: str):
        self.download_dir = download_dir
        self.data_received = False

    def reply_start(self, ctx: CommandContext, reply_json):
        self.data_received = False

    def reply_done(self, ctx: CommandContext):
        if not self.data_received:
            ctx.set_command_result({"status": APIStatus.ERROR_RUNTIME, "details": "protocol error - no data received"})

    def process_error(self, ctx: CommandContext, err: str):
        self.data_received = True
        ctx.set_command_result({"status": APIStatus.ERROR_RUNTIME, "details": err})

    def process_string(self, ctx: CommandContext, item: str):
        try:
            self.data_received = True
            if item.startswith(ftd.DOWNLOAD_URL_MARKER):
                ctx.set_command_result(
                    {
                        "status": APIStatus.SUCCESS,
                        "details": item,
                    }
                )
            else:
                data_bytes = b64str_to_bytes(item)
                unzip_all_from_bytes(data_bytes, self.download_dir)
                ctx.set_command_result(
                    {
                        "status": APIStatus.SUCCESS,
                        "details": "Downloaded to dir {}".format(self.download_dir),
                    }
                )
        except Exception as e:
            secure_log_traceback()
            ctx.set_command_result(
                {
                    "status": APIStatus.ERROR_RUNTIME,
                    "details": f"exception processing reply: {secure_format_exception(e)}",
                }
            )


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
            ftd.DOWNLOAD_FOLDER_FQN: self.download_folder,
            ftd.PULL_BINARY_FQN: self.pull_binary_file,
            ftd.PULL_FOLDER_FQN: self.pull_folder,
        }

    def get_spec(self):
        return CommandModuleSpec(
            name="file_transfer",
            cmd_specs=[
                CommandSpec(
                    name="upload_text",
                    description="upload one or more text files in the upload_dir",
                    usage="upload_text file_name ...",
                    handler_func=self.upload_text_file,
                    visible=False,
                ),
                CommandSpec(
                    name="download_text",
                    description="download one or more text files in the download_dir",
                    usage="download_text file_name ...",
                    handler_func=self.download_text_file,
                    visible=False,
                ),
                CommandSpec(
                    name="upload_binary",
                    description="upload one or more binary files in the upload_dir",
                    usage="upload_binary file_name ...",
                    handler_func=self.upload_binary_file,
                    visible=False,
                ),
                CommandSpec(
                    name="download_binary",
                    description="download one or more binary files in the download_dir",
                    usage="download_binary file_name ...",
                    handler_func=self.download_binary_file,
                    visible=False,
                ),
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
                    name="download_folder",
                    description="download job contents from the server",
                    usage="download_job job_id",
                    handler_func=self.download_folder,
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

    def upload_file(self, args, ctx: CommandContext, cmd_name, file_to_str_func):
        full_cmd_name = _server_cmd_name(cmd_name)
        if len(args) < 2:
            return {"status": APIStatus.ERROR_SYNTAX, "details": "syntax error: missing file names"}

        parts = [full_cmd_name]
        for i in range(1, len(args)):
            file_name = args[i]
            full_path = os.path.join(self.upload_dir, file_name)
            if not os.path.isfile(full_path):
                return {"status": APIStatus.ERROR_RUNTIME, "details": f"no such file: {full_path}"}

            encoded_string = file_to_str_func(full_path)
            parts.append(file_name)
            parts.append(encoded_string)

        command = join_args(parts)
        api = ctx.get_api()
        return api.server_execute(command)

    def upload_text_file(self, args, ctx: CommandContext):
        return self.upload_file(args, ctx, ftd.SERVER_CMD_UPLOAD_TEXT, text_file_to_b64str)

    def upload_binary_file(self, args, ctx: CommandContext):
        return self.upload_file(args, ctx, ftd.SERVER_CMD_UPLOAD_BINARY, binary_file_to_b64str)

    def download_file(self, args, ctx: CommandContext, cmd_name, str_to_file_func):
        full_cmd_name = _server_cmd_name(cmd_name)
        if len(args) < 2:
            return {"status": APIStatus.ERROR_SYNTAX, "details": "syntax error: missing file names"}

        parts = [full_cmd_name]
        for i in range(1, len(args)):
            file_name = args[i]
            parts.append(file_name)

        command = join_args(parts)
        reply_processor = _DownloadProcessor(self.download_dir, str_to_file_func)
        api = ctx.get_api()
        return api.server_execute(command, reply_processor)

    def download_text_file(self, args, ctx: CommandContext):
        return self.download_file(args, ctx, ftd.SERVER_CMD_DOWNLOAD_TEXT, b64str_to_text_file)

    def download_binary_file(self, args, ctx: CommandContext):
        return self.download_file(args, ctx, ftd.SERVER_CMD_DOWNLOAD_BINARY, b64str_to_binary_file)

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
        # is_end = len(args) > 4
        tx_path = self._tx_path(tx_id, folder_name)
        file_path = os.path.join(tx_path, file_name)
        receiver = _ReceiveFileFromServer(file_path)
        api = ctx.get_api()
        api.fire_session_event(EventType.BEFORE_DOWNLOAD_FILE, f"downloading {file_name} ...")
        api = ctx.get_api()
        ctx.set_bytes_receiver(receiver)
        download_start = time.time()
        result = api.server_execute(ctx.get_command(), cmd_ctx=ctx)
        if result.get(ProtoKey.STATUS) != APIStatus.SUCCESS:
            return result
        download_end = time.time()
        api.fire_session_event(
            EventType.AFTER_DOWNLOAD_FILE,
            f"downloaded {file_name} ({receiver.num_bytes_received} bytes) in {download_end-download_start} seconds",
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

    def download_folder(self, args, ctx: CommandContext):
        cmd_entry = ctx.get_command_entry()
        assert isinstance(cmd_entry, CommandEntry)

        if len(args) != 2:
            return {"status": APIStatus.ERROR_SYNTAX, "details": "usage: {}".format(cmd_entry.usage)}
        job_id = args[1]
        parts = [cmd_entry.full_command_name(), job_id]
        command = join_args(parts)
        reply_processor = _DownloadFolderProcessor(self.download_dir)
        api = ctx.get_api()
        return api.server_execute(command, reply_processor)

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
        if not api.insecure:
            # we are not in POC mode
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
