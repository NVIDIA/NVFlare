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

from nvflare.apis.fl_context import FLContext
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe
from nvflare.fuel.utils.pipe.file_pipe import FilePipe
from nvflare.fuel.utils.pipe.pipe import Pipe
from nvflare.fuel.utils.validation_utils import check_object_type


class Piper:
    def __init__(
        self,
        pipe_id: str,
    ):
        """Piper class will init the pipe of this end and return necessary information for external pipe.

        Args:
            piper_id (str): id of the Piper
            pipe_id (str): the id to be used to get the Pipe component from engine.
        """
        self._pipe = None
        self._pipe_id = pipe_id

    def get_external_pipe_class(self) -> str:
        if isinstance(self._pipe, FilePipe):
            return "FilePipe"
        elif isinstance(self._pipe, CellPipe):
            return "CellPipe"
        else:
            raise RuntimeError(f"Pipe ({self._pipe_id}) of type ({type(self._pipe)}) is not supported")

    def get_external_pipe_args(self, fl_ctx: FLContext) -> dict:
        args = {}
        if isinstance(self._pipe, FilePipe):
            args["root_path"] = self._pipe.root_path
            args["mode"] = Mode.ACTIVE if self._pipe.mode == Mode.PASSIVE else Mode.PASSIVE
        elif isinstance(self._pipe, CellPipe):
            args["mode"] = Mode.ACTIVE if self._pipe.mode == Mode.PASSIVE else Mode.PASSIVE
            args["site_name"] = fl_ctx.get_identity_name()
            args["token"] = self._pipe.token
            args["root_url"] = self._pipe.cell.core_cell.root_url
            args["secure_mode"] = self._pipe.cell.core_cell.secure
            args["workspace_dir"] = fl_ctx.get_engine().get_workspace().get_root_dir()
        else:
            raise RuntimeError(f"Pipe ({self._pipe_id}) of type ({type(self._pipe)}) is not supported")
        return args

    def init_pipe(self, fl_ctx: FLContext) -> None:
        engine = fl_ctx.get_engine()

        # gets Pipe using _pipe_id
        pipe = engine.get_component(self._pipe_id)
        check_object_type(self._pipe_id, pipe, Pipe)

        if isinstance(pipe, FilePipe):
            if pipe.root_path is None or pipe.root_path == "":
                app_dir = engine.get_workspace().get_app_dir(fl_ctx.get_job_id())
                pipe.root_path = os.path.abspath(app_dir)
        elif isinstance(pipe, CellPipe):
            if pipe.root_url == "":
                pipe.root_url = engine.get_cell().core_cell.root_url
            if pipe.site_name == "":
                pipe.site_name = fl_ctx.get_identity_name()
            if pipe.workspace_dir == "":
                pipe.workspace_dir = engine.get_workspace().get_root_dir()
            pipe.secure = engine.get_cell().core_cell.secure
        else:
            raise RuntimeError(f"Pipe ({self._pipe_id}) of type ({type(pipe)}) is not supported")

        self._pipe = pipe
