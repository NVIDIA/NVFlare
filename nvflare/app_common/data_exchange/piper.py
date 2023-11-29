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

from nvflare.apis.fl_context import FLContext
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe
from nvflare.fuel.utils.pipe.file_pipe import FilePipe

EXTERNAL_CLASS = "external_class"
EXTERNAL_ARGS_MAPPING = "external_args_mapping"


class Piper:
    PIPE_CLASSES = {
        FilePipe: {
            EXTERNAL_CLASS: "FilePipe",
            EXTERNAL_ARGS_MAPPING: {
                "root_path": lambda pipe, fl_ctx: pipe.root_path,
                "mode": lambda pipe, fl_ctx: Mode.ACTIVE if pipe.mode == Mode.PASSIVE else Mode.PASSIVE,
            },
        },
        CellPipe: {
            EXTERNAL_CLASS: "CellPipe",
            EXTERNAL_ARGS_MAPPING: {
                "mode": lambda pipe, fl_ctx: Mode.ACTIVE if pipe.mode == Mode.PASSIVE else Mode.PASSIVE,
                "site_name": lambda pipe, fl_ctx: fl_ctx.get_identity_name(),
                "token": lambda pipe, fl_ctx: pipe.token,
                "root_url": lambda pipe, fl_ctx: pipe.cell.get_root_url_for_child(),
                "secure_mode": lambda pipe, fl_ctx: pipe.cell.core_cell.secure,
                "workspace_dir": lambda pipe, fl_ctx: fl_ctx.get_engine().get_workspace().get_root_dir(),
            },
        },
    }

    @classmethod
    def get_external_pipe_class(cls, pipe_id: str, fl_ctx: FLContext) -> str:
        pipe = fl_ctx.get_engine().get_component(pipe_id)
        if pipe is None:
            raise RuntimeError(f"Pipe ({pipe_id}) can't be found in components.")
        pipe_class = type(pipe)
        for pipe_type, info in cls.PIPE_CLASSES.items():
            if issubclass(pipe_class, pipe_type):
                return info[EXTERNAL_CLASS]
        raise RuntimeError(f"Pipe of type ({pipe_class}) is not supported")

    @classmethod
    def get_external_pipe_args(cls, pipe_id: str, fl_ctx: FLContext) -> dict:
        pipe = fl_ctx.get_engine().get_component(pipe_id)
        if pipe is None:
            raise RuntimeError(f"Pipe ({pipe_id}) can't be found in components.")
        pipe_class = type(pipe)
        for pipe_type, info in cls.PIPE_CLASSES.items():
            if issubclass(pipe_class, pipe_type):
                args_mapping = info[EXTERNAL_ARGS_MAPPING]
                return {arg: mapper(pipe, fl_ctx) for arg, mapper in args_mapping.items()}
        raise RuntimeError(f"Pipe of type ({pipe_class}) is not supported")
