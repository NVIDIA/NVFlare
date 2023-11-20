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

import json
from typing import List

import psutil

try:
    import pynvml
except ImportError:
    pynvml = None

from nvflare.apis.fl_constant import FLContextKey, SystemComponents
from nvflare.apis.fl_context import FLContext
from nvflare.private.admin_defs import Message
from nvflare.private.defs import SysCommandTopic
from nvflare.private.fed.client.admin import RequestProcessor


class SysInfoProcessor(RequestProcessor):
    def get_topics(self) -> List[str]:
        return [SysCommandTopic.SYS_INFO]

    def process(self, req: Message, app_ctx) -> Message:
        infos = dict(psutil.virtual_memory()._asdict())
        if pynvml:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                gpu_info = {"gpu_count": device_count}
                for index in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                    gpu_info[f"gpu_device_{index}"] = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
                pynvml.nvmlShutdown()
                infos.update(gpu_info)
            except pynvml.nvml.NVMLError_LibraryNotFound:
                pass

        # docker_image_tag = os.environ.get('DOCKER_IMAGE_TAG', 'N/A')
        # infos.update({'docker_image_tag':docker_image_tag})
        message = Message(topic="reply_" + req.topic, body=json.dumps(infos))
        print("return sys_info")
        print(infos)
        return message


class ReportEnvProcessor(RequestProcessor):
    def get_topics(self) -> [str]:
        return [SysCommandTopic.REPORT_ENV]

    def process(self, req: Message, app_ctx) -> Message:
        engine = app_ctx
        fl_ctx = engine.new_context()
        assert isinstance(fl_ctx, FLContext)
        site_name = fl_ctx.get_identity_name()
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_ROOT)
        secure_mode = fl_ctx.get_prop(FLContextKey.SECURE_MODE)
        fed_client = fl_ctx.get_prop(SystemComponents.FED_CLIENT)
        root_url = ""
        if fed_client:
            cell = fed_client.cell
            root_url = cell.get_root_url_for_child()
        env = {
            "site_name": site_name,
            "workspace": workspace,
            "secure_mode": secure_mode,
            "root_url": root_url,
        }
        message = Message(topic="reply_" + req.topic, body=json.dumps(env))
        return message
