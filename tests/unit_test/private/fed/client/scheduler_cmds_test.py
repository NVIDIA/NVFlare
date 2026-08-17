# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock, patch

from nvflare.apis.fl_constant import ReturnCode, SystemComponents
from nvflare.apis.fl_context import FLContext
from nvflare.apis.resource_manager_spec import ResourceManagerSpec
from nvflare.private.admin_defs import Message
from nvflare.private.defs import RequestHeader, TrainingTopic
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec
from nvflare.private.fed.client.scheduler_cmds import CheckResourceProcessor
from nvflare.private.scheduler_constants import ShareableHeader


def test_check_resource_processor_keeps_resource_manager_error_local():
    resource_manager = MagicMock(spec=ResourceManagerSpec)
    resource_manager.check_resources.side_effect = ValueError("resource_requirement is missing num_gpu_key num_of_gpus")
    engine = MagicMock(spec=ClientEngineInternalSpec)
    engine.get_component.return_value = resource_manager
    engine.new_context.return_value.__enter__.return_value = FLContext()
    request = Message(topic=TrainingTopic.CHECK_RESOURCE, body={"license": 2})
    request.set_header(RequestHeader.JOB_ID, "job-1")

    with (
        patch("nvflare.private.fed.client.scheduler_cmds.logger") as logger,
        patch("nvflare.private.fed.client.scheduler_cmds.secure_format_exception", return_value="sensitive detail"),
    ):
        reply = CheckResourceProcessor().process(request, engine)

    assert reply.body.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert not reply.body.get_header(ShareableHeader.IS_RESOURCE_ENOUGH)
    assert (
        reply.body.get_header(ShareableHeader.RESOURCE_RESERVE_TOKEN)
        == "resource manager raised an exception; see site log"
    )
    engine.get_component.assert_called_once_with(SystemComponents.RESOURCE_MANAGER)
    logger.error.assert_called_once_with("Job job-1: resource check failed: sensitive detail")
