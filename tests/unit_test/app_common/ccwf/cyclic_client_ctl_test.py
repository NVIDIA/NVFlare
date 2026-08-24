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

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.common import Constant
from nvflare.app_common.ccwf.cyclic_client_ctl import CyclicClientController
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef


def test_requires_materialized_result_only_for_learn_task():
    controller = CyclicClientController(learn_task_name="custom_train")

    assert controller.requires_materialized_task_result("custom_train") is True
    assert controller.requires_materialized_task_result("validate") is False


def test_do_learn_materializes_external_result_for_local_cyclic_consumer():
    controller = CyclicClientController(learn_task_name="custom_train")
    controller.me = "site-1"
    controller.shareable_generator = MagicMock()
    controller.shareable_generator.shareable_to_learnable.side_effect = [
        {"weight": 1.0},
        {"weight": 2.0},
    ]
    controller.update_status = MagicMock()
    controller.record_last_result = MagicMock()
    controller.broadcast_final_result = MagicMock()

    lazy_result = Shareable({"weight": LazyDownloadRef("trainer", "ref-1", "T0")})
    materialized_result = Shareable({"weight": 2.0})
    backend = MagicMock()
    backend.execute.return_value = lazy_result
    executor = ClientAPIExecutor(execution_mode="external_process", command="python custom/train.py")
    executor._backend = backend
    controller.learn_executor = executor

    data = Shareable()
    data.set_header(AppConstants.CURRENT_ROUND, 0)
    data.set_header(AppConstants.NUM_ROUNDS, 1)
    data.set_header(Constant.CLIENT_ORDER, ["site-1"])

    engine = MagicMock()
    engine.get_all_components.return_value = {"cyclic_controller": controller}
    engine.get_cell.return_value.get_fqcn.return_value = "site-1.job-1"
    workflow_fl_ctx = FLContext()
    workflow_fl_ctx.put(ReservedKey.ENGINE, engine, private=True, sticky=False)
    workflow_fl_ctx.set_prop(FLContextKey.TASK_NAME, "cyclic_learn", private=True, sticky=False)

    with patch(
        "nvflare.app_common.executors.client_api_executor.materialize_lazy_download_refs",
        return_value=materialized_result,
    ) as materialize:
        controller.do_learn_task("custom_train", data, workflow_fl_ctx, Signal())

    learn_fl_ctx = backend.execute.call_args.args[2]
    assert learn_fl_ctx is not workflow_fl_ctx
    assert learn_fl_ctx.get_prop(FLContextKey.TASK_NAME) == "custom_train"
    assert workflow_fl_ctx.get_prop(FLContextKey.TASK_NAME) == "cyclic_learn"
    assert data.get_header(FLContextKey.TASK_NAME) == "custom_train"
    assert controller.shareable_generator.shareable_to_learnable.call_args_list[1].args[0] is materialized_result
    materialize.assert_called_once()
