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

from unittest.mock import Mock, patch

import numpy as np

from nvflare.apis.fl_constant import FilterKey, FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.utils.decomposers import flare_decomposers
from nvflare.apis.utils.task_utils import apply_filters
from nvflare.app_common.decomposers import common_decomposers
from nvflare.app_common.decomposers.numpy_decomposers import NumpyArrayDecomposer
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs import FOBSContextKey, dots
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef


def _make_fl_ctx(cell):
    engine = Mock()
    engine.get_cell.return_value = cell
    fl_ctx = FLContext()
    fl_ctx.put(key=ReservedKey.ENGINE, value=engine, private=True, sticky=False)
    return fl_ctx


def test_apply_filters_does_not_materialize_without_active_filter():
    cell = Mock()
    fl_ctx = _make_fl_ctx(cell)
    value = LazyDownloadRef("server", "ref-1", "T0", dot=dots.NUMPY_DOWNLOAD)
    shareable = Shareable({"weight": value})

    result = apply_filters("task_data_filters", shareable, fl_ctx, {}, "train", FilterKey.IN)

    assert result is shareable
    assert result["weight"] is value
    cell.get_fobs_context.assert_not_called()


def test_apply_filters_materializes_lazy_values_before_filter_process():
    flare_decomposers.register()
    common_decomposers.register()
    fobs.register(NumpyArrayDecomposer)
    cell = Mock()
    cell.get_fobs_context.side_effect = lambda props: {FOBSContextKey.CELL: cell, **props}
    fl_ctx = _make_fl_ctx(cell)
    abort_signal = Signal()
    expected = np.asarray([1.0, 2.0, 3.0])
    shareable = Shareable(
        {
            "weight": LazyDownloadRef(
                fqcn="server",
                ref_id="ref-1",
                item_id="T0",
                dot=dots.NUMPY_DOWNLOAD,
            )
        }
    )
    filter_component = Mock()
    filter_component.process.side_effect = lambda data, _fl_ctx: data

    with patch(
        "nvflare.app_common.decomposers.numpy_decomposers.download_arrays",
        return_value=(None, {"T0": expected}),
    ) as download:
        result = apply_filters(
            "task_data_filters",
            shareable,
            fl_ctx,
            {"train/in": [filter_component]},
            "train",
            FilterKey.IN,
            abort_signal=abort_signal,
        )

    np.testing.assert_array_equal(result["weight"], expected)
    processed = filter_component.process.call_args.args[0]
    np.testing.assert_array_equal(processed["weight"], expected)
    assert fl_ctx.get_prop(FLContextKey.FILTER_DIRECTION) == FilterKey.IN
    assert cell.get_fobs_context.call_args_list[1].kwargs["props"] == {
        FOBSContextKey.PASS_THROUGH: False,
        FOBSContextKey.ABORT_SIGNAL: abort_signal,
    }
    download.assert_called_once()
