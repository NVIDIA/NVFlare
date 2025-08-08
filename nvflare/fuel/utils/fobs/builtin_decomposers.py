# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.fuel.utils.fobs import dots

# Type and DOT hints for decomposer

# This dictionary holds the mapping of type => decomposer_class_name
BUILTIN_DECOMPOSERS: dict[str, str] = {
    "argparse.Namespace": "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "cma.evolution_strategy.CMAOptions": "cma_decomposer.CMAOptionsDecomposer",
    "cma.logger.CMADataLogger": "cma_decomposer.CMADataLoggerDecomposer",
    "cma.sampler.GaussFullSampler": "cma_decomposer.GaussFullSamplerDecomposer",
    "collections.OrderedDict": "nvflare.fuel.utils.fobs.decomposers.core_decomposers.OrderedDictDecomposer",
    "datetime.datetime": "nvflare.fuel.utils.fobs.decomposers.core_decomposers.DatetimeDecomposer",
    "numpy.float32": "nvflare.app_common.decomposers.numpy_decomposers.Float32ScalarDecomposer",
    "numpy.float64": "nvflare.app_common.decomposers.numpy_decomposers.Float64ScalarDecomposer",
    "numpy.int32": "nvflare.app_common.decomposers.numpy_decomposers.Int32ScalarDecomposer",
    "numpy.int64": "nvflare.app_common.decomposers.numpy_decomposers.Int64ScalarDecomposer",
    "numpy.ndarray": "nvflare.app_common.decomposers.numpy_decomposers.NumpyArrayDecomposer",
    "nvflare.apis.client.Client": "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.apis.dxo.DXO": "nvflare.apis.utils.decomposers.flare_decomposers.DXODecomposer",
    "nvflare.apis.fl_context.FLContext": "nvflare.apis.utils.decomposers.flare_decomposers.ContextDecomposer",
    "nvflare.apis.fl_snapshot.RunSnapshot": "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.apis.shareable.Shareable": "nvflare.fuel.utils.fobs.decomposer.DictDecomposer",
    "nvflare.apis.signal.Signal": "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.apis.workspace.FLModel": "nvflare.app_common.decomposers.common_decomposers.FLModelDecomposer",
    "nvflare.apis.workspace.Workspace": "nvflare.apis.utils.decomposers.flare_decomposers.WorkspaceDecomposer",
    "nvflare.app_common.abstract.fl_model.FLModel": "nvflare.app_common.decomposers.common_decomposers.FLModelDecomposer",
    "nvflare.app_common.abstract.learnable.Learnable": "nvflare.fuel.utils.fobs.decomposer.DictDecomposer",
    "nvflare.app_common.abstract.learnable.ModelLearnable": "nvflare.fuel.utils.fobs.decomposer.DictDecomposer",
    "nvflare.app_common.abstract.statistics_spec.Bin": "nvflare.app_common.statistics.statisitcs_objects_decomposer.BinDecomposer",
    "nvflare.app_common.abstract.statistics_spec.BingRange": "nvflare.app_common.statistics.statisitcs_objects_decomposer.BinRangeDecomposer",
    "nvflare.app_common.abstract.statistics_spec.DataType": "nvflare.app_common.statistics.statisitcs_objects_decomposer.DataTypeDecomposer",
    "nvflare.app_common.abstract.statistics_spec.Feature": "nvflare.app_common.statistics.statisitcs_objects_decomposer.FeatureDecomposer",
    "nvflare.app_common.abstract.statistics_spec.Histogram": "nvflare.app_common.statistics.statisitcs_objects_decomposer.HistogramDecomposer",
    "nvflare.app_common.abstract.statistics_spec.HistogramType": "nvflare.app_common.statistics.statisitcs_objects_decomposer.HistogramTypeDecomposer",
    "nvflare.app_common.abstract.statistics_spec.StatisticConfig": "nvflare.app_common.statistics.statisitcs_objects_decomposer.StatisticConfigDecomposer",
    "nvflare.app_common.widgets.event_recorder._CtxPropReq": "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.app_common.widgets.event_recorder._EventReq": "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.app_common.widgets.event_recorder._EventStats": "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.datum.Datum": "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.datum.DatumRef": "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.private.admin_defs.Message": "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.private.fed.server.run_info.RunInfo": "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.private.fed.server.server_state.Cold2HotState": "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.private.fed.server.server_state.ColdState": "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.private.fed.server.server_state.Hot2ColdState": "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.private.fed.server.server_state.HotState": "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.private.fed.server.server_state.ShutdownState": "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "set": "nvflare.fuel.utils.fobs.decomposers.core_decomposers.SetDecomposer",
    "tenseal.CKKSVector": "nvflare.app_opt.he.decomposers.CKKSVectorDecomposer",
    "tests.unit_test.fuel.utils.fobs.fobs_test.ExampleClass": "tests.unit_test.fuel.utils.fobs.fobs_test.ExampleClassDecomposer",
    "tuple": "nvflare.fuel.utils.fobs.decomposers.core_decomposers.TupleDecomposer",
}

# This dictionary holds the mapping of dot -> decomposer_class_name
BUILTIN_HANDLERS: dict[int, str] = {
    dots.NUMPY_BYTES: "nvflare.app_common.decomposers.numpy_decomposers.NumpyArrayDecomposer",
    dots.NUMPY_FILE: "nvflare.app_common.decomposers.numpy_decomposers.NumpyArrayDecomposer",
}
