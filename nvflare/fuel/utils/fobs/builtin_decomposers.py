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

# Wire type names with a dedicated lazy decomposer. The receiver owns each pairing.
_BUILTIN_DECOMPOSER_MODULE = "nvflare.fuel.utils.fobs.decomposer"
_FLARE_DECOMPOSERS = "nvflare.apis.utils.decomposers.flare_decomposers"
_NUMPY_DECOMPOSERS = "nvflare.app_common.decomposers.numpy_decomposers"
_STATISTICS_TYPES = "nvflare.app_common.abstract.statistics_spec"
_STATISTICS_DECOMPOSERS = "nvflare.app_common.statistics.statisitcs_objects_decomposer"

BUILTIN_TYPE_DECOMPOSERS: dict[str, str] = {
    "nvflare.apis.dxo.DXO": f"{_FLARE_DECOMPOSERS}.DXODecomposer",
    "nvflare.apis.fl_context.FLContext": f"{_FLARE_DECOMPOSERS}.ContextDecomposer",
    "nvflare.apis.workspace.Workspace": f"{_FLARE_DECOMPOSERS}.WorkspaceDecomposer",
    "nvflare.app_common.abstract.fl_model.FLModel": "nvflare.app_common.decomposers.common_decomposers.FLModelDecomposer",
    "numpy.ndarray": f"{_NUMPY_DECOMPOSERS}.NumpyArrayDecomposer",
    "numpy.float32": f"{_NUMPY_DECOMPOSERS}.Float32ScalarDecomposer",
    "numpy.float64": f"{_NUMPY_DECOMPOSERS}.Float64ScalarDecomposer",
    "numpy.int32": f"{_NUMPY_DECOMPOSERS}.Int32ScalarDecomposer",
    "numpy.int64": f"{_NUMPY_DECOMPOSERS}.Int64ScalarDecomposer",
    f"{_STATISTICS_TYPES}.Bin": f"{_STATISTICS_DECOMPOSERS}.BinDecomposer",
    f"{_STATISTICS_TYPES}.BinRange": f"{_STATISTICS_DECOMPOSERS}.BinRangeDecomposer",
    f"{_STATISTICS_TYPES}.DataType": f"{_STATISTICS_DECOMPOSERS}.DataTypeDecomposer",
    f"{_STATISTICS_TYPES}.Feature": f"{_STATISTICS_DECOMPOSERS}.FeatureDecomposer",
    f"{_STATISTICS_TYPES}.Histogram": f"{_STATISTICS_DECOMPOSERS}.HistogramDecomposer",
    f"{_STATISTICS_TYPES}.HistogramType": f"{_STATISTICS_DECOMPOSERS}.HistogramTypeDecomposer",
    f"{_STATISTICS_TYPES}.StatisticConfig": f"{_STATISTICS_DECOMPOSERS}.StatisticConfigDecomposer",
    "nvflare.apis.shareable.Shareable": f"{_BUILTIN_DECOMPOSER_MODULE}.DictDecomposer",
    "nvflare.app_common.abstract.learnable.Learnable": f"{_BUILTIN_DECOMPOSER_MODULE}.DictDecomposer",
    "nvflare.app_common.abstract.model.ModelLearnable": f"{_BUILTIN_DECOMPOSER_MODULE}.DictDecomposer",
    "tenseal.CKKSVector": "nvflare.app_opt.he.decomposers.CKKSVectorDecomposer",
}

# These enums have a dedicated decoder when the statistics codecs are registered
# and otherwise use the generic enum format, including older untagged payloads.
BUILTIN_ENUM_TYPES = {
    f"{_STATISTICS_TYPES}.DataType",
    f"{_STATISTICS_TYPES}.HistogramType",
}

# Optional codecs require an explicit type allowlist entry or local registration.
OPTIONAL_TYPE_DECOMPOSERS: dict[str, str] = {
    "cma.logger.CMADataLogger": "cma_decomposer.CMADataLoggerDecomposer",
    "cma.options_parameters.CMAOptions": "cma_decomposer.CMAOptionsDecomposer",
    "cma.sampler.GaussFullSampler": "cma_decomposer.GaussFullSamplerDecomposer",
    "tenseal.tensors.ckksvector.CKKSVector": "nvflare.app_opt.he.decomposers.CKKSVectorDecomposer",
}

# Other built-in wire types use the enum or data-class decomposer, according to the local type.
BUILTIN_TYPES: set[str] = {
    *BUILTIN_TYPE_DECOMPOSERS,
    # --- Data classes registered in flare_decomposers.py ---
    "nvflare.apis.client.Client",
    "nvflare.apis.fl_snapshot.RunSnapshot",
    "nvflare.apis.signal.Signal",
    "argparse.Namespace",  # Used by WorkspaceDecomposer; schema-free, accept only in trusted contexts
    "nvflare.fuel.utils.fobs.datum.Datum",
    "nvflare.fuel.utils.fobs.datum.DatumRef",
    # --- Data classes registered in private_decomposers.py ---
    "nvflare.private.admin_defs.Message",
    "nvflare.private.fed.server.run_info.RunInfo",
    "nvflare.private.fed.server.server_state.HotState",
    # --- Data classes registered in common_decomposers.py ---
    "nvflare.app_common.widgets.event_recorder._CtxPropReq",
    "nvflare.app_common.widgets.event_recorder._EventReq",
    "nvflare.app_common.widgets.event_recorder._EventStats",
    # --- Enum types auto-registered when core FL objects are serialized ---
    # DataKind is embedded in every DXO (str,Enum — triggers auto-registration)
    "nvflare.apis.dxo.DataKind",
    # ParamsType is embedded in FLModel
    "nvflare.app_common.abstract.fl_model.ParamsType",
    # Connector parameters returned by the admin conns command
    "nvflare.fuel.f3.drivers.driver_params.DriverParams",
    # Analytics / tracking enums used in streaming and metrics
    "nvflare.apis.analytix.AnalyticsDataType",
    "nvflare.apis.analytix.LogWriterName",
    # Found in integration test
    "nvflare.apis.job_def.JobMetaKey",
    "xgboost.core.DataSplitMode",
    # --- Edge web model types (dict-based models for edge device communication) ---
    # These types are serialized via FOBS when edge requests/responses flow through
    # F3/cell message passing (e.g., SelectionRequest containing DeviceInfo).
    # Without these entries, edge external-device tests fail with
    # "Type 'nvflare.edge.web.models.device_info.DeviceInfo' is not allowed."
    "nvflare.edge.web.models.device_info.DeviceInfo",
    "nvflare.edge.web.models.user_info.UserInfo",
    "nvflare.edge.web.models.capabilities.Capabilities",
    "nvflare.edge.web.models.job_request.JobRequest",
    "nvflare.edge.web.models.job_response.JobResponse",
    "nvflare.edge.web.models.task_request.TaskRequest",
    "nvflare.edge.web.models.task_response.TaskResponse",
    "nvflare.edge.web.models.selection_request.SelectionRequest",
    "nvflare.edge.web.models.selection_response.SelectionResponse",
    "nvflare.edge.web.models.result_report.ResultReport",
    "nvflare.edge.web.models.result_response.ResultResponse",
    "nvflare.edge.web.models.error_response.ErrorResponse",
}
