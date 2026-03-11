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

# This set holds the names of all the built-in decomposers
BUILTIN_DECOMPOSERS: set[str] = {
    "cma_decomposer.CMADataLoggerDecomposer",
    "cma_decomposer.CMAOptionsDecomposer",
    "cma_decomposer.GaussFullSamplerDecomposer",
    "nvflare.apis.utils.decomposers.flare_decomposers.ContextDecomposer",
    "nvflare.apis.utils.decomposers.flare_decomposers.DXODecomposer",
    "nvflare.apis.utils.decomposers.flare_decomposers.WorkspaceDecomposer",
    "nvflare.app_common.decomposers.common_decomposers.FLModelDecomposer",
    "nvflare.app_common.decomposers.numpy_decomposers.Float32ScalarDecomposer",
    "nvflare.app_common.decomposers.numpy_decomposers.Float64ScalarDecomposer",
    "nvflare.app_common.decomposers.numpy_decomposers.Int32ScalarDecomposer",
    "nvflare.app_common.decomposers.numpy_decomposers.Int64ScalarDecomposer",
    "nvflare.app_common.decomposers.numpy_decomposers.NumpyArrayDecomposer",
    "nvflare.app_common.statistics.statisitcs_objects_decomposer.BinDecomposer",
    "nvflare.app_common.statistics.statisitcs_objects_decomposer.BinRangeDecomposer",
    "nvflare.app_common.statistics.statisitcs_objects_decomposer.DataTypeDecomposer",
    "nvflare.app_common.statistics.statisitcs_objects_decomposer.FeatureDecomposer",
    "nvflare.app_common.statistics.statisitcs_objects_decomposer.HistogramDecomposer",
    "nvflare.app_common.statistics.statisitcs_objects_decomposer.HistogramTypeDecomposer",
    "nvflare.app_common.statistics.statisitcs_objects_decomposer.StatisticConfigDecomposer",
    "nvflare.app_opt.he.decomposers.CKKSVectorDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DictDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.EnumTypeDecomposer",
    "nvflare.fuel.utils.fobs.decomposers.core_decomposers.DatetimeDecomposer",
    "nvflare.fuel.utils.fobs.decomposers.core_decomposers.OrderedDictDecomposer",
    "nvflare.fuel.utils.fobs.decomposers.core_decomposers.SetDecomposer",
    "nvflare.fuel.utils.fobs.decomposers.core_decomposers.TupleDecomposer",
}

# This set holds the fully-qualified names of all built-in types that are allowed
# to be deserialized via generic decomposers (DataClassDecomposer / EnumTypeDecomposer)
# even before an explicit register_data_classes() / register_enum_types() call.
# It pre-populates the type_name whitelist in fobs.py and acts as a static allowlist
# to prevent arbitrary class loading (RCE via deserialization).
BUILTIN_TYPES: set[str] = {
    # --- Types handled by custom built-in decomposers ---
    "nvflare.apis.dxo.DXO",
    "nvflare.apis.fl_context.FLContext",
    "nvflare.apis.workspace.Workspace",
    "nvflare.app_common.abstract.fl_model.FLModel",
    "numpy.ndarray",
    "numpy.float32",
    "numpy.float64",
    "numpy.int32",
    "numpy.int64",
    "nvflare.app_common.abstract.statistics_spec.Bin",
    "nvflare.app_common.abstract.statistics_spec.BinRange",
    "nvflare.app_common.abstract.statistics_spec.DataType",
    "nvflare.app_common.abstract.statistics_spec.Feature",
    "nvflare.app_common.abstract.statistics_spec.Histogram",
    "nvflare.app_common.abstract.statistics_spec.HistogramType",
    "nvflare.app_common.abstract.statistics_spec.StatisticConfig",
    "nvflare.apis.shareable.Shareable",
    "nvflare.app_common.abstract.learnable.Learnable",
    "nvflare.app_common.abstract.model.ModelLearnable",
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
    "nvflare.private.fed.server.server_state.ColdState",
    "nvflare.private.fed.server.server_state.Hot2ColdState",
    "nvflare.private.fed.server.server_state.Cold2HotState",
    "nvflare.private.fed.server.server_state.ShutdownState",
    # --- Data classes registered in common_decomposers.py ---
    "nvflare.app_common.widgets.event_recorder._CtxPropReq",
    "nvflare.app_common.widgets.event_recorder._EventReq",
    "nvflare.app_common.widgets.event_recorder._EventStats",
    # --- Optional HE type handled by CKKSVectorDecomposer (tenseal integration) ---
    # tenseal is an optional dependency; this entry ensures reset() → deserialize works
    # for HE payloads when CKKSVectorDecomposer has not been explicitly re-registered.
    "tenseal.CKKSVector",
    # --- Enum types auto-registered when core FL objects are serialized ---
    # DataKind is embedded in every DXO (str,Enum — triggers auto-registration)
    "nvflare.apis.dxo.DataKind",
    # ParamsType is embedded in FLModel
    "nvflare.app_common.abstract.fl_model.ParamsType",
    # Analytics / tracking enums used in streaming and metrics
    "nvflare.apis.analytix.AnalyticsDataType",
    "nvflare.apis.analytix.LogWriterName",
}
