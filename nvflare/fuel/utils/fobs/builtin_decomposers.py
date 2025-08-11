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
    "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DataClassDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DictDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DictDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.DictDecomposer",
    "nvflare.fuel.utils.fobs.decomposer.EnumTypeDecomposer",
    "nvflare.fuel.utils.fobs.decomposers.core_decomposers.DatetimeDecomposer",
    "nvflare.fuel.utils.fobs.decomposers.core_decomposers.OrderedDictDecomposer",
    "nvflare.fuel.utils.fobs.decomposers.core_decomposers.SetDecomposer",
    "nvflare.fuel.utils.fobs.decomposers.core_decomposers.TupleDecomposer",
}
