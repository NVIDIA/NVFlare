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

import copy
from typing import Any, Type

import cma
import numpy as np
from cma import CMADataLogger, CMAOptions
from cma.constraints_handler import BoundNone
from cma.evolution_strategy import _CMAParameters, _CMASolutionDict_functional, _CMAStopDict
from cma.optimization_tools import BestSolution
from cma.recombination_weights import RecombinationWeights
from cma.sampler import GaussFullSampler
from cma.sigma_adaptation import CMAAdaptSigmaCSA
from cma.transformations import DiagonalDecoding, GenoPheno
from cma.utilities.utils import BlancClass, DictFromTagsInString, ElapsedWCTime, MoreToWrite, SolutionDict

from nvflare.app_common.decomposers.numpy_decomposers import Float64ScalarDecomposer, NumpyArrayDecomposer
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs import Decomposer
from nvflare.fuel.utils.fobs.datum import DatumManager


class GaussFullSamplerDecomposer(Decomposer):
    """A decomposer for serializing and deserializing GaussFullSampler objects.

    This decomposer handles the conversion of GaussFullSampler instances to and from a serializable format.
    It removes non-serializable function attributes and restores them during recomposition.

    Attributes:
        None

    Methods:
        supported_type(): Returns the type this decomposer supports
        decompose(): Converts a GaussFullSampler to a serializable format
        recompose(): Reconstructs a GaussFullSampler from serialized data
    """

    def supported_type(self) -> Type[GaussFullSampler]:
        return GaussFullSampler

    def decompose(self, target: GaussFullSampler, manager: DatumManager = None) -> Any:
        target = copy.deepcopy(target)
        members = vars(target)
        # The functions can't be serialized
        if "randn" in members:
            del members["randn"]
        if "eigenmethod" in members:
            del members["eigenmethod"]
        return members

    def recompose(self, data: dict, manager: DatumManager = None) -> GaussFullSampler:
        instance = GaussFullSampler.__new__(GaussFullSampler)

        # Recreate the removed function fields
        data["randn"] = np.random.randn
        data["eigenmethod"] = np.linalg.eigh
        instance.__dict__.update(data)
        return instance


class CMAOptionsDecomposer(Decomposer):
    """A decomposer for serializing and deserializing CMAOptions objects.

    This decomposer handles the conversion of CMAOptions instances to and from a serializable format.
    It removes non-serializable attributes and restores them during recomposition.

    Attributes:
        None

    Methods:
        supported_type(): Returns the type this decomposer supports
        decompose(): Converts a CMAOptions to a serializable format
        recompose(): Reconstructs a CMAOptions from serialized data
    """

    def supported_type(self) -> Type[GaussFullSampler]:
        return CMAOptions

    def decompose(self, target: CMAOptions, manager: DatumManager = None) -> Any:
        data = {}
        data.update(target)

        fields_to_remove = ["CMA_eigenmethod", "CMA_mirrors", "randn", "BoundaryHandler", "is_feasible"]

        for field in fields_to_remove:
            if field in data:
                del data[field]

        return data

    def recompose(self, data: dict, manager: DatumManager = None) -> GaussFullSampler:
        instance = CMAOptions()
        instance["CMA_eigenmethod"] = np.linalg.eigh
        instance.update(data)
        return instance


class CMADataLoggerDecomposer(Decomposer):
    """A decomposer for serializing and deserializing CMADataLogger objects.

    This decomposer handles the conversion of CMADataLogger instances to and from a serializable format.
    It removes circular references and restores the object structure during recomposition.

    Attributes:
        None

    Methods:
        supported_type(): Returns the type this decomposer supports
        decompose(): Converts a CMADataLogger to a serializable format
        recompose(): Reconstructs a CMADataLogger from serialized data
    """

    def supported_type(self) -> Type[CMADataLogger]:
        return CMADataLogger

    def decompose(self, target: GaussFullSampler, manager: DatumManager = None) -> Any:
        target = copy.deepcopy(target)
        members = vars(target)

        # This field causes a circular reference, FOBS doesn't support this.
        if "es" in members:
            del members["es"]

        return members

    def recompose(self, data: dict, manager: DatumManager = None) -> CMADataLogger:
        instance = CMADataLogger.__new__(CMADataLogger)
        instance.__dict__.update(data)
        return instance


def register_decomposers():
    fobs.register(NumpyArrayDecomposer)
    fobs.register(Float64ScalarDecomposer)
    fobs.register(GaussFullSamplerDecomposer)
    fobs.register(CMAOptionsDecomposer)
    fobs.register(CMADataLoggerDecomposer)
    fobs.register_data_classes(
        cma.CMAEvolutionStrategy,
        GenoPheno,
        SolutionDict,
        BoundNone,
        _CMAParameters,
        RecombinationWeights,
        CMAAdaptSigmaCSA,
        DiagonalDecoding,
        _CMASolutionDict_functional,
        BestSolution,
        BlancClass,
        CMADataLogger,
        DictFromTagsInString,
        ElapsedWCTime,
        _CMAStopDict,
        MoreToWrite,
    )


if __name__ == "__main__":

    register_decomposers()

    es = cma.CMAEvolutionStrategy(4 * [5], 10, dict(ftarget=1e-9, seed=5))

    buffer = fobs.dumps(es)
    print(f"Encoded size: {len(buffer)}")

    new_es = fobs.loads(buffer)
    new_es.logger.register(new_es)

    print(new_es)
