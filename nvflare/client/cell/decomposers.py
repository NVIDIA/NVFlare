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

"""Framework decomposer registration shared by the external_process CJ backend and trainer engine.

Both ends of an external_process job may need to serialize framework-native tensors across the
Cell when the declared server representation is native or ``RAW``. The trainer also serializes
native results before its Client API boundary adapts them when required, so both ends register the
same framework decomposers. Registration is opportunistic: a framework that is not installed is
skipped, and the numpy/FLModel path needs nothing beyond the standard decomposers.
"""

# (module_path, class_name) of each framework's tensor decomposer, mirroring what the legacy
# ex-process trainer registered for ExchangeFormat.PYTORCH (nvflare/client/ex_process/api.py).
_FRAMEWORK_DECOMPOSERS = (("nvflare.app_opt.pt.decomposers", "TensorDecomposer"),)


def register_framework_decomposers(logger=None) -> None:
    """Register available framework tensor decomposers with FOBS. Never raises."""
    from nvflare.fuel.utils import fobs

    for module_name, class_name in _FRAMEWORK_DECOMPOSERS:
        try:
            module = __import__(module_name, fromlist=[class_name])
            fobs.register(getattr(module, class_name))
            if logger is not None:
                logger.debug(f"registered framework decomposer {module_name}.{class_name}")
        except Exception as e:
            if logger is not None:
                logger.debug(f"framework decomposer {module_name}.{class_name} not registered: {e}")
