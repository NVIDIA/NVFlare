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

"""Framework decomposer registration shared by the external-process CJ and trainer.

Both ends may need to serialize framework-native tensors across the Cell when the declared wire
representation is native or ``RAW``. NumPy and Keras wire representations need only the standard
decomposers. A declared native representation is strict; ``RAW`` remains opportunistic because its
concrete payload type is intentionally unspecified.
"""

from nvflare.client.config import ExchangeFormat

# (module_path, class_name) of each framework's tensor decomposer, mirroring what the legacy
# ex-process trainer registered for ExchangeFormat.PYTORCH (nvflare/client/ex_process/api.py).
_FRAMEWORK_DECOMPOSERS = (("nvflare.app_opt.pt.decomposers", "TensorDecomposer"),)


def register_framework_decomposers(params_exchange_format, server_expected_format, logger=None) -> None:
    """Register decomposers required by the declared Client API exchange pair.

    The Cell normally carries the server representation because trainer-side adaptation happens
    after receive and before send. If either side declares ``RAW``, however, conversion is disabled
    and either representation may cross the Cell. Registration is opportunistic when that pair is
    otherwise framework-agnostic, but required when the other side explicitly declares PyTorch.
    """
    params_exchange_format = ExchangeFormat(params_exchange_format)
    server_expected_format = ExchangeFormat(server_expected_format)
    formats = (params_exchange_format, server_expected_format)
    if ExchangeFormat.RAW in formats:
        # RAW disables adaptation, so either declared representation may cross the Cell.
        should_register = True
        required = ExchangeFormat.PYTORCH in formats
    else:
        # With adaptation enabled, the Cell carries the server representation.
        should_register = server_expected_format == ExchangeFormat.PYTORCH
        required = should_register
    if not should_register:
        return

    from nvflare.fuel.utils import fobs

    for module_name, class_name in _FRAMEWORK_DECOMPOSERS:
        try:
            module = __import__(module_name, fromlist=[class_name])
            fobs.register(getattr(module, class_name))
            if logger is not None:
                logger.debug(f"registered framework decomposer {module_name}.{class_name}")
        except Exception as e:
            if required:
                raise RuntimeError(
                    f"cannot register {module_name}.{class_name} required by the declared "
                    f"{ExchangeFormat.PYTORCH.value} wire format"
                ) from e
            if logger is not None:
                logger.debug(f"framework decomposer {module_name}.{class_name} not registered: {e}")
