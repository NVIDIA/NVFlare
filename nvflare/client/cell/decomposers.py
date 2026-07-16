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

"""Register framework decomposers needed by the external-process CJ and trainer."""

from nvflare.client.config import ExchangeFormat

# Mirrors legacy PyTorch ex-process registration.
_FRAMEWORK_DECOMPOSERS = (("nvflare.app_opt.pt.decomposers", "TensorDecomposer"),)


def register_framework_decomposers(params_exchange_format, server_expected_format, logger=None) -> None:
    """Register decomposers for the declared exchange pair.

    ``RAW`` registration is opportunistic because its payload type is unspecified;
    explicitly declared PyTorch formats require registration.
    """
    params_exchange_format = ExchangeFormat(params_exchange_format)
    server_expected_format = ExchangeFormat(server_expected_format)
    formats = (params_exchange_format, server_expected_format)
    if ExchangeFormat.RAW in formats:
        # RAW can carry either representation.
        should_register = True
        required = ExchangeFormat.PYTORCH in formats
    else:
        # Adapted payloads cross Cell in the server representation.
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
