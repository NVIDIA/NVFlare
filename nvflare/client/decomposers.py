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

"""Register framework decomposers needed by Client API runtimes."""

from nvflare.client.config import ExchangeFormat

# Mirrors legacy PyTorch ex-process registration.
_FRAMEWORK_DECOMPOSERS = (("nvflare.app_opt.pt.decomposers", "TensorDecomposer"),)


def register_framework_decomposers(params_exchange_format, server_expected_format, logger=None) -> None:
    """Register decomposers for the declared exchange pair.

    Optional framework support is registered only when the declared pair permits
    PyTorch tensors on the wire. RAW alone does not opt into PyTorch serialization.
    """
    params_exchange_format = ExchangeFormat(params_exchange_format)
    server_expected_format = ExchangeFormat(server_expected_format)
    # TensorDecomposer is required whenever the trainer-side representation
    # is PyTorch. The server-side representation may still be NumPy because
    # parameter conversion can happen after the trainer-side FOBS encoding.
    #
    # In particular, PYTORCH -> NUMPY must register TensorDecomposer because
    # large tensors/download references use DOT TENSOR_DOWNLOAD (6).
    pytorch_wire_format = params_exchange_format == ExchangeFormat.PYTORCH

    if not pytorch_wire_format:
        return

    from nvflare.fuel.utils import fobs

    for module_name, class_name in _FRAMEWORK_DECOMPOSERS:
        try:
            module = __import__(module_name, fromlist=[class_name])
            fobs.register(getattr(module, class_name))
            if logger is not None:
                logger.debug(f"registered framework decomposer {module_name}.{class_name}")
        except Exception as e:
            raise RuntimeError(
                f"cannot register {module_name}.{class_name} required by the declared "
                f"{ExchangeFormat.PYTORCH.value} wire format"
            ) from e
