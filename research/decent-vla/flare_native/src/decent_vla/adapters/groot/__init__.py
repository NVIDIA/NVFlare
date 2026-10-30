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

"""groot adapter — bridges NVIDIA Isaac-GR00T N1.7 to decent-vla.

Drives the ``Gr00tN1d7`` model (Cosmos-Reason2-2B / Qwen3-VL backbone) for
federated **LoRA** finetuning of the foundation model on LIBERO, starting from
the base checkpoint ``nvidia/GR00T-N1.7-3B`` with embodiment ``LIBERO_PANDA``.

Requires Isaac-GR00T (cloned to ``third_party/Isaac-GR00T``) plus its
dependency stack (Python 3.10, transformers 4.57.3, torch 2.7.1, peft) in a
dedicated env — see ``scripts/setup_groot.sh``. The decent-vla ``[groot]``
extra is intentionally empty (``pyproject.toml``): GR00T is installed
out-of-band, not as a pip dependency.

Imports cleanly without gr00t installed (heavy imports are function-local).
"""

from __future__ import annotations

from decent_vla.adapters.groot.backend import GrootBackend

__all__ = ["GrootBackend"]
