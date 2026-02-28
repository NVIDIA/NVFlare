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

"""Auto-register all available model backends.

Each import triggers the ``register_backend()`` call at module level.
Backends whose dependencies are missing are silently skipped so that
a site running only BLIP doesn't need the ``janus`` package installed,
and vice versa.
"""

# Always available (transformers + peft)
from . import blip_backend  # noqa: F401

# Optional - requires `pip install -e .` from the Janus repo
try:
    from . import januspro_backend  # noqa: F401
except ImportError:
    pass
