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

"""CVM Builder: generic confidential VM images and sealed application vaults.

The standalone ``cvm`` namespace under this directory is also importable as
``nvflare.lighter.cc.image_builder.cvm`` and exposed through the
``nvflare-cvmctl`` console script. Guest images and deliveries still receive
only the ``cvm`` payload subsets listed in ``cvm/build/payload.py``.
"""
