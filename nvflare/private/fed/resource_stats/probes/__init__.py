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

"""Per-resource capacity probes, one module per technical domain.

Each probe module returns ``None`` when a dimension cannot be established on
this platform/process, so ``collector.probe_capacity`` can omit that member
rather than discard the other observations. CPU/RAM probes also flag unchecked
limits through the collector's shared issue set, retaining usable values.
"""
