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


class CatalogSourceError(Exception):
    """A safe, dataset-scoped failure. The message must contain no paths and no source text."""

    def __init__(self, safe_message: str, *, budget: bool = False):
        super().__init__(safe_message)
        # True: a budget stop, which is no evidence about the source.
        # False: the source is absent, inaccessible or invalid.
        self.budget = budget
