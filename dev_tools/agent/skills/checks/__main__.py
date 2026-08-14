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

try:
    from .cli import main
except ImportError as e:
    # Only fall back for the script-vs-package case (relative import with no
    # parent package: e.name is None). A missing third-party dep (e.g. PyYAML,
    # e.name == "yaml") must re-raise with its real message instead of being
    # masked as "No module named 'cli'".
    if e.name is not None:
        raise
    from cli import main

raise SystemExit(main())
