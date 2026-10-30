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

"""Checkpoint selection shared by native persistence and evaluation."""

from pathlib import Path


def latest_server_checkpoint(ckpt_dir: Path) -> Path | None:
    paths = [p for p in Path(ckpt_dir).glob("server_round_*.pt") if p.stem.rsplit("_", 1)[-1].isdigit()]
    return max(paths, key=lambda p: int(p.stem.rsplit("_", 1)[-1]), default=None)
