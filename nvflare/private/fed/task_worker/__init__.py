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

"""Runtime contracts for one-assignment application task workers."""

from .artifacts import ArtifactReference, FileTaskArtifactStore, IncompleteTaskArtifactError, TaskCompletion
from .protocol import ContextProperty, TaskAttemptIdentity, WorkerBootstrap, read_bootstrap, write_bootstrap

__all__ = [
    "ArtifactReference",
    "ContextProperty",
    "FileTaskArtifactStore",
    "IncompleteTaskArtifactError",
    "TaskAttemptIdentity",
    "TaskCompletion",
    "WorkerBootstrap",
    "read_bootstrap",
    "write_bootstrap",
]
