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

import json
import os
from typing import Optional

from .job_result_validator import FinishJobResultValidator


class JsonStatsResultValidator(FinishJobResultValidator):
    def __init__(self, relative_path: str, required_paths: Optional[list[str]] = None):
        super().__init__()
        self.relative_path = relative_path
        self.required_paths = required_paths or []

    def validate_finished_results(self, job_result, client_props) -> bool:
        stats_file = self._find_stats_file(job_result["workspace_root"])
        if not stats_file:
            self.logger.error(f"stats file {self.relative_path} doesn't exist.")
            return False

        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
        except Exception as ex:
            self.logger.error(f"failed to load stats file {stats_file}: {ex}")
            return False

        if not stats:
            self.logger.error(f"stats file {stats_file} is empty.")
            return False

        for required_path in self.required_paths:
            if not self._has_path(stats, required_path):
                self.logger.error(f"stats file {stats_file} is missing required path {required_path}.")
                return False

        return True

    def _find_stats_file(self, workspace_root: str):
        normalized_suffix = os.path.normpath(self.relative_path)
        if os.path.isabs(normalized_suffix) or normalized_suffix == os.pardir:
            return None
        if normalized_suffix.startswith(os.pardir + os.sep):
            return None

        stats_file = os.path.join(workspace_root, normalized_suffix)
        if os.path.isfile(stats_file):
            return stats_file

        # The downloaded job-result layout can include an extra workspace/app level;
        # keep the validator stable across those layouts.
        for root, _, files in os.walk(workspace_root):
            for file_name in files:
                candidate = os.path.join(root, file_name)
                candidate_rel = os.path.normpath(os.path.relpath(candidate, workspace_root))
                if candidate_rel == normalized_suffix or candidate_rel.endswith(os.sep + normalized_suffix):
                    return candidate

        return None

    @staticmethod
    def _has_path(stats: dict, required_path: str):
        current = stats
        for part in required_path.split("."):
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
        return True
