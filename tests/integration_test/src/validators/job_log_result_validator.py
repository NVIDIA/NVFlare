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

import os
from typing import Iterable, Optional

from .job_result_validator import FinishJobResultValidator


class JobLogResultValidator(FinishJobResultValidator):
    """Validate client logs streamed into the server's downloaded job result."""

    def __init__(
        self,
        log_file_name: str = "log.json",
        required_patterns: Optional[Iterable[str]] = None,
    ):
        super().__init__()
        if not log_file_name or os.path.basename(log_file_name) != log_file_name:
            raise ValueError("log_file_name must be a file name without directory components")
        self.log_file_name = log_file_name
        self.required_patterns = list(required_patterns or [])

    def validate_finished_results(self, job_result, client_props) -> bool:
        workspace_root = job_result["workspace_root"]

        for client_prop in client_props:
            client_log = self._find_client_log(workspace_root, client_prop.name)
            if not client_log:
                self.logger.error(
                    f"streamed log {client_prop.name}/{self.log_file_name} doesn't exist under {workspace_root}."
                )
                return False

            try:
                with open(client_log, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as ex:
                self.logger.error(f"failed to read streamed log {client_log}: {ex}")
                return False

            if not content.strip():
                self.logger.error(f"streamed log {client_log} is empty.")
                return False

            missing_patterns = []
            for pattern in self.required_patterns:
                expected = pattern.replace("{client}", client_prop.name)
                if expected not in content:
                    missing_patterns.append(expected)

            if missing_patterns:
                self.logger.error(f"streamed log {client_log} is missing required patterns: {missing_patterns}")
                return False

        return True

    def _find_client_log(self, workspace_root: str, client_name: str):
        expected_suffix = os.path.join(client_name, self.log_file_name)
        expected_path = os.path.join(workspace_root, expected_suffix)
        if os.path.isfile(expected_path):
            return expected_path

        # Downloaded job results can contain an additional workspace/app level.
        for root, _, files in os.walk(workspace_root):
            if self.log_file_name not in files:
                continue
            candidate = os.path.join(root, self.log_file_name)
            candidate_rel = os.path.normpath(os.path.relpath(candidate, workspace_root))
            if candidate_rel == expected_suffix or candidate_rel.endswith(os.sep + expected_suffix):
                return candidate

        return None
