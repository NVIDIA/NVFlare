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


class LogPatternValidator(FinishJobResultValidator):
    """Validate required/forbidden patterns in client job logs."""

    def __init__(
        self,
        required_patterns: Optional[Iterable[str]] = None,
        forbidden_patterns: Optional[Iterable[str]] = None,
        file_extensions: Optional[Iterable[str]] = None,
    ):
        super().__init__()
        self.required_patterns = list(required_patterns or [])
        self.forbidden_patterns = list(forbidden_patterns or [])
        self.file_extensions = tuple(file_extensions or [".txt", ".log"])

    def _iter_log_files(self, run_dir: str):
        for root, _, files in os.walk(run_dir):
            for file_name in files:
                if file_name.endswith(self.file_extensions):
                    yield os.path.join(root, file_name)

    def validate_finished_results(self, job_result, client_props) -> bool:
        job_id = job_result["job_id"]
        for client_prop in client_props:
            client_run_dir = os.path.join(client_prop.root_dir, job_id)
            required_hits = {p: False for p in self.required_patterns}
            saw_log_file = False

            for log_file in self._iter_log_files(client_run_dir):
                saw_log_file = True
                try:
                    with open(log_file, "r", errors="ignore") as f:
                        content = f.read()
                except Exception as e:
                    self.logger.error(f"failed reading log file {log_file}: {e}")
                    return False

                for pattern in self.forbidden_patterns:
                    if pattern in content:
                        self.logger.error(
                            f"forbidden pattern '{pattern}' found in client '{client_prop.name}' log file: {log_file}"
                        )
                        return False

                for pattern in self.required_patterns:
                    if pattern in content:
                        required_hits[pattern] = True

            if not saw_log_file:
                self.logger.error(f"no log files found in client run dir: {client_run_dir}")
                return False

            missing = [p for p, hit in required_hits.items() if not hit]
            if missing:
                self.logger.error(
                    f"missing required log patterns for client '{client_prop.name}' in {client_run_dir}: {missing}"
                )
                return False

        return True
