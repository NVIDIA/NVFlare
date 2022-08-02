# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import signal
from abc import ABC, abstractmethod
from collections import defaultdict

from nvflare.tool.package_checker.utils import run_command_in_subprocess


class PackageChecker(ABC):
    def __init__(self):
        self.report = defaultdict(list)
        self.problem_len = len("Problems")
        self.fix_len = len("How to fix")
        self.dry_run_process = None

    @abstractmethod
    def should_be_checked(self, package_path) -> bool:
        """Check if this package should be checked by this checker."""
        pass

    @abstractmethod
    def check(self, package_path):
        """Checks if the package is runnable on the current system."""
        pass

    @abstractmethod
    def dry_run(self, package_path):
        pass

    def check_package(self, package_path) -> bool:
        self.check(package_path)
        if self.report[package_path]:
            return False
        return True

    def stop_dry_run(self, package_path):
        if self.dry_run_process:
            os.killpg(self.dry_run_process.pid, signal.SIGTERM)
        process = run_command_in_subprocess(f"pkill -9 -f '{package_path}'")
        process.wait()

    def add_report(self, package_path: str, problem_text: str, fix_text: str):
        lines = problem_text.splitlines()
        self.report[package_path].append((problem_text, fix_text))
        self.problem_len = max(self.problem_len, len(lines[0]))
        self.fix_len = max(self.fix_len, len(fix_text))
        for line in lines[1:]:
            self.problem_len = max(self.problem_len, len(line))

    def _print_line(self):
        print("|" + "-" * (self.problem_len + self.fix_len + 5) + "|")

    def print_report(self):
        total_width = self.problem_len + self.fix_len + 7
        for package_path, results in self.report.items():
            print("Checking Package: " + package_path)
            print("-" * total_width)
            if results:
                print(
                    "| {problems:<{width1}s} | {fix:<{width2}s} |".format(
                        problems="Problems", fix="How to fix", width1=self.problem_len, width2=self.fix_len
                    )
                )
            else:
                print("| {:{}s} |".format("Passed", total_width - 4))
            for row in results:
                self._print_line()
                lines = row[0].splitlines()
                print(
                    "| {problems:<{width1}s} | {fix:<{width2}s} |".format(
                        problems=lines[0], fix=row[1], width1=self.problem_len, width2=self.fix_len
                    )
                )
                for line in lines[1:]:
                    print(
                        "| {problems:<{width1}s} | {fix:<{width2}s} |".format(
                            problems=line, fix="", width1=self.problem_len, width2=self.fix_len
                        )
                    )
            print("-" * total_width)
            print()
