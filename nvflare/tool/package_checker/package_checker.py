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
from subprocess import TimeoutExpired

from nvflare.tool.package_checker.check_rule import CHECK_PASSED, CheckResult, CheckRule
from nvflare.tool.package_checker.utils import run_command_in_subprocess, split_by_len


class PackageChecker(ABC):
    def __init__(self):
        self.report = defaultdict(list)
        self.check_len = len("Checks")
        self.problem_len = 80
        self.fix_len = len("How to fix")
        self.dry_run_timeout = 5
        self.package_path = None
        self.rules = []

    @abstractmethod
    def init_rules(self, package_path: str):
        pass

    def init(self, package_path: str):
        if not os.path.exists(package_path):
            raise RuntimeError(f"Package path: {package_path} does not exist.")
        self.package_path = package_path
        self.init_rules(package_path)

    @abstractmethod
    def should_be_checked(self) -> bool:
        """Check if this package should be checked by this checker."""
        pass

    @abstractmethod
    def get_dry_run_command(self) -> str:
        pass

    def stop_dry_run(self):
        process = run_command_in_subprocess(f"pkill -9 -f '{self.package_path}'")
        process.wait()

    def check(self):
        """Checks if the package is runnable on the current system."""
        try:
            all_passed = True
            for rule in self.rules:
                if isinstance(rule, CheckRule):
                    result: CheckResult = rule(self.package_path, data=None)
                    self.add_report(rule.name, result.problem, result.solution)
                    if result.problem != CHECK_PASSED:
                        all_passed = False
                elif isinstance(rule, list):
                    result = CheckResult()
                    # ordered rules
                    for r in rule:
                        result = r(self.package_path, data=result.data)
                        self.add_report(r.name, result.problem, result.solution)
                        if result.problem != CHECK_PASSED:
                            all_passed = False
                            break

            # check if server can run
            if all_passed:
                self.check_dry_run(timeout=self.dry_run_timeout)

        except Exception as e:
            self.add_report(
                "Package Error",
                f"Exception happens in checking: {e}, this package is not in correct format.",
                "Please download a new package.",
            )

    def check_dry_run(self, timeout: int):
        command = self.get_dry_run_command()
        process = run_command_in_subprocess(command)
        try:
            out, _ = process.communicate(timeout=timeout)
            self.add_report(
                "Check dry run",
                f"Can't start successfully: {out}",
                "Please check the error message of dry run.",
            )
        except TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            self.add_report(
                "Check dry run",
                CHECK_PASSED,
                "N/A",
            )
        finally:
            if process:
                self.stop_dry_run()

    def add_report(self, check_name, problem_text: str, fix_text: str):
        self.report[self.package_path].append((check_name, problem_text, fix_text))
        self.check_len = max(self.check_len, len(check_name))
        self.fix_len = max(self.fix_len, len(fix_text))

    def _print_line(self):
        print("|" + "-" * (self.check_len + self.problem_len + self.fix_len + 8) + "|")

    def _print_row(self, check, problem, fix):
        print(
            "| {check:<{width1}s} | {problems:<{width2}s} | {fix:<{width3}s} |".format(
                check=check,
                problems=problem,
                fix=fix,
                width1=self.check_len,
                width2=self.problem_len,
                width3=self.fix_len,
            )
        )

    def print_report(self):
        total_width = self.check_len + self.problem_len + self.fix_len + 10
        for package_path, results in self.report.items():
            print("Checking Package: " + package_path)
            print("-" * total_width)
            if results:
                self._print_row("Checks", "Problems", "How to fix")
            else:
                print("| {:{}s} |".format("Passed", total_width - 4))
            for row in results:
                self._print_line()
                lines = split_by_len(row[1], max_len=self.problem_len)
                self._print_row(row[0], lines[0], row[2])
                for line in lines[1:]:
                    self._print_row("", line, "")
            print("-" * total_width)
            print()
