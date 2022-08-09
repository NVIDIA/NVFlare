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

from nvflare.tool.package_checker.check_rule import CheckResult, CheckRule
from nvflare.tool.package_checker.utils import run_command_in_subprocess, try_bind_address, try_write


class PackageChecker(ABC):
    def __init__(self):
        self.report = defaultdict(list)
        self.problem_len = len("Problems")
        self.fix_len = len("How to fix")
        self.dry_run_process = None
        self.dry_run_timeout = 5
        self.package_path = None
        self.rules = []

    def init(self, package_path: str):
        if not os.path.exists(package_path):
            raise RuntimeError(f"Package path: {package_path} does not exist.")
        self.package_path = package_path
        self.init_rules(package_path)

    @abstractmethod
    def init_rules(self, package_path: str):
        pass

    @abstractmethod
    def should_be_checked(self) -> bool:
        """Check if this package should be checked by this checker."""
        pass

    def check(self):
        """Checks if the package is runnable on the current system."""
        try:
            for rule in self.rules:
                if isinstance(rule, CheckRule):
                    result: CheckResult = rule(self.package_path, data=None)
                    if result.problem:
                        self.add_report(result.problem, result.solution)
                elif isinstance(rule, list):
                    result = CheckResult()
                    # ordered rules
                    for r in rule:
                        result = r(self.package_path, data=result.data)
                        if result.problem:
                            self.add_report(result.problem, result.solution)
                            break

            # check if server can run
            if len(self.report[self.package_path]) == 0:
                self.check_dry_run(timeout=self.dry_run_timeout)

        except Exception as e:
            self.add_report(
                f"Exception happens in checking {e}, this package is not in correct format.",
                "Please download a new package.",
            )

    @abstractmethod
    def get_dry_run_command(self) -> str:
        pass

    def check_package(self) -> bool:
        self.check()
        if self.report[self.package_path]:
            return False
        return True

    def dry_run(self):
        self.dry_run_process = run_command_in_subprocess(self.get_dry_run_command())

    def stop_dry_run(self):
        if self.dry_run_process:
            os.killpg(self.dry_run_process.pid, signal.SIGTERM)
        process = run_command_in_subprocess(f"pkill -9 -f '{self.package_path}'")
        process.wait()

    def add_report(self, problem_text: str, fix_text: str):
        lines = problem_text.splitlines()
        self.report[self.package_path].append((problem_text, fix_text))
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

    def check_bind_service_address(self, host: str, port: int, service_name: str = ""):
        e = try_bind_address(host, int(port))
        if e:
            self.add_report(
                f"Can't bind to address ({host}:{port}) for service {service_name}: {e}",
                "Please check the DNS and port.",
            )

    def check_write_location(self, path_to_write: str, path_meaning: str = ""):
        e = try_write(path_to_write)
        if e:
            self.add_report(
                f"Can't write to {path_to_write} ({path_meaning}): {e}.",
                "Please check the user permission.",
            )

    def check_dry_run(self, timeout: int):
        command = self.get_dry_run_command()
        process = run_command_in_subprocess(command)
        try:
            out, _ = process.communicate(timeout=timeout)
            self.add_report(
                f"Can't start successfully: \n{out}",
                "Please check the error message of dry run.",
            )
        except TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
