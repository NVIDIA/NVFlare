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


"""Regression checks for timing patterns that previously escaped CI review."""

import pytest

from ci.check_unit_test_contracts import check_source


@pytest.mark.parametrize(
    "timer", ["import threading as threads; threads.Timer", "from threading import Timer as Later; Later"]
)
def test_timer_alias_cannot_bypass_the_interleaving_guard(timer):
    import_line, constructor = timer.split("; ")
    source = f"{import_line}\ndef test_abort():\n    {constructor}(0.3, abort).start()\n"
    assert "timer-controlled" in check_source(source, "tests/unit_test/sample_test.py")[0][1]


def test_isolating_one_clock_does_not_allow_patching_another():
    source = """
from tests.timing_utils import isolate_time
from unittest.mock import patch
import nvflare.first as first

def test_deadline(monkeypatch):
    isolate_time(monkeypatch, first)
    with patch("nvflare.second.time.monotonic", return_value=0):
        pass
"""
    assert "shared time-module" in check_source(source, "sample_test.py")[0][1]


def test_local_clock_and_bounded_readiness_wait_are_allowed():
    source = """
from tests.timing_utils import isolate_time
from unittest.mock import patch
import nvflare.subject as subject

def test_deadline(monkeypatch):
    isolate_time(monkeypatch, subject)
    with patch("nvflare.subject.time.monotonic", return_value=0):
        wait_for(lambda: ready.is_set())
"""
    assert not check_source(source, "sample_test.py")


def test_real_timer_implementation_in_a_test_double_is_not_an_interleaving():
    source = """
import threading
class FakeDispatcher:
    def send(self, timeout):
        return threading.Timer(timeout, self.expired)
"""
    assert not check_source(source, "sample.py")


def test_monkeypatch_dotted_target_cannot_mutate_shared_clock():
    source = """
def test_deadline(monkeypatch):
    monkeypatch.setattr("nvflare.subject.time.monotonic", lambda: 0)
"""
    assert "shared time-module" in check_source(source, "sample_test.py")[0][1]


def test_local_isolation_does_not_excuse_another_test():
    source = """
from tests.timing_utils import isolate_time

def test_isolated(monkeypatch):
    isolate_time(monkeypatch, "nvflare.subject")

def test_unisolated(monkeypatch):
    monkeypatch.setattr("nvflare.subject.time.monotonic", lambda: 0)
"""
    assert len(check_source(source, "sample_test.py")) == 1


@pytest.mark.parametrize("decorator", ["@pytest.fixture", "@pytest.fixture(autouse=True)"])
def test_clock_mutations_inside_fixtures_are_checked(decorator):
    source = f"""
import pytest
from unittest.mock import patch
{decorator}
def clock():
    with patch("nvflare.subject.time.sleep", return_value=None):
        yield

def test_uses_clock(clock):
    pass
"""
    assert "shared time-module" in check_source(source, "sample_test.py")[0][1]


def test_guard_keeps_pytest_clock_checks_outside_standalone_contract_tree(tmp_path, monkeypatch, capsys):
    from ci import check_unit_test_contracts as checker

    source = 'def test_clock(monkeypatch):\n    monkeypatch.setattr("subject.time.sleep", lambda _: None)\n'
    regular = tmp_path / "tests/unit_test/app_opt/confidential_computing/clock_test.py"
    standalone = tmp_path / "tests/unit_test/lighter/cc/image_builder/test_clock.py"
    for path in (regular, standalone):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    monkeypatch.setattr(checker, "__file__", str(tmp_path / "ci/check_unit_test_contracts.py"))

    assert checker.main() == 1
    output = capsys.readouterr().out
    assert "confidential_computing/clock_test.py" in output
    assert "image_builder/test_clock.py" not in output


@pytest.mark.parametrize(
    "imports,launch",
    [
        ("import subprocess", "subprocess.run(['bash', 'publish.sh'], timeout=15)"),
        ("import subprocess as proc", "proc.Popen(['python3', 'helper.py'])"),
        ("from subprocess import run as execute", "execute(['opa', 'eval'])"),
        ("import os", "os.system('bash publish.sh')"),
    ],
)
def test_coco_process_launches_are_checked_in_helpers_and_fixtures(imports, launch):
    source = f"{imports}\nimport pytest\n@pytest.fixture\ndef helper():\n    {launch}\n"
    path = "tests/unit_test/lighter/cc_provision/impl/new_workflow_test.py"
    assert "CoCo unit process launch" in check_source(source, path)[0][1]


def test_coco_process_guard_allows_a_mocked_component_invocation():
    source = """
from unittest.mock import patch
def test_build():
    with patch('nvflare.component.subprocess.run') as command:
        component.build()
    command.assert_called_once()
"""
    assert not check_source(source, "tests/unit_test/lighter/cc_provision/impl/build_test.py")
