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

import logging
from logging.handlers import RotatingFileHandler
from unittest.mock import Mock

import pytest

from nvflare.apis.job_def import RunStatus
from nvflare.fuel.flare_api.api_spec import InternalError, JobNotRunning
from tests.integration_test.src import nvf_test_driver
from tests.timing_utils import ManualClock

JOB_ID = "12345678-1234-5678-1234-567812345678"
WORKFLOWS = {"AbortTestController": {"phase": "waiting_for_release"}}
STATS = {"server": {"AbortTestController": {"tasks": {}, **WORKFLOWS["AbortTestController"]}}}


@pytest.fixture
def driver(monkeypatch):
    clock = ManualClock()
    sleeps = []

    def sleep(delay):
        assert len(sleeps) < 20, "event polling did not stop at its deadline"
        sleeps.append(delay)
        clock.advance(delay)

    clock.sleep = sleep
    monkeypatch.setattr(nvf_test_driver, "time", clock)
    driver = nvf_test_driver.NVFTestDriver("unused", Mock(), poll_period=0.5, event_sequence_timeout=2)
    driver.job_id = JOB_ID
    driver.super_admin_api = Mock()
    driver.super_admin_api.get_job_meta.return_value = {"status": RunStatus.RUNNING.value}
    driver.super_admin_api.cat_target.return_value = "Server runner starting ...\n"
    driver.super_admin_api.ls_target.return_value = ""
    driver.super_admin_api.show_stats.return_value = STATS
    driver.server_status = Mock(return_value="started")
    driver.client_status = Mock(return_value=[])
    return driver


def _state(status=RunStatus.RUNNING.value):
    return {"job_status": status, "job_terminal": False, "run_finished": False, "workflows": None}


def _event(data):
    return {
        "trigger": {"type": "run_state", "data": data},
        "actions": ["mark_test_done"],
        "result": {"type": "run_state", "data": {}},
    }


def test_completion_polling_never_contacts_the_job_process(driver):
    driver.super_admin_api.get_job_meta.side_effect = [
        {"status": status}
        for status in (
            RunStatus.DISPATCHED.value,
            RunStatus.RUNNING.value,
            RunStatus.FINISHED_COMPLETED.value,
            RunStatus.FINISHED_COMPLETED.value,
        )
    ]
    driver.super_admin_api.show_stats.side_effect = AssertionError("queried an uninitialized or exited job")

    driver.run_event_sequence([_event({"run_finished": True})])

    assert driver.test_done
    driver.super_admin_api.show_stats.assert_not_called()
    driver.super_admin_api.cat_target.assert_not_called()


def test_workflow_polling_waits_for_the_current_jobs_startup(driver):
    log_reads = []

    def job_log(target, file):
        assert file == f"{JOB_ID}/log.txt"
        text = ["No such file", "Loading server app", "Server runner starting ..."][len(log_reads)]
        log_reads.append(text)
        return text

    def stats(*args, **kwargs):
        assert len(log_reads) == 3, "show_stats reached the job before its engine was ready"
        return STATS

    driver.super_admin_api.cat_target.side_effect = job_log
    driver.super_admin_api.show_stats.side_effect = stats

    driver.run_event_sequence([_event({"job_status": RunStatus.RUNNING.value, "workflows": WORKFLOWS})])

    assert driver.test_done
    assert len(log_reads) == 3
    driver.super_admin_api.show_stats.assert_called_once_with(JOB_ID, "server")


@pytest.mark.parametrize("status", [RunStatus.DISPATCHED.value, RunStatus.FINISHED_COMPLETED.value])
def test_workflow_stats_are_not_requested_for_non_running_jobs(driver, status):
    driver._get_workflow_state(_state(status))
    driver.super_admin_api.cat_target.assert_not_called()
    driver.super_admin_api.show_stats.assert_not_called()


def test_missing_readiness_cannot_reuse_previous_workflow_state(driver):
    driver.super_admin_api.cat_target.return_value = "Loading a new server app"
    state = _state()
    state["workflows"] = WORKFLOWS

    assert driver._get_workflow_state(state)["workflows"] is None
    driver.super_admin_api.show_stats.assert_not_called()


def test_a_job_that_exits_after_readiness_is_polled_again_via_metadata(driver):
    driver.super_admin_api.show_stats.side_effect = JobNotRunning("job exited")
    assert driver._get_workflow_state(_state())["workflows"] is None
    driver.super_admin_api.get_job_meta.return_value = {"status": RunStatus.FINISHED_COMPLETED.value}
    assert driver._get_run_state(_state())["run_finished"] is True


def test_unrelated_stats_errors_are_not_suppressed(driver):
    driver.super_admin_api.show_stats.side_effect = InternalError("broken stats response")
    with pytest.raises(InternalError, match="broken stats response"):
        driver._get_workflow_state(_state())


def test_failed_job_does_not_wait_for_readiness_or_the_event_deadline(driver):
    driver.super_admin_api.get_job_meta.return_value = {"status": RunStatus.FINISHED_ABNORMAL.value}
    with pytest.raises(nvf_test_driver.NVFTestError, match="terminal status 'FINISHED:ABNORMAL'"):
        driver.run_event_sequence([_event({"job_status": RunStatus.RUNNING.value, "workflows": WORKFLOWS})])
    driver.super_admin_api.cat_target.assert_not_called()
    driver.super_admin_api.show_stats.assert_not_called()


def test_readiness_that_never_arrives_fails_at_the_existing_event_deadline(driver):
    driver.super_admin_api.cat_target.return_value = "Loading server app"
    with pytest.raises(nvf_test_driver.NVFTestError, match="Event sequence timed out"):
        driver.run_event_sequence([_event({"workflows": WORKFLOWS})])
    assert driver.super_admin_api.cat_target.call_count == 6
    driver.super_admin_api.show_stats.assert_not_called()


def test_workflow_assertions_in_event_results_still_collect_statistics(driver):
    event = _event({"job_status": RunStatus.RUNNING.value})
    event["result"]["data"] = {"workflows": WORKFLOWS}
    driver.run_event_sequence([event])
    assert driver.test_done
    driver.super_admin_api.show_stats.assert_called_once_with(JOB_ID, "server")


@pytest.fixture
def rotating_job_log(driver, tmp_path):
    job_dir = tmp_path / JOB_ID
    job_dir.mkdir()
    logger = logging.Logger("readiness-rotation", level=logging.INFO)
    handler = RotatingFileHandler(job_dir / "log.txt", maxBytes=128, backupCount=2)
    logger.addHandler(handler)

    def cat_log(target, file):
        assert target == "server"
        path = tmp_path / file
        return path.read_text() if path.is_file() else "No such file"

    def list_logs(target, path):
        assert target == "server"
        folder = tmp_path / path
        return "\n".join(p.name for p in folder.iterdir()) if folder.is_dir() else "No such directory"

    driver.super_admin_api.cat_target.side_effect = cat_log
    driver.super_admin_api.ls_target.side_effect = list_logs
    try:
        yield logger, handler, job_dir
    finally:
        logger.removeHandler(handler)
        handler.close()


def test_observed_readiness_survives_rotation_and_expired_backups(driver, rotating_job_log):
    logger, handler, job_dir = rotating_job_log
    logger.info("Server runner starting ...")
    assert driver._get_workflow_state(_state())["workflows"] == WORKFLOWS
    # Rotate enough times that even the retained backups no longer hold startup.
    for _ in range(3):
        handler.doRollover()
        logger.info("Training continues")
    assert all("Server runner starting ..." not in p.read_text() for p in job_dir.iterdir())
    driver.super_admin_api.cat_target.reset_mock()
    driver.super_admin_api.ls_target.reset_mock()

    driver.run_event_sequence([_event({"workflows": WORKFLOWS})])

    assert driver.test_done
    assert driver.super_admin_api.show_stats.call_count == 2
    driver.super_admin_api.cat_target.assert_not_called()
    driver.super_admin_api.ls_target.assert_not_called()


@pytest.mark.parametrize("rotations", [1, 2])
def test_first_workflow_poll_finds_startup_in_rotated_log(driver, rotating_job_log, rotations):
    logger, handler, job_dir = rotating_job_log
    logger.info("Server runner starting ...")
    for _ in range(rotations):
        handler.doRollover()
        logger.info("Training continues")
    assert "Server runner starting ..." not in (job_dir / "log.txt").read_text()
    assert "Server runner starting ..." in (job_dir / f"log.txt.{rotations}").read_text()

    driver.run_event_sequence([_event({"workflows": WORKFLOWS})])

    assert driver.test_done
    driver.super_admin_api.show_stats.assert_called_once_with(JOB_ID, "server")
    driver.super_admin_api.ls_target.assert_called_once_with("server", path=JOB_ID)


def test_readiness_from_one_job_cannot_authorize_another_job(driver, rotating_job_log):
    logger, _handler, _job_dir = rotating_job_log
    logger.info("Server runner starting ...")
    assert driver._get_workflow_state(_state())["workflows"] == WORKFLOWS
    driver.super_admin_api.show_stats.reset_mock()
    driver.job_id = "another-job"

    with pytest.raises(nvf_test_driver.NVFTestError, match="Event sequence timed out"):
        driver.run_event_sequence([_event({"workflows": WORKFLOWS})])

    driver.super_admin_api.show_stats.assert_not_called()


def test_rotated_log_lookup_ignores_other_files_and_jobs(driver):
    driver.super_admin_api.cat_target.return_value = "Loading server app"
    driver.super_admin_api.ls_target.return_value = "log.txt\nlog.json.1\n../other/log.txt.1\nold_log.txt.1\n"

    assert driver._get_workflow_state(_state())["workflows"] is None

    driver.super_admin_api.cat_target.assert_called_once_with("server", file=f"{JOB_ID}/log.txt")
    driver.super_admin_api.show_stats.assert_not_called()
