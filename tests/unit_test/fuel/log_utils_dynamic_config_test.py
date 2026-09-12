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

import io
import json
import logging
import sys
from unittest.mock import patch

import pytest


def test_dynamic_log_config_accepts_inline_json_string(tmp_path):
    from nvflare.fuel.utils.log_utils import dynamic_log_config

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {"console": {"class": "logging.StreamHandler", "level": "INFO"}},
        "root": {"handlers": ["console"], "level": "INFO"},
    }

    with patch("nvflare.fuel.utils.log_utils.apply_log_config") as mock_apply:
        dynamic_log_config(json.dumps(config), str(tmp_path), str(tmp_path / "reload.json"))

    mock_apply.assert_called_once()
    assert mock_apply.call_args.args[0] == config


def test_dynamic_log_config_invalid_inline_json_raises_value_error(tmp_path):
    from nvflare.fuel.utils.log_utils import dynamic_log_config

    with pytest.raises(ValueError, match="Invalid dictConfig JSON"):
        dynamic_log_config('{"version": 1,', str(tmp_path), str(tmp_path / "reload.json"))


def test_log_modes_use_concise_without_an_extra_mode():
    from nvflare.fuel.utils.log_utils import LogMode, logmode_config_dict

    assert logmode_config_dict[LogMode.CONCISE]["handlers"]["consoleHandler"]["filters"] == ["ProgressFilter"]
    assert "progress" not in logmode_config_dict
    assert logmode_config_dict[LogMode.MSG_ONLY]["formatters"]["consoleFormatter"]["fmt"] == "%(message)s"
    assert logmode_config_dict[LogMode.MSG_ONLY]["handlers"]["consoleHandler"]["filters"] == ["ConciseFilter"]
    assert logmode_config_dict[LogMode.FULL]["filters"]["FLFilter"]["()"] == (
        "nvflare.fuel.utils.log_utils.LoggerNameFilter"
    )
    assert logmode_config_dict[LogMode.FULL]["handlers"]["FLFileHandler"]["filters"] == ["FLFilter"]


@pytest.mark.parametrize(
    "logger_name,level,expected",
    [
        ("trainer.SimpleTrainer", logging.INFO, True),
        ("torch.distributed", logging.INFO, True),
        ("custom.trainer", logging.INFO, True),
        ("nvflare_custom.trainer", logging.INFO, True),
        ("nvflare.app_common.workflows.fedavg", logging.INFO, True),
        ("nvflare.app_opt.tracking", logging.INFO, True),
        ("nvflare.private.fed.client", logging.INFO, False),
        ("nvflare.fuel.f3.cellnet", logging.INFO, False),
        ("nvflare.private.fed.client", logging.WARNING, True),
    ],
)
def test_concise_log_filter(logger_name, level, expected):
    from nvflare.fuel.utils.log_utils import ConciseLogFilter

    log_filter = ConciseLogFilter(logger_names=["custom", "nvflare.app_common", "nvflare.app_opt"])
    record = logging.LogRecord(logger_name, level, __file__, 1, "message", (), None)

    assert log_filter.filter(record) is expected


def test_color_formatter_omits_ansi_when_stdout_is_not_tty(monkeypatch):
    from nvflare.fuel.utils.log_utils import ColorFormatter

    monkeypatch.setattr(sys, "stdout", io.StringIO())
    formatter = ColorFormatter("%(message)s")
    record = logging.LogRecord("nvflare.test", logging.INFO, __file__, 1, "hello", (), None)

    assert formatter.format(record) == "hello"


@pytest.mark.parametrize("mode", ["msg_only"])
def test_concise_console_keeps_client_output_and_errors_but_filters_bookkeeping(mode):
    from nvflare.fuel.utils.log_utils import ConciseLogFilter, logmode_config_dict

    config = logmode_config_dict[mode]
    filter_config = {k: v for k, v in config["filters"]["ConciseFilter"].items() if k != "()"}
    log_filter = ConciseLogFilter(**filter_config)
    output = io.StringIO()
    console = logging.StreamHandler(output)
    console.addFilter(log_filter)
    diagnostic_output = io.StringIO()
    diagnostic_handler = logging.StreamHandler(diagnostic_output)
    records = [
        ("nvflare.app_common.np.np_downloader.ArrayDownloadable", logging.INFO, "transfer detail"),
        ("nvflare.app_common.executors.client_api_executor.ClientAPIExecutor", logging.INFO, "executor detail"),
        ("__main__.ClientTaskWorker", logging.INFO, "worker detail"),
        ("nvflare.app_common.executors.task_script_runner.TaskScriptRunner", logging.INFO, "user training output"),
        ("nvflare.app_common.np.np_downloader.ArrayDownloadable", logging.WARNING, "transfer problem"),
        ("nvflare.app_common.executors.client_api_executor.ClientAPIExecutor", logging.ERROR, "executor failure"),
    ]
    for name, level, message in records:
        record = logging.LogRecord(name, level, __file__, 1, message, (), None)
        console.handle(record)
        diagnostic_handler.handle(record)
        assert message in diagnostic_output.getvalue()
    assert output.getvalue().splitlines() == ["user training output", "transfer problem", "executor failure"]
    # Only the console uses the concise filter; the actual file configuration
    # continues to retain the records suppressed above.
    assert not config["handlers"]["logFileHandler"].get("filters")
    assert not config["handlers"]["jsonFileHandler"].get("filters")
    assert logmode_config_dict["full"]["handlers"]["consoleHandler"]["filters"] == []


def test_color_formatter_emits_ansi_when_stdout_is_tty(monkeypatch):
    from nvflare.fuel.utils.log_utils import ColorFormatter

    class TTYStringIO(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdout", TTYStringIO())
    formatter = ColorFormatter("%(message)s")
    record = logging.LogRecord("nvflare.test", logging.INFO, __file__, 1, "hello", (), None)

    formatted = formatter.format(record)
    assert formatted.startswith("\x1b[")
    assert formatted.endswith("\x1b[0m")
    assert "hello" in formatted


def test_validate_site_log_config_accepts_levels_and_modes():
    from nvflare.fuel.utils.log_utils import LogMode, validate_site_log_config

    assert validate_site_log_config("INFO") == "INFO"
    assert validate_site_log_config("20") == "20"
    assert validate_site_log_config(LogMode.MSG_ONLY) == LogMode.MSG_ONLY


def test_validate_site_log_config_rejects_dicts_and_file_paths():
    from nvflare.fuel.utils.log_utils import validate_site_log_config

    with pytest.raises(ValueError, match="configure_site_log only supports log levels and built-in log modes"):
        validate_site_log_config({"version": 1})

    with pytest.raises(ValueError, match="configure_site_log only supports log levels and built-in log modes"):
        validate_site_log_config("/my workspace/log.conf")


def test_progress_view_retains_diagnostic_records_and_wraps_readable_output():
    from nvflare.fuel.utils.log_utils import LoggerNameFilter, ProgressFormatter, logmode_config_dict

    view = io.StringIO()
    handler = logging.StreamHandler(view)
    handler.setFormatter(ProgressFormatter())
    filter_config = logmode_config_dict["concise"]["filters"]["ProgressFilter"].copy()
    assert filter_config.pop("()") == "nvflare.fuel.utils.log_utils.LoggerNameFilter"
    handler.addFilter(LoggerNameFilter(**filter_config))
    detail = io.StringIO()
    diagnostic = logging.StreamHandler(detail)
    records = [
        logging.LogRecord("custom.trainer", logging.INFO, "", 0, "raw weights: [1, 2, 3]", (), None),
        logging.LogRecord("nvflare.metrics.progress", logging.INFO, "", 0, "site-1 | loss=0.25", (), None),
        logging.LogRecord(
            "nvflare.transport",
            logging.WARNING,
            "",
            0,
            "[identity=site-2, run=job-123]: connection interrupted " + "details " * 20,
            (),
            None,
        ),
    ]
    for record in records:
        handler.handle(record)
        diagnostic.handle(record)
    assert "raw weights" not in view.getvalue()
    assert "site-1 | loss=0.25" in view.getvalue()
    assert "WARNING (site-2/transport): connection interrupted" in view.getvalue()
    assert "run=job-123" not in view.getvalue()
    assert all(len(line) <= 80 for line in view.getvalue().splitlines())
    assert "raw weights" in detail.getvalue()
    assert "[identity=site-2, run=job-123]" in detail.getvalue()
    config = logmode_config_dict["concise"]
    for name in ("consoleHandler", "FLFileHandler"):
        assert config["handlers"][name]["filters"] == ["ProgressFilter"]
        assert config["handlers"][name]["formatter"] == "progressFormatter"
    for name in ("logFileHandler", "jsonFileHandler"):
        assert config["handlers"][name] == logmode_config_dict["full"]["handlers"][name]


def test_metric_formatting_cannot_propagate_application_object_errors():
    from nvflare.fuel.utils.log_utils import format_metric_summary

    class BrokenNumber(float):
        def __format__(self, spec):
            raise RuntimeError("application formatting failed")

    class Unprintable:
        def __str__(self):
            raise RuntimeError("must not stringify application objects")

    assert format_metric_summary({"bad": BrokenNumber(1)}) == "[see saved result for metrics]"
    assert format_metric_summary({"value": Unprintable()}) == "value=[see saved result]"


def test_metric_table_keeps_columns_aligned_through_log_formatting():
    from nvflare.fuel.utils.log_utils import ProgressFormatter, format_metric_table

    rows = [
        ("site-1", {"accuracy": 1, "accuracy_after_local_training": 20}),
        ("site-2", {"accuracy": 30, "accuracy_after_local_training": 70}),
    ]
    table = format_metric_table(rows)
    record = logging.LogRecord("nvflare.metrics.progress", logging.INFO, "", 0, table, (), None)
    output = ProgressFormatter().format(record)
    assert output == table  # Wrapping must not collapse table alignment or merge rows.
    header, first, second = output.splitlines()
    assert first.split() == ["site-1", "1", "20"]
    assert second.split() == ["site-2", "30", "70"]
    assert first.rindex("20") == second.rindex("70")
    assert len(header) <= 80


def test_metric_table_bounds_display_and_handles_application_values():
    from nvflare.fuel.utils.log_utils import format_metric_table

    class BadNumber(float):
        def __format__(self, spec):
            raise RuntimeError("application-defined formatting failure")

    rows = [("site\n" + "x" * 100, {"accuracy": BadNumber(1), "samples": list(range(10000)), "extra": 5})] * 100
    table = format_metric_table(rows)
    assert "[see artifact]" in table
    assert "saved artifacts" in table
    assert "9999" not in table
    assert len(table.splitlines()) == 12  # Header, ten rows, one truncation notice.
    assert all(len(line) <= 80 for line in table.splitlines())
    assert "site\\n" in table


def test_metric_table_missing_values_are_not_reported_as_zero():
    from nvflare.fuel.utils.log_utils import format_metric_table

    table = format_metric_table([("site-1", {"loss": 0.25}), ("site-2", {"accuracy": 1})])
    assert table.splitlines()[1].split() == ["site-1", "0.25", "—"]
    assert table.splitlines()[2].split() == ["site-2", "—", "1"]


@pytest.mark.parametrize("whole_lines", [False, True])
@pytest.mark.parametrize("size", [0, 5, 16, 4096])
def test_shared_log_tail_bounds_reads_and_preserves_byte_or_record_contract(size, whole_lines):
    from nvflare.fuel.utils.log_utils import _read_log_tail

    class TrackedStream(io.BytesIO):
        bytes_read = 0

        def read(self, size=-1):
            data = super().read(size)
            self.bytes_read += len(data)
            return data

    data = (b"old record\n" * size) + b"final record\n" if size else b""
    stream = TrackedStream(data)
    tail, truncated = _read_log_tail(stream, 32, whole_lines=whole_lines)
    expected = data[-32:]
    if len(data) > 32 and whole_lines:
        expected = expected.partition(b"\n")[2]
    assert tail == expected
    assert truncated == (len(data) > 32)
    assert stream.bytes_read <= 32
