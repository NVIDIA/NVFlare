# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import inspect
import logging
import logging.config
import os
from logging import Logger
from logging.handlers import RotatingFileHandler

from nvflare.apis.workspace import Workspace


class ANSIColor:
    GREY = "38"
    YELLOW = "33"
    RED = "31"
    BOLD_RED = "31;1"
    CYAN = "36"
    RESET = "0"


DEFAULT_LEVEL_COLORS = {
    "DEBUG": ANSIColor.GREY,
    "INFO": ANSIColor.GREY,
    "WARNING": ANSIColor.YELLOW,
    "ERROR": ANSIColor.RED,
    "CRITICAL": ANSIColor.BOLD_RED,
}


class BaseFormatter(logging.Formatter):
    def __init__(self, fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt=None, style="%"):
        """BaseFormatter is the default formatter for log records.

        Shortens logger %(name)s to the suffix, full name can be accessed with %(fullName)s

        Args:
            fmt: format string which uses LogRecord attributes.
            datefmt: date/time format string. Defaults to '%Y-%m-%d %H:%M:%S'.
            style: style character '%' '{' or '$' for format string.

        """
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)

    def format(self, record):
        if not hasattr(record, "fullName"):
            record.fullName = record.name
            record.name = record.name.split(".")[-1]

        return super().format(record)


def ansi_sgr(code):
    # ANSI Select Graphics Rendition
    return "\x1b[" + code + "m"


def get_module_logger(module=None, name=None):
    if module is None:
        caller_globals = inspect.stack()[1].frame.f_globals
        module = caller_globals.get("__name__", "")

    return logging.getLogger(f"{module}.{name}" if name else module)


def get_obj_logger(obj):
    return logging.getLogger(f"{obj.__module__}.{obj.__class__.__qualname__}")


def get_script_logger():
    caller_frame = inspect.stack()[1]
    package = caller_frame.frame.f_globals.get("__package__", "")
    file = caller_frame.frame.f_globals.get("__file__", "")

    return logging.getLogger(
        f"{package + '.' if package else ''}{os.path.splitext(os.path.basename(file))[0] if file else ''}"
    )


def configure_logging(workspace: Workspace):
    log_config_file_path = workspace.get_log_config_file_path()
    assert os.path.isfile(log_config_file_path), f"missing log config file {log_config_file_path}"
    logging.config.fileConfig(fname=log_config_file_path, disable_existing_loggers=False)


def add_log_file_handler(log_file_name):
    root_logger = logging.getLogger()
    main_handler = root_logger.handlers[0]
    file_handler = RotatingFileHandler(log_file_name, maxBytes=20 * 1024 * 1024, backupCount=10)
    file_handler.setLevel(main_handler.level)
    file_handler.setFormatter(main_handler.formatter)
    root_logger.addHandler(file_handler)


def print_logger_hierarchy(package_name="nvflare", level_colors=DEFAULT_LEVEL_COLORS):
    all_loggers = logging.root.manager.loggerDict

    # Filter for package loggers based on package_name
    package_loggers = {name: logger for name, logger in all_loggers.items() if name.startswith(package_name)}
    sorted_package_loggers = sorted(package_loggers.keys())

    # Print package loggers with hierarcjy
    print(f"hierarchical loggers ({len(package_loggers)}):")

    def get_effective_level(logger_name):
        # Search for effective level from parent loggers
        parts = logger_name.split(".")
        for i in range(len(parts), 0, -1):
            parent_name = ".".join(parts[:i])
            parent_logger = package_loggers.get(parent_name)
            if isinstance(parent_logger, Logger) and parent_logger.level != logging.NOTSET:
                return logging.getLevelName(parent_logger.level)

        # If no parent has a set level, default to the root logger's effective level
        return logging.getLevelName(logging.root.level)

    def print_hierarchy(logger_name, indent_level=0):
        logger = package_loggers.get(logger_name)
        level_name = get_effective_level(logger_name)

        # Indicate "(unset)" placeholders if logger.level == NOTSET
        is_unset = isinstance(logger, Logger) and logger.level == logging.NOTSET or not isinstance(logger, Logger)
        level_display = f"{level_name} (SET)" if not is_unset else level_name

        # Print the logger with color and indentation
        color = level_colors.get(level_name, ANSIColor.RESET)
        print("    " * indent_level + f"{ansi_sgr(color)}{logger_name} [{level_display}]{ansi_sgr(ANSIColor.RESET)}")

        # Find child loggers based on the current hierarchy level
        for name in sorted_package_loggers:
            if name.startswith(logger_name + ".") and name.count(".") == logger_name.count(".") + 1:
                print_hierarchy(name, indent_level + 1)

    print_hierarchy(package_name)
